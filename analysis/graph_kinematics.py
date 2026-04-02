"""
graph_kinematics.py
--------------------
Kinematic analysis: Tracking Error, Attitudes, and Cable Kinematics.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter

import formation as formation
from mission import MissionContext, equations_of_motion, update_guidance, update_dynamic_wind
import physics
from parameters import SysParams

PHASE_STYLES = {
    0: dict(color='#AAAAAA', label='Ph.0 Pretension'),
    1: dict(color='#FFD700', label='Ph.1 Liftoff'),
    2: dict(color='#90EE90', label='Ph.2 Alignment'),
    3: dict(color='#87CEEB', label='Ph.3 Attitude'),
    4: dict(color='#FFA07A', label='Ph.4 Navigation'),
    5: dict(color='#DDA0DD', label='Ph.5 Winch'),
    6: dict(color='#98FB98', label='Ph.6 Hold'),
}

def shade_phases(ax, t_res, phase_res, alpha=0.12):
    phases = np.unique(phase_res)
    for ph in phases:
        mask = phase_res == ph
        if not np.any(mask): continue
        idx = np.where(mask)[0]
        ax.axvspan(t_res[idx[0]], t_res[idx[-1]], color=PHASE_STYLES.get(ph, dict(color='white'))['color'], alpha=alpha, zorder=0)

def shade_gusts(ax, gusts):
    for g in gusts:
        ax.axvspan(g['t_start'], g['t_start'] + g['duration'], color='lightsteelblue', alpha=0.25, zorder=0)

def run_simulation():
    p = SysParams()
    p.uav_offsets, p.attach_vecs, geo_radius = formation.compute_geometry(p)
    dist_h = geo_radius - (min(p.pay_l, p.pay_w) / 2 if p.payload_shape in ['box', 'rect'] else p.R_disk)
    dist_h = max(dist_h, 0.1)
    p.safe_altitude_offset = np.sqrt(p.L ** 2 - dist_h ** 2) if p.L > dist_h else p.L

    ctx = MissionContext(p)

    uav_pos_0 = np.zeros((3, p.N))
    for i in range(p.N):
        uav_pos_0[:, i] = p.home_pos + p.uav_offsets[:, i] + np.array([0, 0, p.floor_z])
    pay_pos_0 = p.home_pos + np.array([0, 0, p.floor_z + p.pay_h / 2.0])

    x0 = np.concatenate([
        uav_pos_0.flatten('F'), pay_pos_0, np.zeros(3), np.zeros(3 * p.N),
        np.zeros(3), np.zeros(3), np.zeros(3 * p.N), np.zeros(3)
    ])

    SIM_DURATION = 80.0
    dt           = 0.02
    steps        = int(SIM_DURATION / dt)

    n_state   = len(x0)
    y_res     = np.zeros((n_state, steps))
    t_res     = np.zeros(steps)
    phase_res = np.zeros(steps, dtype=int)
    
    ref_pos_res = np.zeros((3, steps))
    ref_att_res = np.zeros((3, steps)) 
    
    # Inizializziamo a p.L per sicurezza contro gli zeri all'avvio
    L_winch_cmd_res = np.full((p.N, steps), p.L)
    target_uav_pos_res = np.zeros((3, p.N, steps))
    
    f_ext_xy_res = np.zeros(steps)
    f_inertial_xy_res = np.zeros(steps) # Aggiunto per il primo grafico

    current_x          = x0.copy()
    current_t          = 0.0
    last_guidance_time = -1.0
    guidance_dt        = p.optimization_dt
    m_tot              = p.m_payload + p.m_liquid

    print(f"Running simulation for Kinematics  N={p.N}, T={SIM_DURATION}s …")
    
    for k in range(steps):
        y_res[:, k]  = current_x
        t_res[k]     = current_t
        phase_res[k] = ctx.phase
        
        if hasattr(ctx, 'current_L_rest') and ctx.current_L_rest is not None:
            # Controllo di sicurezza: se il comando scende sotto 0.1, mantieni p.L
            L_winch_cmd_res[:, k] = np.where(ctx.current_L_rest < 0.1, p.L, ctx.current_L_rest)

        if (current_t - last_guidance_time >= guidance_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guidance_time = current_t

        if ctx.current_ref is not None:
            ref_pos_res[:, k] = ctx.current_ref['pos']
            # Calcolo Forza Inerziale
            f_inertial_xy_res[k] = np.linalg.norm(m_tot * ctx.current_ref['acc'][:2])
        else:
            ref_pos_res[:, k] = current_x[3*p.N : 3*p.N+3]
            f_inertial_xy_res[k] = 0.0
            
        if hasattr(ctx, 'current_target_att'):
            ref_att_res[:, k] = ctx.current_target_att
        else:
            ref_att_res[:, k] = ref_att_res[:, k-1] if k > 0 else np.zeros(3)

        if hasattr(ctx, 'filt_targets') and ctx.filt_targets is not None:
            target_uav_pos_res[:, :, k] = ctx.filt_targets
        else:
            target_uav_pos_res[:, :, k] = p.uav_offsets

        # Calcolo forza vento per plot 
        update_dynamic_wind(current_t, p)
        
        # 1. Somma esplicita: garantiamo che il vento sia Vento Base + Gust Attuale
        wind_total = p.initial_wind_vec.copy()
        if getattr(p, 'enable_gusts', False):
            for g in p.gust_schedule:
                if g['t_start'] <= current_t <= (g['t_start'] + g['duration']):
                    wind_total += g['vec']
        p.wind_vel = wind_total
        
        state_curr = physics.unpack_state(current_x, p)
        R_pay_curr = physics.get_rotation_matrix(*state_curr['pay_att'])
        
        # 2. Per il grafico usiamo np.zeros(3) invece di state_curr['pay_vel'].
        # Questo calcola la forza pura dell'aria sul payload (come se fosse fermo),
        # creando un'impennata perfetta e stabile sul grafico in corrispondenza del gust.
        f_aero_pure = physics.compute_payload_aero(np.zeros(3), p.wind_vel, R_pay_curr, p)
        f_ext_xy_res[k] = np.linalg.norm(f_aero_pure[:2])

        n_sub = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if k % 100 == 0:
            print(f"\r  {current_t / SIM_DURATION * 100:5.1f}%  t={current_t:.2f}s  phase={ctx.phase}", end='')

    print("\nDone")
    return t_res, y_res, phase_res, ref_pos_res, ref_att_res, L_winch_cmd_res, target_uav_pos_res, f_ext_xy_res, f_inertial_xy_res, ctx, p

def parse_state(y, p):
    N = p.N
    uav_pos  = y[: 3 * N, :].reshape(3, N, -1, order='F')     
    pay_pos  = y[3*N     : 3*N+3,  :]                          
    pay_att  = y[3*N+3   : 3*N+6,  :]                          
    return uav_pos, pay_pos, pay_att

def plot_kinematics(t_res, y_res, phase_res, ref_pos_res, ref_att_res, L_winch_cmd_res, target_uav_pos_res, f_ext_xy_res, f_inertial_xy_res, ctx, p):
    uav_pos, pay_pos, pay_att = parse_state(y_res, p)
    steps = len(t_res)

    error_vec = pay_pos - ref_pos_res
    window_len = min(51, steps)
    if window_len % 2 == 0: window_len -= 1
    error_vec_smooth = savgol_filter(error_vec, window_length=window_len, polyorder=3, axis=1) if window_len > 3 else error_vec
    pos_error_smooth = np.linalg.norm(error_vec_smooth, axis=0)

    pay_world_roll, pay_world_pitch = np.zeros(steps), np.zeros(steps)
    ref_world_roll, ref_world_pitch = np.zeros(steps), np.zeros(steps)
    pay_world_yaw = np.zeros(steps)
    ref_world_yaw = np.zeros(steps)
    ref_form_yaw = np.zeros(steps)
    form_roll, form_pitch, form_yaw = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    achievable_ref_roll, achievable_ref_pitch = np.zeros(steps), np.zeros(steps)
    
    L_actual_res = np.full((p.N, steps), p.L)

    for k in range(steps):
        # 1. Payload e Reference World Attitude
        att = pay_att[:, k]
        pay_world_roll[k], pay_world_pitch[k] = np.degrees(att[0]), np.degrees(att[1])
        pay_world_yaw[k] = np.degrees(att[2])
        
        att_ref = ref_att_res[:, k]
        ref_world_roll[k], ref_world_pitch[k] = np.degrees(att_ref[0]), np.degrees(att_ref[1])
        ref_world_yaw[k] = np.degrees(att_ref[2])
        
        # 2. Formation Attitude Reale
        curr_uavs = uav_pos[:, :, k]  
        uav_center = np.mean(curr_uavs, axis=1)
        v_head = curr_uavs[:, 0] - uav_center
        form_yaw[k] = np.degrees(np.arctan2(v_head[1], v_head[0]))
        
        uav_centered = curr_uavs - uav_center[:, None]
        u_svd, _, _ = np.linalg.svd(uav_centered)
        normal = u_svd[:, 2]
        if normal[2] < 0: normal = -normal 
            
        yaw_rad = np.radians(form_yaw[k])
        c_y, s_y = np.cos(-yaw_rad), np.sin(-yaw_rad)
        normal_unyawed = np.array([
            c_y * normal[0] - s_y * normal[1],
            s_y * normal[0] + c_y * normal[1], normal[2]
        ])
        
        form_roll[k] = np.degrees(np.arcsin(np.clip(-normal_unyawed[1], -1.0, 1.0)))
        form_pitch[k] = np.degrees(np.arctan2(normal_unyawed[0], normal_unyawed[2]))

        # 3. Target Formazione Ottimizzato
        target_uavs = target_uav_pos_res[:, :, k]
        t_center = np.mean(target_uavs, axis=1)
        
        t_v_head = target_uavs[:, 0] - t_center
        t_yaw = np.arctan2(t_v_head[1], t_v_head[0])
        ref_form_yaw[k] = np.degrees(t_yaw) 
        
        t_centered = target_uavs - t_center[:, None]
        u_svd_ref, _, _ = np.linalg.svd(t_centered)
        normal_ref = u_svd_ref[:, 2]
        if normal_ref[2] < 0: normal_ref = -normal_ref

        c_y_ref, s_y_ref = np.cos(-t_yaw), np.sin(-t_yaw)
        normal_unyawed_ref = np.array([
            c_y_ref * normal_ref[0] - s_y_ref * normal_ref[1],
            s_y_ref * normal_ref[0] + c_y_ref * normal_ref[1], 
            normal_ref[2]
        ])
        
        achievable_ref_roll[k] = np.degrees(np.arcsin(np.clip(-normal_unyawed_ref[1], -1.0, 1.0)))
        achievable_ref_pitch[k] = np.degrees(np.arctan2(normal_unyawed_ref[0], normal_unyawed_ref[2]))

        # 4. Lunghezza Fisica Cavi Reali
        R_pay = physics.get_rotation_matrix(*att)
        for i in range(p.N):
            anchor_pos = pay_pos[:, k] + (R_pay @ p.attach_vecs[:, i])
            dist = np.linalg.norm(curr_uavs[:, i] - anchor_pos)
            # Evita fisicamente che il cavo segni 0 all'inizio (usa distanza reale se > 0.1, altrimenti p.L)
            L_actual_res[i, k] = dist if dist > 0.1 else p.L

    # --- Plotting ---
    fig, axs = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    skip = max(1, steps // 2000) 
    colors = plt.cm.tab10(np.linspace(0, 0.9, p.N))
    
    legend_patches  = [mpatches.Patch(color=PHASE_STYLES[ph]['color'], label=PHASE_STYLES[ph]['label'], alpha=0.5)
                       for ph in np.unique(phase_res) if ph in PHASE_STYLES]
    if getattr(p, 'enable_gusts', False):
        legend_patches.append(mpatches.Patch(color='lightsteelblue', alpha=0.6, label='Active gust window'))

    # Panel 1: Tracking Error & Forces (Wind + Inertial)
    ax = axs[0]
    ax2r = ax.twinx()
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    
    ax.plot(t_res[::skip], pos_error_smooth[::skip], "#36393B", lw=1.5, label='Payload pos. error')
    ax.set_ylabel('Position Error [m]', fontsize=10)
    
    ax2r.plot(t_res[::skip], f_ext_xy_res[::skip], color='#0092BE', linestyle='--', lw=1.5, label=r'$F_{wind,XY}$')
    ax2r.plot(t_res[::skip], f_inertial_xy_res[::skip], color='#4C45B4', linestyle='--', lw=1.5, label=r'$F_{inertial,XY}$')
    ax2r.set_ylabel('Horizontal Forces [N]', color='black', fontsize=10)
    ax2r.tick_params(axis='y', labelcolor='black')
    
    f_max = max(np.max(f_ext_xy_res), np.max(f_inertial_xy_res)) * 1.2
    if f_max < 1e-3: f_max = 1.0
    ax2r.set_ylim(0, f_max)

    ax.set_title('Payload Trajectory Tracking Error vs Aerodynamic & Inertial Forces', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + legend_patches, labels1 + labels2 + [pt.get_label() for pt in legend_patches], loc='upper right', fontsize=8, ncol=5)

    # Panel 2: Payload Attitude
    ax = axs[1]
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    ax.plot(t_res, ref_world_roll, color='tomato', lw=2.0, linestyle='--', alpha=0.35, label='Ref World Roll')
    ax.plot(t_res, ref_world_pitch, color='mediumblue', lw=2.0, linestyle='--', alpha=0.35, label='Ref World Pitch')
    ax.plot(t_res, ref_world_yaw, color='seagreen', lw=2.0, linestyle='--', alpha=0.35, label='Ref World Yaw')
    ax.plot(t_res[::skip], pay_world_roll[::skip], 'tomato', lw=1.5, label='World Roll  φ')
    ax.plot(t_res[::skip], pay_world_pitch[::skip], 'mediumblue', lw=1.5, label='World Pitch θ')
    ax.plot(t_res[::skip], pay_world_yaw[::skip], 'green', lw=1.5, label='World Yaw   ψ')
    ax.axhline(0.0, color='black', lw=0.7, linestyle='--')
    ax.set_ylabel('Attitude [deg]', fontsize=10)
    ax.set_title('Payload World Attitude (Solid) vs Target (Dashed)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9, ncol=3)

    # Panel 3: Formation Attitude
    ax = axs[2]
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    ax.plot(t_res, achievable_ref_roll, color='tomato', lw=2.0, linestyle='--', alpha=0.35, label='Achievable Ref Roll')
    ax.plot(t_res, achievable_ref_pitch, color='mediumblue', lw=2.0, linestyle='--', alpha=0.35, label='Achievable Ref Pitch')
    ax.plot(t_res, ref_form_yaw, color='seagreen', lw=2.0, linestyle='--', alpha=0.35, label='Ref Form Yaw')
    ax.plot(t_res[::skip], form_roll[::skip], 'tomato', lw=1.5, label='Form Roll  φ')
    ax.plot(t_res[::skip], form_pitch[::skip], 'mediumblue', lw=1.5, label='Form Pitch θ')
    ax.plot(t_res[::skip], form_yaw[::skip], 'green', lw=1.5, label='Form Yaw   ψ')
    ax.axhline(0.0, color='black', lw=0.7, linestyle='--')
    ax.set_ylabel('Attitude [deg]', fontsize=10)
    ax.set_title('UAV Formation Attitude (Solid) vs Achievable Target (Dashed)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    # Panel 4: Winch Kinematics
    ax = axs[3]
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    for i in range(p.N):
        ax.plot(t_res[::skip], L_actual_res[i, ::skip], color=colors[i], lw=1.8, label=f'UAV {i+1} Actual L')
        ax.plot(t_res[::skip], L_winch_cmd_res[i, ::skip], color=colors[i], lw=1.0, linestyle='--', alpha=0.6)
    ax.axhline(p.L, color='gray', linestyle=':', lw=1.5, label='Nominal Length')
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Cable Length [m]', fontsize=10)
    ax.set_title('Winch Kinematics: Actual Length (Solid) vs Commanded (Dashed)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = [h for h, l in zip(handles, labels) if 'Actual' in l or 'Nominal' in l]
    filtered_labels = [l for l in labels if 'Actual' in l or 'Nominal' in l]
    ax.legend(filtered_handles, filtered_labels, loc='upper left', fontsize=9, ncol=p.N+1)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.35)
    plt.show()

if __name__ == "__main__":
    t_res, y_res, phase_res, ref_pos_res, ref_att_res, L_winch_cmd_res, target_uav_pos_res, f_ext_xy_res, f_inertial_xy_res, ctx, p = run_simulation()
    plot_kinematics(t_res, y_res, phase_res, ref_pos_res, ref_att_res, L_winch_cmd_res, target_uav_pos_res, f_ext_xy_res, f_inertial_xy_res, ctx, p)