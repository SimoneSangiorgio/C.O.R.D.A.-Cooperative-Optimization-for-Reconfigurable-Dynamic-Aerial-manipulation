"""
graph_dynamics.py
--------------------
Dynamic analysis: Forces, Thrusts, Tension Spreads, and System Power.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def tension_spread(y, p, t):
    N    = p.N
    dt   = t[1] - t[0]
    uvel = y[3*N+6 : 6*N+6, :]
    g_v  = np.array([[0], [0], [-9.81]])
    T = []
    for i in range(N):
        acc_i = np.gradient(uvel[i*3 : (i+1)*3, :], dt, axis=1)
        F_i   = np.linalg.norm(p.m_drone * (acc_i - g_v), axis=0)
        T.append(F_i)
    T = np.array(T)
    return np.max(T, axis=0) - np.min(T, axis=0)

def system_power(y, p, t):
    N    = p.N
    dt   = t[1] - t[0]
    uvel = y[3*N+6 : 6*N+6, :]
    g_v  = np.array([[0], [0], [-9.81]])
    
    P_tot = np.zeros(len(t))
    for i in range(N):
        acc_i = np.gradient(uvel[i*3 : (i+1)*3, :], dt, axis=1)
        F_i   = np.linalg.norm(p.m_drone * (acc_i - g_v), axis=0)
        P_tot += F_i**(1.5)
    
    return P_tot

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
    
    thrust_res = np.zeros((p.N, steps))
    f_ext_xy_res = np.zeros(steps)  
    f_inertial_xy_res = np.zeros(steps) 

    current_x          = x0.copy()
    current_t          = 0.0
    last_guidance_time = -1.0
    guidance_dt        = p.optimization_dt

    m_tot = p.m_payload + p.m_liquid

    print(f"Running simulation for Dynamics  N={p.N}, T={SIM_DURATION}s …")

    for k in range(steps):
        y_res[:, k]  = current_x
        t_res[k]     = current_t
        phase_res[k] = ctx.phase

        if (current_t - last_guidance_time >= guidance_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guidance_time = current_t

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

        if ctx.current_ref is not None:
            f_inertial_xy_res[k] = np.linalg.norm(m_tot * ctx.current_ref['acc'][:2])

        n_sub = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if hasattr(ctx, 'last_u_acc'):
            for i in range(p.N):
                acc_cmd  = ctx.last_u_acc[:, i]
                thrust_v = p.m_drone * (acc_cmd + np.array([0, 0, 9.81]))
                thrust_res[i, k] = np.linalg.norm(thrust_v)

        if k % 100 == 0:
            print(f"\r  {current_t / SIM_DURATION * 100:5.1f}%  t={current_t:.2f}s  phase={ctx.phase}", end='')

    print("\nDone")
    return t_res, y_res, phase_res, thrust_res, f_ext_xy_res, f_inertial_xy_res, ctx, p

def plot_dynamics(t_res, y_res, phase_res, thrust_res, f_ext_xy_res, f_inertial_xy_res, ctx, p):
    steps = len(t_res)
    dt = t_res[1] - t_res[0]

    if ctx.theta_log:
        log_t  = np.array([v[0] for v in ctx.theta_log])
        log_th = np.array([v[1] for v in ctx.theta_log]) 
        theta_interp_rad = np.interp(t_res, log_t, log_th)
        theta_interp_deg = np.degrees(theta_interp_rad)
    else:
        theta_interp_deg = np.full_like(t_res, p.theta_ref)

    sp_raw = tension_spread(y_res, p, t_res)
    pow_raw = system_power(y_res, p, t_res)

    smooth_window_sp = int(0.5 / dt) 
    kernel_sp = np.ones(smooth_window_sp) / smooth_window_sp
    sp_smooth = np.convolve(sp_raw, kernel_sp, mode='same')

    smooth_window_p = int(1.0 / dt) 
    kernel_p = np.ones(smooth_window_p) / smooth_window_p
    pow_smooth = np.convolve(pow_raw, kernel_p, mode='same')

    fig, axs = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    skip = max(1, steps // 2000) 
    colors = plt.cm.tab10(np.linspace(0, 0.9, p.N)) 
    
    legend_patches  = [mpatches.Patch(color=PHASE_STYLES[ph]['color'], label=PHASE_STYLES[ph]['label'], alpha=0.5)
                       for ph in np.unique(phase_res) if ph in PHASE_STYLES]
    if getattr(p, 'enable_gusts', False):
        legend_patches.append(mpatches.Patch(color='lightsteelblue', alpha=0.6, label='Active gust window'))

    # Panel 1: Cone Angle & External Forces (Wind + Inertial)
    ax = axs[0]
    ax2r = ax.twinx()
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    
    ax.plot(t_res[::skip], theta_interp_deg[::skip], 'black', lw=1.5, label=r'$\alpha_{\mathrm{opt}}$')
    ax.axhline(p.theta_ref, color='gray', linestyle='--', lw=1.0, label=rf'$\alpha_{{\mathrm{{ref}}}}$')
    
    ax2r.plot(t_res[::skip], f_ext_xy_res[::skip], color='#0092BE', linestyle='--', lw=1.5, label=r'$F_{wind,XY}$')
    ax2r.plot(t_res[::skip], f_inertial_xy_res[::skip], color='#4C45B4', linestyle='--', lw=1.5, label=r'$F_{inertial,XY}$')
    
    ax2r.set_ylabel('Horizontal Forces [N]', color='black', fontsize=10)
    ax2r.tick_params(axis='y', labelcolor='black')
    y1_min, y1_max = ax.get_ylim()
    
    f_max = max(np.max(f_ext_xy_res), np.max(f_inertial_xy_res)) * 1.1
    if f_max < 1e-3: f_max = 1.0
    
    ratio = (p.theta_ref - y1_min) / (y1_max - y1_min)
    if ratio < 0.99:
        f_min = -(ratio * f_max) / (1.0 - ratio)
    else:
        f_min = 0.0
        
    ax2r.set_ylim(f_min, f_max)
    
    ax.set_ylabel(r'Cone Angle $\alpha_{\mathrm{opt}}$ [deg]', fontsize=10)
    ax.set_title('Optimal Cone Angle vs External & Inertial Forces', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, ncol=2)

    # Panel 2: UAV Thrust Magnitudes
    ax = axs[1]
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    for i in range(p.N):
        ax.plot(t_res[::skip], thrust_res[i, ::skip], color=colors[i], lw=1.2, label=f'UAV {i+1}')
    m_tot = p.m_payload + p.m_liquid
    T_safe_base = p.k_safe * (m_tot * p.g / p.N)
    T_max_limit = (p.F_max_thrust - p.m_drone * p.g) * p.k_limit
    ax.axhline(T_safe_base, color='darkorange', linestyle=':', lw=1.2, label=f'$T_{{safe}}$')
    ax.axhline(T_max_limit, color='crimson',    linestyle=':', lw=1.2, label=f'$T_{{lim}}$')
    ax.set_ylabel('Thrust [N]', fontsize=10)
    ax.set_title('UAV Thrust Magnitudes', fontweight='bold', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=p.N + 2)

    # Panel 3: Cable Tension Spread
    ax = axs[2]
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    ax.plot(t_res[::skip], sp_smooth[::skip], color='#E63946', lw=1.5, label='$T_{max} − T_{min}$')
    ax.set_ylabel('$T_{max} − T_{min}$ [N]', fontsize=10)
    ax.set_title('Cable Tension Spread (Load Imbalance)', fontweight='bold', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    # Panel 4: Average System Power
    ax = axs[3]
    shade_phases(ax, t_res, phase_res)
    if getattr(p, 'enable_gusts', False): shade_gusts(ax, p.gust_schedule)
    ax.plot(t_res[::skip], pow_smooth[::skip], color='#1D7EC2', lw=1.5, label='System Power')
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel(r'Power ($\sum T^{1.5}$)', fontsize=10)
    ax.set_title('Average System Power (Energy Consumption)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.35)
    plt.show()

if __name__ == "__main__":
    t_res, y_res, phase_res, thrust_res, f_ext_xy_res, f_inertial_xy_res, ctx, p = run_simulation()
    plot_dynamics(t_res, y_res, phase_res, thrust_res, f_ext_xy_res, f_inertial_xy_res, ctx, p)