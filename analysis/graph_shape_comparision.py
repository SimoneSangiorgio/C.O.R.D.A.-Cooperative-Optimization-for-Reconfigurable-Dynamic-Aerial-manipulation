"""
graph_shape_comparison.py
-------------------------
Confronto delle prestazioni della formazione con lambda_shape = 0 e lambda_shape = 1
sotto l'effetto di vento costante.
"""

import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import formation as formation
from mission import MissionContext, equations_of_motion, update_guidance
from parameters import SysParams
import physics  # <-- AGGIUNTO per usare la get_rotation_matrix

# ==============================================================================
# CONFIGURAZIONI GLOBALI
# ==============================================================================
# Imposta qui il vettore del vento desiderato [x, y, z] in m/s
WIND_VEC = np.array([2.0, -4.0, 0.0])

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

# ==============================================================================
# FUNZIONI DI CALCOLO METRICHE (CORRETTE FISICAMENTE)
# ==============================================================================
def compute_forces(y, p, t, wind_vec):
    """Calcola le tensioni sui cavi e la spinta reale richiesta ai droni considerando il payload."""
    N = p.N
    steps = len(t)
    dt = t[1] - t[0]
    
    uav_pos = y[0 : 3*N, :]
    pay_pos = y[3*N : 3*N+3, :]
    pay_att = y[3*N+3 : 3*N+6, :]
    uav_vel = y[3*N+6 : 6*N+6, :]
    pay_vel = y[6*N+6 : 6*N+9, :]
    
    # Accelerazioni
    uav_acc = np.zeros((3*N, steps))
    for i in range(N):
        uav_acc[i*3:(i+1)*3, :] = np.gradient(uav_vel[i*3:(i+1)*3, :], dt, axis=1)
    pay_acc = np.gradient(pay_vel, dt, axis=1)
    
    g_v = np.array([0, 0, -9.81])
    m_pay_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)
    
    Tensions = np.zeros((N, steps))
    Thrusts = np.zeros((N, steps))
    
    for k in range(steps):
        # 1. Calcolo Drag Aerodinamico sul Payload
        R_pay = physics.get_rotation_matrix(*pay_att[:, k])
        v_rel = pay_vel[:, k] - wind_vec
        v_rel_norm = np.linalg.norm(v_rel)
        
        F_aero = np.zeros(3)
        if v_rel_norm > 1e-3:
            v_rel_dir = v_rel / v_rel_norm
            v_rel_body = R_pay.T @ v_rel_dir
            
            if p.payload_shape == 'sphere':
                A_proj = np.pi * (p.R_disk**2)
            else:
                A_x = p.pay_w * p.pay_h
                A_y = p.pay_l * p.pay_h
                A_z = p.pay_l * p.pay_w
                A_proj = (A_x * abs(v_rel_body[0]) + A_y * abs(v_rel_body[1]) + A_z * abs(v_rel_body[2]))
            
            f_mag = -0.5 * p.rho * p.Cd_pay * A_proj * (v_rel_norm**2)
            F_aero = f_mag * v_rel_dir
            
        # 2. Forza totale che i cavi devono applicare AL payload
        F_cables_on_pay = m_pay_tot * (pay_acc[:, k] - g_v) - F_aero
        
        # 3. Costruzione della matrice direzionale dei cavi (U)
        U = np.zeros((3, N))
        for i in range(N):
            attach_world = pay_pos[:, k] + R_pay @ p.attach_vecs[:, i]
            vec = uav_pos[i*3:(i+1)*3, k] - attach_world
            dist = np.linalg.norm(vec)
            if dist > 1e-4:
                U[:, i] = vec / dist
            else:
                U[:, i] = np.array([0, 0, 1])
                
        # 4. Risoluzione tensioni (minima norma)
        tau = np.linalg.pinv(U) @ F_cables_on_pay
        tau = np.maximum(tau, 0.0) # I cavi non possono spingere
        Tensions[:, k] = tau
        
        # 5. Spinta netta richiesta al singolo drone
        for i in range(N):
            req_thrust = p.m_drone * (uav_acc[i*3:(i+1)*3, k] - g_v) + tau[i] * U[:, i]
            Thrusts[i, k] = np.linalg.norm(req_thrust)
            
    return Tensions, Thrusts

# ==============================================================================
# SIMULAZIONE
# ==============================================================================
def run_sim(lambda_shape_val, wind_vec, label):
    p = SysParams()
    p.lambda_shape = lambda_shape_val
    p.initial_wind_vec = wind_vec.copy()
    p.enable_gusts = False  # Vento costante per questo test
    
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

    current_x          = x0.copy()
    current_t          = 0.0
    last_guidance_time = -1.0
    guidance_dt        = p.optimization_dt

    print(f"Running simulation for {label}  (N={p.N}, T={SIM_DURATION}s) ...")
    t0_cpu = time.time()
    
    for k in range(steps):
        y_res[:, k]  = current_x
        t_res[k]     = current_t
        phase_res[k] = ctx.phase

        if (current_t - last_guidance_time >= guidance_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guidance_time = current_t

        if ctx.current_ref is not None:
            ref_pos_res[:, k] = ctx.current_ref['pos']
        else:
            ref_pos_res[:, k] = current_x[3*p.N : 3*p.N+3]
            
        if hasattr(ctx, 'current_target_att'):
            ref_att_res[:, k] = ctx.current_target_att
        else:
            ref_att_res[:, k] = ref_att_res[:, k-1] if k > 0 else np.zeros(3)

        n_sub = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if k % 200 == 0:
            print(f"\r    {current_t / SIM_DURATION * 100:.0f}%", end='')

    print(f"\r    Done in {time.time() - t0_cpu:.1f} s")
    return t_res, y_res, phase_res, ref_pos_res, ref_att_res, p

# ==============================================================================
# VISUALIZZAZIONE
# ==============================================================================
def plot_shape_comparison():
    print(f"Wind vector set to: {WIND_VEC}")

    t_0, y_0, ph_0, rpos_0, ratt_0, p_0 = run_sim(0.0, WIND_VEC, "Lambda Shape = 0 (Static)")
    t_1, y_1, ph_1, rpos_1, ratt_1, p_1 = run_sim(1.0, WIND_VEC, "Lambda Shape = 1 (Adaptive)")

    N = p_0.N
    steps = len(t_0)
    dt = t_0[1] - t_0[0]
    
    # 1. Trajectory Tracking Error
    pay_pos_0 = y_0[3*N : 3*N+3, :]
    pay_pos_1 = y_1[3*N : 3*N+3, :]
    
    err_vec_0 = pay_pos_0 - rpos_0
    err_vec_1 = pay_pos_1 - rpos_1
    
    window_len = min(51, steps)
    if window_len % 2 == 0: window_len -= 1
    err_vec_0_smooth = savgol_filter(err_vec_0, window_length=window_len, polyorder=3, axis=1) if window_len > 3 else err_vec_0
    err_vec_1_smooth = savgol_filter(err_vec_1, window_length=window_len, polyorder=3, axis=1) if window_len > 3 else err_vec_1
    
    pos_err_0 = np.linalg.norm(err_vec_0_smooth, axis=0)
    pos_err_1 = np.linalg.norm(err_vec_1_smooth, axis=0)

    # 2. Attitude Error (Roll & Pitch)
    pay_att_0 = y_0[3*N+3 : 3*N+6, :]
    pay_att_1 = y_1[3*N+3 : 3*N+6, :]
    
    att_err_0 = np.linalg.norm(pay_att_0[:2, :] - ratt_0[:2, :], axis=0) * 180.0 / np.pi
    att_err_1 = np.linalg.norm(pay_att_1[:2, :] - ratt_1[:2, :], axis=0) * 180.0 / np.pi

    # --- CALCOLO FISICAMENTE CORRETTO DI TENSIONI E POTENZA ---
    tens_0, thr_0 = compute_forces(y_0, p_0, t_0, WIND_VEC)
    tens_1, thr_1 = compute_forces(y_1, p_1, t_1, WIND_VEC)

    # 3. Tension Spread
    sp_0_raw = np.max(tens_0, axis=0) - np.min(tens_0, axis=0)
    sp_1_raw = np.max(tens_1, axis=0) - np.min(tens_1, axis=0)
    
    smooth_window = int(0.5 / dt) 
    kernel = np.ones(smooth_window) / smooth_window
    sp_0 = np.convolve(sp_0_raw, kernel, mode='same')
    sp_1 = np.convolve(sp_1_raw, kernel, mode='same')

    # 4. System Power
    pow_0_raw = np.sum(thr_0**1.5, axis=0)
    pow_1_raw = np.sum(thr_1**1.5, axis=0)
    
    smooth_window_p = int(1.0 / dt) 
    kernel_p = np.ones(smooth_window_p) / smooth_window_p
    pow_0 = np.convolve(pow_0_raw, kernel_p, mode='same')
    pow_1 = np.convolve(pow_1_raw, kernel_p, mode='same')

    # --- Plotting ---
    fig, axs = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    skip = max(1, steps // 2000) 
    
    C_0 = '#1D7EC2' # Blu per Statico (0)
    C_1 = '#E63946' # Rosso per Adattivo (1)

    # === PANEL 1: Pos Error ===
    ax = axs[0]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], pos_err_0[::skip] * 100, color=C_0, lw=1.8, label=r'$\lambda_{shape} = 0$')
    ax.plot(t_1[::skip], pos_err_1[::skip] * 100, color=C_1, lw=1.8, label=r'$\lambda_{shape} = 1$')
    
    mask_eval = (ph_0 == 4)
    if np.any(mask_eval):
        mean_err_0 = np.mean(pos_err_0[mask_eval]) * 100
        mean_err_1 = np.mean(pos_err_1[mask_eval]) * 100
        red_err = (1.0 - mean_err_1 / max(mean_err_0, 1e-6)) * 100
        
        idx_peak = np.argmax(pos_err_0[mask_eval])
        t_peak = t_0[mask_eval][idx_peak]
        
        y_top = pos_err_0[mask_eval][idx_peak] * 100
        y_bot = pos_err_1[mask_eval][idx_peak] * 100
        
        if abs(y_top - y_bot) > 0.5:
            ax.annotate('', xy=(t_peak, y_top), xytext=(t_peak, y_bot),
                        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
            ax.text(t_peak - 0.6, ((y_top + y_bot) / 2) + 0.5, 
                    f'Avg. Reduction:\n{red_err:.1f}%',
                    fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')

    ax.set_ylabel('Position Error [cm]', fontsize=10)
    ax.set_title('Payload Trajectory Tracking Error', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(bottom=0)

    # Zoom dinamico asse Y per Fase 3
    if np.any(mask_eval):
        y_min = min(np.min(pos_err_0[mask_eval]), np.min(pos_err_1[mask_eval])) * 100
        y_max = max(np.max(pos_err_0[mask_eval]), np.max(pos_err_1[mask_eval])) * 100
        dy = max((y_max - y_min) * 0.15, 0.5)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    # === PANEL 2: Attitude Error ===
    ax = axs[1]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], att_err_0[::skip], color=C_0, lw=1.8, label=r'$\lambda_{shape} = 0$')
    ax.plot(t_1[::skip], att_err_1[::skip], color=C_1, lw=1.8, label=r'$\lambda_{shape} = 1$')
    
    if np.any(mask_eval):
        mean_att_0 = np.mean(att_err_0[mask_eval])
        mean_att_1 = np.mean(att_err_1[mask_eval])
        red_att = (1.0 - mean_att_1 / max(mean_att_0, 1e-6)) * 100
        
        idx_peak_att = np.argmax(att_err_0[mask_eval])
        t_peak_att = t_0[mask_eval][idx_peak_att]
        
        y_top_a = att_err_0[mask_eval][idx_peak_att]
        y_bot_a = att_err_1[mask_eval][idx_peak_att]
        
        if abs(y_top_a - y_bot_a) > 0.1:
            ax.annotate('', xy=(t_peak_att, y_top_a), xytext=(t_peak_att, y_bot_a),
                        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
            ax.text(t_peak_att - 0.6, ((y_top_a + y_bot_a) / 2), 
                    f'Avg. Reduction:\n{red_att:.1f}%',
                    fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')

    ax.set_ylabel('Attitude Error [deg]', fontsize=10)
    ax.set_title('Payload Attitude Error (World vs Target)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(bottom=0)

    # Zoom dinamico asse Y per Fase 3
    if np.any(mask_eval):
        y_min = min(np.min(att_err_0[mask_eval]), np.min(att_err_1[mask_eval]))
        y_max = max(np.max(att_err_0[mask_eval]), np.max(att_err_1[mask_eval]))
        dy = max((y_max - y_min) * 0.15, 0.5)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    # === PANEL 3: Tension Spread ===
    ax = axs[2]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], sp_0[::skip], color=C_0, lw=1.8, label=r'$\lambda_{shape} = 0$')
    ax.plot(t_1[::skip], sp_1[::skip], color=C_1, lw=1.8, label=r'$\lambda_{shape} = 1$')

    if np.any(mask_eval):
        mean_sp_0 = np.mean(sp_0[mask_eval])
        mean_sp_1 = np.mean(sp_1[mask_eval])
        red_sp = (1.0 - mean_sp_1 / max(mean_sp_0, 1e-6)) * 100
        
        idx_peak_sp = np.argmax(sp_0[mask_eval])
        t_peak_sp = t_0[mask_eval][idx_peak_sp]
        
        y_top_sp = sp_0[mask_eval][idx_peak_sp]
        y_bot_sp = sp_1[mask_eval][idx_peak_sp]
        
        if abs(y_top_sp - y_bot_sp) > 0.01:
            ax.annotate('', xy=(t_peak_sp, y_top_sp), xytext=(t_peak_sp, y_bot_sp),
                        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
            ax.text(t_peak_sp - 0.6, ((y_top_sp + y_bot_sp) / 2), 
                    f'Avg. Reduction:\n{red_sp:.1f}%',
                    fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')

    ax.set_ylabel('$T_{max} - T_{min}$ [N]', fontsize=10)
    ax.set_title('Cable Tension Spread (Formation Load Imbalance)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_ylim(bottom=0)

    # Zoom dinamico asse Y per Fase 3
    if np.any(mask_eval):
        y_min = min(np.min(sp_0[mask_eval]), np.min(sp_1[mask_eval]))
        y_max = max(np.max(sp_0[mask_eval]), np.max(sp_1[mask_eval]))
        dy = max((y_max - y_min) * 0.15, 0.02)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    # === PANEL 4: System Power ===
    ax = axs[3]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], pow_0[::skip], color=C_0, lw=1.8, label=r'$\lambda_{shape} = 0$')
    ax.plot(t_1[::skip], pow_1[::skip], color=C_1, lw=1.8, label=r'$\lambda_{shape} = 1$')

    if np.any(mask_eval):
        mean_pow_0 = np.mean(pow_0[mask_eval])
        mean_pow_1 = np.mean(pow_1[mask_eval])
        pow_diff = ((mean_pow_1 / mean_pow_0) - 1.0) * 100
        
        sign = "+" if pow_diff > 0 else ""
        testo_energia = f"Avg Power variation:\n{sign}{pow_diff:.2f}%"
        
        y_pos = mean_pow_0
        x_target = t_0[mask_eval][-1] - (t_0[mask_eval][-1] - t_0[mask_eval][0]) * 0.03
        
        ax.text(x_target, y_pos, testo_energia, 
                fontsize=10, color='indigo', fontweight='bold', ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel(r'Power ($\sum T^{1.5}$)', fontsize=10)
    ax.set_title('Average System Power (Energy Consumption)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    if np.any(mask_eval):
        y_min = min(np.min(pow_0[mask_eval]), np.min(pow_1[mask_eval]))
        y_max = max(np.max(pow_0[mask_eval]), np.max(pow_1[mask_eval]))
        dy = max((y_max - y_min) * 0.15, 1.0)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)
    
    if np.any(mask_eval):
        t_start = t_0[mask_eval][0]
        t_end = t_0[mask_eval][-1]
        axs[0].set_xlim(t_start, t_end)
        
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.35)
    plt.show()

if __name__ == "__main__":
    plot_shape_comparison()