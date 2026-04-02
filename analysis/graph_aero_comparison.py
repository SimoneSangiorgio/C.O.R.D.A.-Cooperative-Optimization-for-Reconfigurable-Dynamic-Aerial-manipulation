"""
graph_aero_comparison.py
-------------------------
Confronto delle prestazioni aerodinamiche mantenendo la sfera come default
e testando l'effetto "weather-vane" su geometrie estreme con vento frontale.
"""

import sys, os
import time
import numpy as np
#import matplotlib.subplots as plt_subplots
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import formation as formation
from mission import MissionContext, equations_of_motion, update_guidance
from parameters import SysParams
import physics

# ==============================================================================
# CONFIGURAZIONI GLOBALI
# ==============================================================================
# Vento puramente frontale per evidenziare chiaramente la riduzione di sezione frontale
WIND_VEC = np.array([-2.0, 4.0, 0.0])

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
# FUNZIONI DI CALCOLO METRICHE (POST-PROCESSING)
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
            vec = uav_pos[i*3:(i+1)*3, k] - attach_world # Vettore DA payload A drone
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
            # La spinta contrasta gravità, inerzia drone, e la tensione che lo tira in giù
            req_thrust = p.m_drone * (uav_acc[i*3:(i+1)*3, k] - g_v) + tau[i] * U[:, i]
            Thrusts[i, k] = np.linalg.norm(req_thrust)
            
    return Tensions, Thrusts

def payload_aerodynamics(y, p, t, wind_vec):
    """Mantenuta identica per calcolare Area e Drag orizzontale nei pannelli 3 e 4"""
    N = p.N
    steps = len(t)
    area = np.zeros(steps)
    drag = np.zeros(steps)
    
    pay_att = y[3*N+3 : 3*N+6, :]
    pay_vel = y[6*N+6 : 6*N+9, :]
    
    for k in range(steps):
        R_pay = physics.get_rotation_matrix(*pay_att[:, k])
        v_rel = pay_vel[:, k] - wind_vec
        v_rel_norm = np.linalg.norm(v_rel)
        
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
            
            f_mag = 0.5 * p.rho * p.Cd_pay * A_proj * (v_rel_norm**2)
            drag_xy = np.linalg.norm((f_mag * v_rel_dir)[:2])
        else:
            A_proj = 0.0
            drag_xy = 0.0
            
        area[k] = A_proj
        drag[k] = drag_xy
        
    return area, drag

# ==============================================================================
# SIMULAZIONE
# ==============================================================================
def run_sim(lambda_aero_val, wind_vec, label, shape_override='sphere', dim_override=(1.0, 1.0, 0.2)):
    p = SysParams()
    p.lambda_aero = lambda_aero_val
    p.lambda_shape = 0.0  
    p.initial_wind_vec = wind_vec.copy()
    p.enable_gusts = False  

    p.payload_shape = shape_override
    p.pay_l, p.pay_w, p.pay_h = dim_override

    if shape_override == 'sphere':
        r = p.R_disk
        I_solid = (2.0/5.0) * p.m_payload * (r**2)
        J_solid = np.diag([I_solid, I_solid, I_solid])
        I_liquid = (2.0/5.0) * p.m_liquid * (r**2)
        J_liquid = np.diag([I_liquid, I_liquid, I_liquid])
        p.A_pay_z = np.pi * p.R_disk**2
    else:
        Ixx = (1/12) * p.m_payload * (p.pay_w**2 + p.pay_h**2)
        Iyy = (1/12) * p.m_payload * (p.pay_l**2 + p.pay_h**2)
        Izz = (1/12) * p.m_payload * (p.pay_l**2 + p.pay_w**2)
        J_solid = np.diag([Ixx, Iyy, Izz])
        I_lxx = (1/12) * p.m_liquid * (p.pay_w**2 + p.pay_h**2)
        I_lyy = (1/12) * p.m_liquid * (p.pay_l**2 + p.pay_h**2)
        I_lzz = (1/12) * p.m_liquid * (p.pay_l**2 + p.pay_w**2)
        J_liquid = np.diag([I_lxx, I_lyy, I_lzz])
        p.A_pay_z = p.pay_l * p.pay_w

    p.J = J_solid + J_liquid
    p.invJ = np.linalg.inv(p.J)

    p.uav_offsets, p.attach_vecs, geo_radius = formation.compute_geometry(p)
    dist_h = max(geo_radius - (min(p.pay_l, p.pay_w) / 2), 0.1)
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

    SIM_DURATION = 70.0
    dt           = 0.02
    steps        = int(SIM_DURATION / dt)

    n_state   = len(x0)
    y_res     = np.zeros((n_state, steps))
    t_res     = np.zeros(steps)
    phase_res = np.zeros(steps, dtype=int)
    ref_pos_res = np.zeros((3, steps))

    current_x          = x0.copy()
    current_t          = 0.0
    last_guidance_time = -1.0
    guidance_dt        = p.optimization_dt

    print(f"Running simulation for {label} ...")
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

        n_sub = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if k % 400 == 0:
            print(f"\r    {current_t / SIM_DURATION * 100:.0f}%", end='')

    print(f"\r    Done in {time.time() - t0_cpu:.1f} s")
    return t_res, y_res, phase_res, ref_pos_res, p

# ==============================================================================
# VISUALIZZAZIONE
# ==============================================================================
def plot_aero_comparison():
    print(f"\n[Wind vector set to: {WIND_VEC} (Norm: {np.linalg.norm(WIND_VEC):.1f} m/s)]\n")
    
    t_0, y_0, ph_0, rpos_0, p_0 = run_sim(0.0, WIND_VEC, "L0 Tall Box", shape_override='rect', dim_override=(0.4, 0.4, 1.5))
    t_1, y_1, ph_1, rpos_1, p_1 = run_sim(1.0, WIND_VEC, "L1 Tall Box", shape_override='rect', dim_override=(0.4, 0.4, 1.5))
    t_2, y_2, ph_2, rpos_2, p_2 = run_sim(1.0, WIND_VEC, "L1 Flat Box", shape_override='rect', dim_override=(1.8, 1.8, 0.4))
    t_3, y_3, ph_3, rpos_3, p_3 = run_sim(1.0, WIND_VEC, "L1 Tall Box", shape_override='rect', dim_override=(0.4, 0.4, 1.5))

    N = p_0.N
    steps = len(t_0)
    dt = t_0[1] - t_0[0]
    win_len = min(51, steps)
    if win_len % 2 == 0: win_len -= 1
    
    # 1. Pos Error
    err_0 = np.linalg.norm(y_0[3*N : 3*N+3, :] - rpos_0, axis=0)
    err_1 = np.linalg.norm(y_1[3*N : 3*N+3, :] - rpos_1, axis=0)
    err_2 = np.linalg.norm(y_2[3*N : 3*N+3, :] - rpos_2, axis=0)
    err_3 = np.linalg.norm(y_3[3*N : 3*N+3, :] - rpos_3, axis=0)
    
    err_0_sm = savgol_filter(err_0, win_len, 3) if win_len > 3 else err_0
    err_1_sm = savgol_filter(err_1, win_len, 3) if win_len > 3 else err_1
    err_2_sm = savgol_filter(err_2, win_len, 3) if win_len > 3 else err_2
    err_3_sm = savgol_filter(err_3, win_len, 3) if win_len > 3 else err_3

    # Calcolo Forze e Spinte Accurate
    tens_0, thr_0 = compute_forces(y_0, p_0, t_0, WIND_VEC)
    tens_1, thr_1 = compute_forces(y_1, p_1, t_1, WIND_VEC)
    tens_2, thr_2 = compute_forces(y_2, p_2, t_2, WIND_VEC)
    tens_3, thr_3 = compute_forces(y_3, p_3, t_3, WIND_VEC)

    # 2. Tension Spread (Only L0 Sphere vs L1 Sphere)
    sp_0_raw = np.max(tens_0, axis=0) - np.min(tens_0, axis=0)
    sp_1_raw = np.max(tens_1, axis=0) - np.min(tens_1, axis=0)
    
    smooth_window_sp = int(0.5 / dt) 
    kernel_sp = np.ones(smooth_window_sp) / smooth_window_sp
    sp_0 = np.convolve(sp_0_raw, kernel_sp, mode='same')
    sp_1 = np.convolve(sp_1_raw, kernel_sp, mode='same')

    # 3 & 4. Area and Drag
    a_0, d_0 = payload_aerodynamics(y_0, p_0, t_0, WIND_VEC)
    a_1, d_1 = payload_aerodynamics(y_1, p_1, t_1, WIND_VEC)
    a_2, d_2 = payload_aerodynamics(y_2, p_2, t_2, WIND_VEC)
    a_3, d_3 = payload_aerodynamics(y_3, p_3, t_3, WIND_VEC)
    
    a_0_sm = savgol_filter(a_0, win_len, 3) if win_len > 3 else a_0
    a_1_sm = savgol_filter(a_1, win_len, 3) if win_len > 3 else a_1
    a_2_sm = savgol_filter(a_2, win_len, 3) if win_len > 3 else a_2
    a_3_sm = savgol_filter(a_3, win_len, 3) if win_len > 3 else a_3

    d_0_sm = savgol_filter(d_0, win_len, 3) if win_len > 3 else d_0
    d_1_sm = savgol_filter(d_1, win_len, 3) if win_len > 3 else d_1
    d_2_sm = savgol_filter(d_2, win_len, 3) if win_len > 3 else d_2
    d_3_sm = savgol_filter(d_3, win_len, 3) if win_len > 3 else d_3

    # 5. Power
    pow_0_raw = np.sum(thr_0**1.5, axis=0)
    pow_1_raw = np.sum(thr_1**1.5, axis=0)
    pow_2_raw = np.sum(thr_2**1.5, axis=0)
    pow_3_raw = np.sum(thr_3**1.5, axis=0)
    
    smooth_window_p = int(1.0 / dt) 
    kernel_p = np.ones(smooth_window_p) / smooth_window_p
    pow_0 = np.convolve(pow_0_raw, kernel_p, mode='same')
    pow_1 = np.convolve(pow_1_raw, kernel_p, mode='same')
    pow_2 = np.convolve(pow_2_raw, kernel_p, mode='same')
    pow_3 = np.convolve(pow_3_raw, kernel_p, mode='same')

    # --- Plotting ---
    fig, axs = plt.subplots(5, 1, figsize=(13, 20), sharex=True)
    skip = max(1, steps // 2000) 
    
    C_0 = '#1D7EC2' # Blu per Statico Sphere
    C_1 = '#E63946' # Rosso per Adaptive Sphere
    C_2 = '#2A9D8F' # Verde per Adaptive Flat Box
    C_3 = '#F4A261' # Arancio per Adaptive Tall Box

    mask_eval = (ph_0 == 3)

    # === PANEL 1: Pos Error ===
    ax = axs[0]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], err_0_sm[::skip] * 100, color=C_0, lw=1.8, label=r'$\lambda_{aero} = 0$ (Sphere)')
    ax.plot(t_1[::skip], err_1_sm[::skip] * 100, color=C_1, lw=1.8, label=r'$\lambda_{aero} = 1$ (Sphere)')

    if np.any(mask_eval):
        mean_err_0 = np.mean(err_0_sm[mask_eval]) * 100
        mean_err_1 = np.mean(err_1_sm[mask_eval]) * 100
        red_err = (1.0 - mean_err_1 / max(mean_err_0, 1e-6)) * 100
        
        idx_peak = np.argmax(err_0_sm[mask_eval])
        t_peak = t_0[mask_eval][idx_peak]
        y_top = err_0_sm[mask_eval][idx_peak] * 100
        y_bot = err_1_sm[mask_eval][idx_peak] * 100
        
        if abs(y_top - y_bot) > 0.5:
            ax.annotate('', xy=(t_peak, y_top), xytext=(t_peak, y_bot),
                        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
            ax.text(t_peak - 0.6, ((y_top + y_bot) / 2), 
                    f'L0->L1 Red:\n{red_err:.1f}%',
                    fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')

        y_min = min(np.min(err_0_sm[mask_eval]), np.min(err_1_sm[mask_eval])) * 100
        y_max = max(np.max(err_0_sm[mask_eval]), np.max(err_1_sm[mask_eval])) * 100
        dy = max((y_max - y_min) * 0.15, 0.5)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    ax.set_ylabel('Position Error [cm]', fontsize=10)
    ax.set_title('Payload Trajectory Tracking Error', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # === PANEL 2: Tension Spread ===
    ax = axs[1]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], sp_0[::skip], color=C_0, lw=1.8, label=r'$\lambda_{aero} = 0$ (Sphere)')
    ax.plot(t_1[::skip], sp_1[::skip], color=C_1, lw=1.8, label=r'$\lambda_{aero} = 1$ (Sphere)')

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

        y_min = min(np.min(sp_0[mask_eval]), np.min(sp_1[mask_eval]))
        y_max = max(np.max(sp_0[mask_eval]), np.max(sp_1[mask_eval]))
        dy = max((y_max - y_min) * 0.15, 0.02)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    ax.set_ylabel('$T_{max} - T_{min}$ [N]', fontsize=10)
    ax.set_title('Cable Tension Spread (Formation Load Imbalance)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # === PANEL 3: Area ===
    ax = axs[2]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], a_0_sm[::skip], color=C_0, lw=1.8, label=r'L0 (Sphere)')
    ax.plot(t_1[::skip], a_1_sm[::skip], color=C_1, lw=1.8, label=r'L1 (Sphere)')
    ax.plot(t_2[::skip], a_2_sm[::skip], color=C_2, lw=1.8, linestyle='--', label=r'L1 (Flat Box)')
    ax.plot(t_3[::skip], a_3_sm[::skip], color=C_3, lw=1.8, linestyle='-.', label=r'L1 (Tall Thin Box)')

    if np.any(mask_eval):
        mean_a2 = np.mean(a_2_sm[mask_eval])
        mean_a3 = np.mean(a_3_sm[mask_eval])
        red_area = (1.0 - mean_a3 / max(mean_a2, 1e-6)) * 100
        
        idx_peak = np.argmax(a_2_sm[mask_eval])
        t_peak = t_0[mask_eval][idx_peak]
        y_top = a_2_sm[mask_eval][idx_peak]
        y_bot = a_3_sm[mask_eval][idx_peak]
        
        if abs(y_top - y_bot) > 0.05:
            ax.annotate('', xy=(t_peak, y_top), xytext=(t_peak, y_bot),
                        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
            ax.text(t_peak - 0.6, ((y_top + y_bot) / 2), 
                    f'Flat->Tall Red:\n{red_area:.1f}%',
                    fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')

        y_min = min(np.min(a_0_sm[mask_eval]), np.min(a_1_sm[mask_eval]), np.min(a_2_sm[mask_eval]), np.min(a_3_sm[mask_eval]))
        y_max = max(np.max(a_0_sm[mask_eval]), np.max(a_1_sm[mask_eval]), np.max(a_2_sm[mask_eval]), np.max(a_3_sm[mask_eval]))
        dy = max((y_max - y_min) * 0.15, 0.05)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    ax.set_ylabel('Projected Area [m²]', fontsize=10)
    ax.set_title('Aerodynamic Profile Exposed to Relative Wind', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # === PANEL 4: Drag ===
    ax = axs[3]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], d_0_sm[::skip], color=C_0, lw=1.8, label=r'L0 (Sphere)')
    ax.plot(t_1[::skip], d_1_sm[::skip], color=C_1, lw=1.8, label=r'L1 (Sphere)')
    ax.plot(t_2[::skip], d_2_sm[::skip], color=C_2, lw=1.8, linestyle='--', label=r'L1 (Flat Box)')
    ax.plot(t_3[::skip], d_3_sm[::skip], color=C_3, lw=1.8, linestyle='-.', label=r'L1 (Tall Box)')

    if np.any(mask_eval):
        mean_d2 = np.mean(d_2_sm[mask_eval])
        mean_d3 = np.mean(d_3_sm[mask_eval])
        red_drag = (1.0 - mean_d3 / max(mean_d2, 1e-6)) * 100
        
        idx_peak = np.argmax(d_2_sm[mask_eval])
        t_peak = t_0[mask_eval][idx_peak]
        y_top = d_2_sm[mask_eval][idx_peak]
        y_bot = d_3_sm[mask_eval][idx_peak]
        
        if abs(y_top - y_bot) > 0.5:
            ax.annotate('', xy=(t_peak, y_top), xytext=(t_peak, y_bot),
                        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
            ax.text(t_peak - 0.6, ((y_top + y_bot) / 2), 
                    f'Flat->Tall Red:\n{red_drag:.1f}%',
                    fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')

        y_min = min(np.min(d_0_sm[mask_eval]), np.min(d_1_sm[mask_eval]), np.min(d_2_sm[mask_eval]), np.min(d_3_sm[mask_eval]))
        y_max = max(np.max(d_0_sm[mask_eval]), np.max(d_1_sm[mask_eval]), np.max(d_2_sm[mask_eval]), np.max(d_3_sm[mask_eval]))
        dy = max((y_max - y_min) * 0.15, 0.5)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    ax.set_ylabel('Horizontal Drag [N]', fontsize=10)
    ax.set_title('Payload Disturbance Force', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)

    # === PANEL 5: Power ===
    ax = axs[4]
    shade_phases(ax, t_0, ph_0)
    ax.plot(t_0[::skip], pow_0[::skip], color=C_0, lw=1.8, label=r'L0 (Sphere)')
    ax.plot(t_1[::skip], pow_1[::skip], color=C_1, lw=1.8, label=r'L1 (Sphere)')
    ax.plot(t_2[::skip], pow_2[::skip], color=C_2, lw=1.8, linestyle='--', label=r'L1 (Flat Box)')
    ax.plot(t_3[::skip], pow_3[::skip], color=C_3, lw=1.8, linestyle='-.', label=r'L1 (Tall Box)')

    if np.any(mask_eval):
        m_p0 = np.mean(pow_0[mask_eval])
        m_p1 = np.mean(pow_1[mask_eval])
        m_p2 = np.mean(pow_2[mask_eval])
        m_p3 = np.mean(pow_3[mask_eval])
        
        txt_pow = (f"Avg Power vs L0 Sphere:\n"
                   f"L1 Sphere: {(m_p1/m_p0 - 1)*100:+.1f}%\n"
                   f"L1 Flat Box: {(m_p2/m_p0 - 1)*100:+.1f}%\n"
                   f"L1 Tall Box: {(m_p3/m_p0 - 1)*100:+.1f}%")
        
        y_pos = np.mean(pow_1[mask_eval])
        x_target = t_0[mask_eval][-1] - (t_0[mask_eval][-1] - t_0[mask_eval][0]) * 0.03
        
        ax.text(x_target, y_pos, txt_pow, 
                fontsize=9, color='indigo', fontweight='bold', ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

        y_min = min(np.min(pow_0[mask_eval]), np.min(pow_1[mask_eval]), np.min(pow_2[mask_eval]), np.min(pow_3[mask_eval]))
        y_max = max(np.max(pow_0[mask_eval]), np.max(pow_1[mask_eval]), np.max(pow_2[mask_eval]), np.max(pow_3[mask_eval]))
        dy = max((y_max - y_min) * 0.15, 1.0)
        ax.set_ylim(max(0, y_min - dy), y_max + dy * 2.5)

    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel(r'Power ($\sum T^{1.5}$)', fontsize=10)
    ax.set_title('Average System Power (Energy Consumption)', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Adatta i margini X alla Fase in analisi (Fase 4)
    if np.any(mask_eval):
        t_start = t_0[mask_eval][0]
        t_end = t_0[mask_eval][-1]
        axs[0].set_xlim(t_start, t_end)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.35)
    plt.show()

if __name__ == "__main__":
    plot_aero_comparison()