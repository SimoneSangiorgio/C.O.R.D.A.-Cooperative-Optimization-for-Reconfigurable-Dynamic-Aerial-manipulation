"""
graph_winch_damping.py
----------------------
Dynamic analysis: active winch damping effectiveness.

Two parallel simulations are run:
  • Passive  — winch damping ratio ζ = 0  (no active cable modulation)
  • Active   — winch damping ratio ζ = p.winch_damping_ratio (typically 0.7–0.9)

A short wind impulse is injected during Phase 4 (stable navigation) to excite
payload pendulum oscillation.  Using Phase 4 ensures the system is fully in
flight and any response/non-response is due to the winch logic, not to the
liftoff / pretension transient.

Gust timing
-----------
The script first runs a "probe" simulation (no gust) to find t_phase4, then
places the gust 5 s into Phase 4 so the system is guaranteed to be in steady
navigation before the disturbance.

Metrics plotted
---------------
  Panel A — Payload horizontal displacement from pre-gust position [cm]
  Panel B — Payload horizontal velocity |v_xy| [m/s]
  Panel C — Payload angular velocity |ω| [deg/s]
  Panel D — Commanded cable lengths L_i(t) [m]  (winch activity proxy)

Settling time (5 cm threshold, 2-s hold) is annotated on Panel A.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy, time
import numpy as np
import matplotlib.pyplot as plt

import formation as formation
from mission import MissionContext, equations_of_motion, update_guidance
from parameters import SysParams


# ==============================================================================
# PHASE SHADING
# ==============================================================================
PHASE_COLORS = {0:'#AAAAAA', 1:'#FFD700', 2:'#90EE90', 3:'#87CEEB',
                4:'#FFA07A', 5:'#DDA0DD', 6:'#98FB98'}

def shade_phases(ax, t, phases, alpha=0.10):
    for ph in np.unique(phases):
        mask = phases == ph
        idx  = np.where(mask)[0]
        ax.axvspan(t[idx[0]], t[idx[-1]],
                   color=PHASE_COLORS.get(ph, 'white'), alpha=alpha, zorder=0)


# ==============================================================================
# PROBE — find start time of Phase 4
# ==============================================================================
def find_phase4_start(p_ref) -> float:
    p = copy.deepcopy(p_ref)
    p.enable_gusts = False

    ctx = MissionContext(p)
    uav_pos_0 = np.zeros((3, p.N))
    for i in range(p.N):
        uav_pos_0[:, i] = p.home_pos + p.uav_offsets[:, i] + np.array([0, 0, p.floor_z])
    pay_pos_0 = p.home_pos + np.array([0, 0, p.floor_z + p.pay_h / 2.0])
    x0 = np.concatenate([
        uav_pos_0.flatten('F'), pay_pos_0, np.zeros(3), np.zeros(3 * p.N),
        np.zeros(3), np.zeros(3), np.zeros(3 * p.N), np.zeros(3)
    ])

    dt = 0.02;  current_x = x0.copy();  current_t = 0.0
    last_guid = -1.0;  guid_dt = p.optimization_dt

    print("  Probe: locating Phase 4 …", end='')
    while current_t < 80.0:
        if (current_t - last_guid >= guid_dt) or (current_t == 0.0):
            update_guidance(current_t, current_x, p, ctx)
            last_guid = current_t
        if ctx.phase == 4:
            print(f" found at t = {current_t:.1f} s")
            return float(current_t)
        n_sub  = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            current_x = current_x + (
                equations_of_motion(current_t, current_x, p, ctx) * dt_sub)
            current_t += dt_sub
    print(" not found — using fallback t = 30 s")
    return 30.0


# ==============================================================================
# SIMULATION
# ==============================================================================
def run_sim(zeta: float, label: str,
            gust_t_start: float,
            gust_duration: float = 1.5,
            gust_vec=None):
    if gust_vec is None:
        gust_vec = np.array([5.0, 5.0, 0.0])

    p = SysParams()
    p.k_active_damp = zeta
    p.enable_gusts  = True
    p.gust_schedule = [
        {'t_start': gust_t_start, 'duration': gust_duration,
         'vec': gust_vec, 'ramp': 0.3}
    ]

    p.uav_offsets, p.attach_vecs, geo_radius = formation.compute_geometry(p)
    dist_h = geo_radius - (min(p.pay_l, p.pay_w) / 2
                           if p.payload_shape in ['box', 'rect'] else p.R_disk)
    dist_h = max(dist_h, 0.1)
    p.safe_altitude_offset = np.sqrt(p.L**2 - dist_h**2) if p.L > dist_h else p.L

    ctx = MissionContext(p)
    uav_pos_0 = np.zeros((3, p.N))
    for i in range(p.N):
        uav_pos_0[:, i] = p.home_pos + p.uav_offsets[:, i] + np.array([0, 0, p.floor_z])
    pay_pos_0 = p.home_pos + np.array([0, 0, p.floor_z + p.pay_h / 2.0])
    x0 = np.concatenate([
        uav_pos_0.flatten('F'), pay_pos_0, np.zeros(3), np.zeros(3 * p.N),
        np.zeros(3), np.zeros(3), np.zeros(3 * p.N), np.zeros(3)
    ])

    SIM_END  = gust_t_start + gust_duration + 25.0
    dt       = 0.02
    steps    = int(SIM_END / dt) + 1

    y_res     = np.zeros((len(x0), steps))
    t_res     = np.zeros(steps)
    phase_res = np.zeros(steps, dtype=int)
    L_log     = np.zeros((p.N, steps))

    current_x = x0.copy();  current_t = 0.0
    last_guid  = -1.0;       guid_dt   = p.optimization_dt

    print(f"  [{label}]  gust at t = {gust_t_start:.1f} s …")
    t0_cpu = time.time()

    for k in range(steps):
        y_res[:, k]  = current_x
        t_res[k]     = current_t
        phase_res[k] = ctx.phase
        if hasattr(ctx, 'current_L_rest'):
            L_log[:, k] = ctx.current_L_rest

        if (current_t - last_guid >= guid_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guid = current_t

        n_sub  = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if k % 200 == 0:
            print(f"\r    {current_t / SIM_END * 100:.0f}%", end='')

    print(f"\r    done in {time.time() - t0_cpu:.1f} s")
    return t_res, y_res, phase_res, L_log, p


# ==============================================================================
# SETTLING TIME
# ==============================================================================
def settling_time(t, signal, threshold: float, t_impulse: float):
    mask  = t >= t_impulse
    t_a   = t[mask];  sig_a = signal[mask]
    if len(t_a) < 2:
        return None
    hold = max(int(2.0 / (t_a[1] - t_a[0])), 1)
    for i in range(len(sig_a) - hold):
        if np.all(np.abs(sig_a[i: i + hold]) < threshold):
            return float(t_a[i])
    return None

# ==============================================================================
# TENSION SPREAD
# ==============================================================================
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

# ==============================================================================
# SYSTEM POWER (Energy waste proxy)
# ==============================================================================
def system_power(y, p, t):
    """
    Calcola un proxy della potenza meccanica totale del sistema.
    Nei multirotori, la potenza indotta è proporzionale a T^(3/2).
    Restituisce la potenza totale istantanea nel tempo.
    """
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

# ==============================================================================
# PLOT
# ==============================================================================
def plot_winch_damping():
    # Step 1: find Phase 4
    p_probe = SysParams()
    p_probe.uav_offsets, p_probe.attach_vecs, _ = formation.compute_geometry(p_probe)
    t_ph4  = find_phase4_start(p_probe)
    gust_t = t_ph4 + 5.0

    zeta_active = p_probe.winch_damping_ratio
    print(f"\n  ζ_active,  gust at t = {gust_t:.1f} s\n")

    # Step 2: simulations
    print("Running winch-damping analysis  (2 simulations) …")
    t, y_pass, ph_pass, L_pass, p = run_sim(0.0, "Passive (No Active Damp)", gust_t_start=gust_t)
    t, y_act,  ph_act,  L_act,  _ = run_sim(10.0, "Active (k_damp = 20)", gust_t_start=gust_t)

    N = p.N

    def pay_pos(y): return y[3*N    : 3*N+3,  :]
    def pay_vel(y): return y[6*N+6  : 6*N+9,  :]
    def pay_omg(y): return y[6*N+9  : 6*N+12, :]

    # Reference = position 5 steps before gust
    k_ref = max(0, np.searchsorted(t, gust_t) - 5)
    rp    = pay_pos(y_pass)[:2, k_ref]
    ra    = pay_pos(y_act )[:2, k_ref]

    disp_pass = np.linalg.norm(pay_pos(y_pass)[:2, :] - rp[:, None], axis=0)
    disp_act  = np.linalg.norm(pay_pos(y_act )[:2, :] - ra[:, None], axis=0)
    vxy_pass  = np.linalg.norm(pay_vel(y_pass)[:2, :], axis=0)
    vxy_act   = np.linalg.norm(pay_vel(y_act )[:2, :], axis=0)
    omg_pass  = np.linalg.norm(pay_omg(y_pass), axis=0)
    omg_act   = np.linalg.norm(pay_omg(y_act ), axis=0)

    ts_pass = settling_time(t, disp_pass, 0.05, gust_t)
    ts_act  = settling_time(t, disp_act,  0.05, gust_t)

    # Trim to region of interest: 3 s before gust → end
    t0v  = max(gust_t - 3.0, t[0])
    view = t >= t0v

    C_PASS = '#E63946'
    C_ACT  = '#1D7EC2'

    fig, axs = plt.subplots(5, 1, figsize=(13, 17), sharex=True)
    

    v_gust_mag = 8.0  # La velocità della raffica che hai impostato in run_sim
    A_side = 2.0 * p.R_disk * p.pay_h # Area laterale del cilindro esposta al vento
    force_newtons = 0.5 * p.rho * p.Cd_pay * A_side * (v_gust_mag**2)

    for i, ax in enumerate(axs):
        shade_phases(ax, t[view], ph_pass[view])
        
        # Mettiamo la label nella legenda SOLO per il primo grafico (i == 0)
        if i == 0:
            ax.axvline(gust_t, color='black', ls='--', lw=1.2, zorder=5, 
                       label=f'Wind Gust (~{force_newtons:.1f} N)')
        else:
            # Per gli altri grafici disegniamo la linea ma SENZA label
            ax.axvline(gust_t, color='black', ls='--', lw=1.2, zorder=5)

    # Panel A
    ax = axs[0]
    ax.plot(t[view], disp_pass[view] * 100, color=C_PASS, lw=1.8,
            label='Passive')
    ax.plot(t[view], disp_act[view]  * 100, color=C_ACT,  lw=1.8,
            label=f'Active')
    if ts_pass is not None and ts_pass >= t0v:
        ax.axvline(ts_pass, color=C_PASS, ls=':', lw=1.3,
                   label=f'Settled passive: {ts_pass - gust_t:.1f} s')
    if ts_act is not None and ts_act >= t0v:
        ax.axvline(ts_act,  color=C_ACT,  ls=':', lw=1.3,
                   label=f'Settled active:  {ts_act  - gust_t:.1f} s')
        
    # --- FRECCIA VERDE RIDUZIONE DISPLACEMENT ---
    # Analizziamo l'ultimo tratto stabile del grafico (es. dopo i 45 secondi)
    mask_eval = t[view] > 45.0
    if np.any(mask_eval):
        # Calcolo della riduzione percentuale (sulle medie per onestà matematica)
        mean_pass_disp = np.mean(disp_pass[view][mask_eval]) * 100
        mean_act_disp  = np.mean(disp_act[view][mask_eval]) * 100
        red_disp = (1.0 - mean_act_disp / max(mean_pass_disp, 1e-6)) * 100
        
        # Trova il tempo esatto in cui c'è l'ultimo grande PICCO ROSSO
        idx_peak = np.argmax(disp_pass[view][mask_eval])
        t_peak = t[view][mask_eval][idx_peak]
        
        # Aggancia la freccia esattamente in cima alla cresta dell'onda rossa
        y_top = disp_pass[view][mask_eval][idx_peak] * 100
        # Aggancia la base della freccia alla curva blu nello stesso istante
        y_bot = disp_act[view][mask_eval][idx_peak] * 100
        
        x_arr = t_peak
        
        # Disegna la freccia
        ax.annotate('', xy=(x_arr, y_top), xytext=(x_arr, y_bot),
                    arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
        
        # Aggiunge il testo spostato un po' a SINISTRA per non coprire l'onda
        ax.text(x_arr - 0.6, ((y_top + y_bot) / 2)+1, 
                f'Avg. Reduction:\n{red_disp:.1f}%',
                fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')
    
    ax.set_ylabel('Horiz. Displacement [cm]', fontsize=10)
    ax.set_title('Payload Displacement', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, ncol=2);  ax.grid(True, ls=':', alpha=0.5)
    ax.set_ylim(bottom=0)

    # Panel B
    ax = axs[1]
    ax.plot(t[view], vxy_pass[view], color=C_PASS, lw=1.5, label='Passive')
    ax.plot(t[view], vxy_act[view],  color=C_ACT,  lw=1.5, label='Active')

    # --- FRECCIA VERDE RIDUZIONE VELOCITY ---
    mask_eval_v = t[view] > 45.0
    if np.any(mask_eval_v):
        mean_pass_vel = np.mean(vxy_pass[view][mask_eval_v])
        mean_act_vel  = np.mean(vxy_act[view][mask_eval_v])
        red_vel = (1.0 - mean_act_vel / max(mean_pass_vel, 1e-6)) * 100
        
        # Trova il picco della curva di velocità rossa
        idx_peak_v = np.argmax(vxy_pass[view][mask_eval_v])
        t_peak_v = t[view][mask_eval_v][idx_peak_v]
        
        y_top_v = vxy_pass[view][mask_eval_v][idx_peak_v]
        y_bot_v = vxy_act[view][mask_eval_v][idx_peak_v]
        
        x_arr_v = t_peak_v
        
        ax.annotate('', xy=(x_arr_v, y_top_v), xytext=(x_arr_v, y_bot_v),
                    arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.5))
        
        ax.text(x_arr_v - 0.5, ((y_top_v + y_bot_v) / 2)+1, 
                f'Avg. Reduction:\n{red_vel:.1f}%',
                fontsize=9, color='darkgreen', fontweight='bold', ha='right', va='center')

    ax.set_ylabel('Horiz. Velocity [m/s]', fontsize=10)
    ax.set_title('Payload Horizontal Velocity', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9);  ax.grid(True, ls=':', alpha=0.5)
    ax.set_ylim(bottom=0)

    # Panel D
    ax = axs[2]
    colors_uav = plt.cm.tab10(np.linspace(0, 0.9, N))
    for i in range(N):
        ax.plot(t[view], L_pass[i, view], color=colors_uav[i], lw=0.9, ls='--', alpha=0.5)
        ax.plot(t[view], L_act[i,  view], color=colors_uav[i], lw=1.4,
                label=f'UAV {i+1}')
    ax.plot([], [], 'dimgray', lw=0.9, ls='--', label='Passive (all)')
    ax.plot([], [], 'dimgray', lw=1.4,           label='Active  (all)')
    ax.set_ylabel('Cable Length [m]', fontsize=10)
    ax.set_title('Commanded Cable Lengths (Winch Modulation)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, ncol=3);  ax.grid(True, ls=':', alpha=0.5)

    # ── Panel D: Cable Tension Spread (axs[3]) ────────────────────────────────
    ax = axs[3]
    
    sp_pass_raw = tension_spread(y_pass, p, t)
    sp_act_raw = tension_spread(y_act, p, t)
    
    # Smussamento per leggibilità
    smooth_window = int(0.5 / 0.02) # Usa 0.02 che è il dt
    kernel = np.ones(smooth_window) / smooth_window
    sp_pass = np.convolve(sp_pass_raw, kernel, mode='same')
    sp_act = np.convolve(sp_act_raw, kernel, mode='same')

    ax.plot(t[view], sp_pass[view], color=C_PASS, lw=1.2, label='$T_{max} - T_{min}$ – passive')
    ax.plot(t[view], sp_act[view], color=C_ACT, lw=1.2, label='$T_{max} - T_{min}$ – active')
    
    ax.set_ylabel('$T_{max} - T_{min}$ [N]', fontsize=10)
    ax.set_title('Cable Tension Spread (Load Imbalance)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_ylim(bottom=0)

    # ── Panel E: System Power (SPOSTATO SU axs[4]) ────────────────────────────
    ax = axs[4]
    
    dt = t[1] - t[0]
    pow_pass_raw = system_power(y_pass, p, t)
    pow_act_raw  = system_power(y_act, p, t)
    
    # Smussamento (Media mobile di 1 secondo)
    smooth_window_p = int(1.0 / dt) 
    kernel_p = np.ones(smooth_window_p) / smooth_window_p
    pow_pass = np.convolve(pow_pass_raw, kernel_p, mode='same')
    pow_act  = np.convolve(pow_act_raw, kernel_p, mode='same')

    ax.plot(t[view], pow_pass[view], color=C_PASS, lw=1.5, label='System Power – passive')
    ax.plot(t[view], pow_act[view],  color=C_ACT,  lw=1.5, label='System Power – active')
            
    # Calcolo energia totale consumata (Solo a partire dal momento della raffica)
    mask_energy = (t >= gust_t)
    if np.any(mask_energy):
        energy_pass = np.trapz(pow_pass_raw[mask_energy], dx=dt)
        energy_act  = np.trapz(pow_act_raw[mask_energy], dx=dt)
        
        # Calcolo dell'incremento percentuale di energia
        energy_diff = ((energy_act / energy_pass) - 1.0) * 100
        
        testo_energia = f"Energy overhead: +{abs(energy_diff):.1f}%"
        
        # Annotazione sul grafico
        y_pos = max(np.mean(pow_pass[mask_energy]), np.mean(pow_act[mask_energy])) * 1.06
        x_target = t[-1] - 2.0  
        ax.text(x_target, y_pos, testo_energia, 
                fontsize=10, color='indigo', fontweight='bold', ha='right')

    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel(r'Power ($\sum T^{1.5}$)', fontsize=10)
    ax.set_title('Average System Power (Energy Consumption)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, ls=':', alpha=0.5)

    ax.set_ylim(bottom=117.0, top=140.0)

    plt.xlim(right=60.0) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Lascia un piccolo margine extra
    
    # Oppure usa questo per un controllo millimetrico:
    plt.subplots_adjust(
        left=0.1,    # Margine sinistro
        right=0.95,  # Margine destro
        top=0.95,    # Margine superiore
        bottom=0.1,  # Aumenta questo valore (es. da 0.05 a 0.1) per la scritta "Time"
        hspace=0.3   # Spazio verticale tra i vari pannelli
    )
    
    plt.show()


if __name__ == "__main__":
    plot_winch_damping()