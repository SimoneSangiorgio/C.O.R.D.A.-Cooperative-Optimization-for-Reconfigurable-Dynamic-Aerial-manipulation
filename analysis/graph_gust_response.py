"""
graph_gust_response.py
----------------------
Dynamic analysis: aerodynamic disturbance rejection under wind gusts.

Three parallel simulations are run, each with a different adaptation strategy,
subject to the same wind gust sequence defined in the parameters.

Strategies compared
-------------------
  • Rigid          (λ_shape=0, λ_aero=0) – no adaptation
  • Shape-Adaptive (λ_shape=1, λ_aero=0) – elliptical formation deformation
  • Aero-Tilt      (λ_shape=0, λ_aero=1) – payload tilts into the wind

Gust profile
------------
Three gusts are applied sequentially during navigation (Phase 4):
  • Gust 1: lateral push  [0,  +6, 0]  m/s  for 5 s
  • Gust 2: strong head wind [−8, 0, 0] m/s  for 6 s
  • Gust 3: diagonal gust  [+5, +5, 0] m/s  for 4 s

Metrics plotted
---------------
  Panel A — Payload position error ‖p − p_ref‖ [m]
  Panel B — Payload attitude error (roll² + pitch²)^{0.5} [deg]
  Panel C — Wind force magnitude on the payload [N]
  Panel D — Maximum cable tension [N]
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import formation as formation
from mission import MissionContext, equations_of_motion, update_guidance
from parameters import SysParams
import physics


# ==============================================================================
# PHASE / GUST ANNOTATION
# ==============================================================================
PHASE_COLORS = {0:'#AAAAAA',1:'#FFD700',2:'#90EE90',3:'#87CEEB',
                4:'#FFA07A',5:'#DDA0DD',6:'#98FB98'}

GUST_STYLE = dict(color='lightsteelblue', alpha=0.25, zorder=0)

def shade_phases(ax, t, phases):
    for ph in np.unique(phases):
        mask = phases == ph
        idx  = np.where(mask)[0]
        ax.axvspan(t[idx[0]], t[idx[-1]],
                   color=PHASE_COLORS.get(ph, 'white'), alpha=0.10, zorder=0)

def shade_gusts(ax, gusts):
    for g in gusts:
        ax.axvspan(g['t_start'], g['t_start'] + g['duration'], **GUST_STYLE)


# ==============================================================================
# WIND FORCE HELPER (sphere only for simplicity)
# ==============================================================================
def wind_force_magnitude(p, v_wind_world: np.ndarray, attitude) -> float:
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    if wind_mag < 1e-3:
        return 0.0
    proj_area = np.pi * p.R_disk ** 2   # sphere: orientation-independent
    return 0.5 * p.rho * p.Cd_pay * proj_area * wind_mag ** 2


# ==============================================================================
# SYSTEM POWER (Energy waste proxy)
# ==============================================================================
def system_power(y, p, t):
    N, dt = p.N, t[1] - t[0]
    uvel = y[3*N+6 : 6*N+6, :]
    g_v = np.array([[0], [0], [-9.81]])
    P_tot = np.zeros(len(t))
    for i in range(N):
        acc_i = np.gradient(uvel[i*3 : (i+1)*3, :], dt, axis=1)
        F_i = np.linalg.norm(p.m_drone * (acc_i - g_v), axis=0)
        P_tot += F_i**(1.5)
    return P_tot


# ==============================================================================
# SIMULATION
# ==============================================================================
GUST_SCHEDULE = [
    {'t_start': 25.0, 'duration': 5.0, 'vec': np.array([ 0.0,  6.0, 0.0]), 'ramp': 0.8},
    {'t_start': 35.0, 'duration': 6.0, 'vec': np.array([-8.0,  0.0, 0.0]), 'ramp': 1.0},
    {'t_start': 47.0, 'duration': 4.0, 'vec': np.array([ 5.0,  5.0, 0.0]), 'ramp': 0.6},
]


def run_sim(lambda_shape: float, lambda_aero: float, label: str):
    p = SysParams()

    p.initial_wind_vec = np.array([8, 4, 0.0]) 
    p.F_ref = 8.0

    p.lambda_shape    = lambda_shape
    p.lambda_aero     = lambda_aero
    p.enable_gusts    = False
    p.gust_schedule   = copy.deepcopy(GUST_SCHEDULE)

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

    SIM_DURATION = 65.0
    dt    = 0.02
    steps = int(SIM_DURATION / dt)

    N = p.N
    y_res      = np.zeros((len(x0), steps))
    t_res      = np.zeros(steps)
    phase_res  = np.zeros(steps, dtype=int)
    thrust_res = np.zeros((N, steps))
    wind_log   = np.zeros(steps)   # wind force magnitude on payload
    ref_pos_res = np.zeros((3, steps))

    current_x = x0.copy();  current_t = 0.0
    last_guid  = -1.0;       guid_dt   = p.optimization_dt

    print(f"  Simulating: {label} …")
    t0_cpu = time.time()

    for k in range(steps):
        y_res[:, k]  = current_x
        t_res[k]     = current_t
        phase_res[k] = ctx.phase

        # Log wind force magnitude
        pay_att_k = current_x[3*N+3 : 3*N+6]
        wind_log[k] = wind_force_magnitude(p, p.wind_vel, pay_att_k)

        if (current_t - last_guid >= guid_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guid = current_t

        if ctx.current_ref is not None:
            ref_pos_res[:, k] = ctx.current_ref['pos']
        else:
            ref_pos_res[:, k] = current_x[3*N : 3*N+3]

        n_sub  = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if hasattr(ctx, 'last_u_acc'):
            for i in range(N):
                acc_cmd    = ctx.last_u_acc[:, i]
                thrust_vec = p.m_drone * (acc_cmd + np.array([0, 0, 9.81]))
                thrust_res[i, k] = np.linalg.norm(thrust_vec)

        if k % 200 == 0:
            print(f"\r    {current_t / SIM_DURATION * 100:.0f}%", end='')

    print(f"\r    done in {time.time() - t0_cpu:.1f}s")
    return t_res, y_res, phase_res, thrust_res, wind_log, ref_pos_res, p


# ==============================================================================
# MAIN
# ==============================================================================
def plot_gust_response():
    print("Running gust-response analysis (3 simulations) …")

    results = [
        run_sim(0.0, 0.0, 'Rigid          (λ_s=0, λ_a=0)'),
        run_sim(1.0, 0.0, 'Shape-Adaptive (λ_s=1, λ_a=0)'),
        run_sim(0.0, 1.0, 'Aero-Tilt      (λ_s=0, λ_a=1)'),
    ]

    labels = ['Rigid', 'Shape-Adaptive', 'Aero-Tilt']
    colors = ['dimgray', 'mediumseagreen', 'royalblue']
    lstyle = ['--', '-', '-']

    fig, axs = plt.subplots(5, 1, figsize=(13, 19), sharex=True)
    #fig.suptitle('Aerodynamic Disturbance Rejection — Gust Response Comparison',
    #             fontsize=14, fontweight='bold')

    t_ref = results[0][0]
    ph_ref = results[0][2]
    # Creiamo una maschera per vedere solo la fase di navigazione (Phase 4)
    mask_ph4 = (ph_ref == 4)
    t_view = t_ref[mask_ph4]
    ph_ref = results[0][2]
    p_ref  = results[0][5]

    for ax in axs:
        shade_phases(ax, t_ref, ph_ref)
        shade_gusts(ax, GUST_SCHEDULE)

    # Gust legend patch
    gust_patch = mpatches.Patch(color='lightsteelblue', alpha=0.6, label='Active gust window')

    # ── Panel A: Position error ───────────────────────────────────────────────
    ax = axs[0]
    for (t, y, ph, thrust, wind, ref_pos, p), lab, col, ls in zip(results, labels, colors, lstyle):
        N = p.N
        pos = y[3*N : 3*N+3, :]
        err = np.linalg.norm(pos - ref_pos, axis=0) 
        
        # SMUSZAMENTO: Media mobile di 0.4s (20 campioni a 50Hz)
        err_smooth = np.convolve(err, np.ones(20)/20, mode='same')
        ax.plot(t[mask_ph4], err_smooth[mask_ph4], color=col, ls=ls, lw=1.6, label=lab)
    
    ax.set_ylabel('Position Error [m]', fontsize=10)
    ax.set_title('Payload Trajectory Tracking Error (Phase 4 - Smoothed)', fontweight='bold', fontsize=11)

    # ── Panel B: Attitude error ───────────────────────────────────────────────
    ax = axs[1]
    for i in range(len(results)):
        lab = labels[i]
        
        # Escludiamo l'Aero-Tilt dal grafico dell'errore (il suo target non è 0 gradi!)
        if 'Aero-Tilt' in lab:
            continue
            
        res = results[i]
        t = res[0]
        y = res[1]
        p = res[-1]  # p è sempre l'ultimo parametro restituito
        
        N   = p.N
        att = y[3*N+3 : 3*N+6, :]
        ang_err = np.degrees(np.sqrt(att[0,:]**2 + att[1,:]**2))
        ax.plot(t, ang_err, color=colors[i], ls=lstyle[i], lw=1.5, label=lab)

    ax.set_ylabel('Attitude Error  √(φ²+θ²) [deg]', fontsize=10)
    ax.set_title('Payload Attitude Error', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Aggiungiamo la legenda per le due linee rimaste (in alto a sinistra per non dar fastidio)
    ax.legend(fontsize=9, loc='upper left')

    # Aggiungiamo l'"agendina" esplicativa per l'Aero-Tilt in alto a destra
    testo_aero = ("* Aero-Tilt omitted:\n"
                  "The strategy actively tilts the payload\n"
                  r"to track the total force vector ($f_{tot}$)")
    
    props = dict(boxstyle='round,pad=0.4', facecolor='aliceblue', 
                 edgecolor='royalblue', alpha=0.9)
    
    ax.text(0.98, 0.90, testo_aero, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', 
            bbox=props, color='midnightblue', fontweight='bold')

    # ── Panel C: Wind force on payload ────────────────────────────────────────
    ax = axs[2]
    # All simulations see the same wind (same gust schedule), plot once
    ax.plot(t_ref, results[0][4], color='steelblue', lw=1.5,
            label='Wind force on payload [N]')
    ax.set_ylabel('Wind Force [N]', fontsize=10)
    ax.set_title('Aerodynamic Force Magnitude on Payload', fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)

    # ── Panel D: Maximum cable tension ───────────────────────────────────────
    ax = axs[3]
    p_ref_actual = results[0][6] 
    m_tot = p_ref_actual.m_payload + p_ref_actual.m_liquid
    T_max_limit = (p_ref_actual.F_max_thrust - p_ref_actual.m_drone * p_ref_actual.g) * p_ref_actual.k_limit

    # Aggiungi 'ref' nello scompattamento qui sotto:
    for (t, y, ph, thrust, wind, ref, p), lab, col, ls in zip(results, labels, colors, lstyle):
        ax.plot(t, np.max(thrust, axis=0), color=col, ls=ls, lw=1.5, label=lab)
    ax.axhline(T_max_limit, color='crimson', linestyle=':',
               label=f'Saturation limit = {T_max_limit:.1f} N')
    ax.set_ylabel('Max Cable Tension [N]', fontsize=10)
    ax.set_title('Maximum Cable Tension', fontweight='bold', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=8, ncol=4)

    # ── Panel E: System Power & Energy Overhead ───────────────────────────────
    ax = axs[4]
    energies = []
    
    for (t, y, ph, thrust, wind, ref_pos, p), lab, col, ls in zip(results, labels, colors, lstyle):
        p_raw = system_power(y, p, t)
        
        # SMUSSAMENTO POTENZA (0.5s) per pulizia visiva
        p_smooth = np.convolve(p_raw, np.ones(25)/25, mode='same')
        ax.plot(t[mask_ph4], p_smooth[mask_ph4], color=col, ls=ls, lw=1.5, label=lab)
        
        # Calcolo integrale dell'energia solo nella fase 4
        energy = np.trapz(p_raw[mask_ph4], dx=0.02)
        energies.append(energy)

    # Annotazioni Energy Overhead (Rigid è il riferimento all'indice 0)
    e_rigid = energies[0]
    for i, (energy, lab, col) in enumerate(zip(energies, labels, colors)):
        overhead = (energy / e_rigid - 1.0) * 100
        # Posizioniamo il testo in modo ordinato in alto a sinistra del grafico
        ax.text(0.02, 0.92 - i*0.08, f"{lab} Energy: {overhead:+.1f}%", 
                transform=ax.transAxes, color=col, fontweight='bold', fontsize=9)

    ax.set_ylabel(r'Power ($\sum T^{1.5}$)', fontsize=10)
    ax.set_title('System Power Consumption', fontweight='bold', fontsize=11)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylim(110, 150) # Range tipico per evidenziare le differenze

    for ax in axs:
        ax.set_xlim(t_view[0], t_view[-1])

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
    plot_gust_response()