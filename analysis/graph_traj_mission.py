"""
graph_trajectory.py
--------------------
Spatial trajectory and cable kinematics visualization.
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

# ==============================================================================
# PHASE STYLE CATALOGUE
# ==============================================================================
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
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        t_start = t_res[idx[0]]
        t_end   = t_res[idx[-1]]
        style   = PHASE_STYLES.get(ph, dict(color='white'))
        ax.axvspan(t_start, t_end, color=style['color'], alpha=alpha, zorder=0)

# ==============================================================================
# SIMULATION RUNNER (Adattato per registrare la lunghezza dei cavi)
# ==============================================================================
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
    
    # NUOVO: Array per registrare la lunghezza comandata dal winch
    L_winch_cmd_res = np.zeros((p.N, steps))

    current_x          = x0.copy()
    current_t          = 0.0
    last_guidance_time = -1.0
    guidance_dt        = p.optimization_dt

    print(f"Running simulation for Trajectory Analysis N={p.N}, T={SIM_DURATION}s …")
    t0_cpu = time.time()

    for k in range(steps):
        y_res[:, k]  = current_x
        t_res[k]     = current_t
        phase_res[k] = ctx.phase
        L_winch_cmd_res[:, k] = ctx.current_L_rest  # Log lunghezza comandata

        if (current_t - last_guidance_time >= guidance_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guidance_time = current_t

        n_sub = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if k % 100 == 0:
            print(f"\r  {current_t / SIM_DURATION * 100:5.1f}%  t={current_t:.2f}s  phase={ctx.phase}", end='')

    print(f"\nDone in {time.time() - t0_cpu:.1f}s")
    return t_res, y_res, phase_res, L_winch_cmd_res, p

def parse_state(y, p):
    N = p.N
    uav_pos  = y[: 3 * N, :].reshape(3, N, -1, order='F')     
    pay_pos  = y[3*N     : 3*N+3,  :]                          
    pay_att  = y[3*N+3   : 3*N+6,  :]                          
    return uav_pos, pay_pos, pay_att

# ==============================================================================
# PLOTTING LOGIC
# ==============================================================================
def plot_trajectory_and_cables(t_res, y_res, phase_res, L_winch_cmd_res, p):
    uav_pos, pay_pos, pay_att = parse_state(y_res, p)
    steps = len(t_res)

    # --- Calcolo Lunghezza Fisica Reale dei Cavi ---
    L_actual_res = np.zeros((p.N, steps))
    for k in range(steps):
        R_pay = physics.get_rotation_matrix(*pay_att[:, k])
        for i in range(p.N):
            # Posizione reale del punto di aggancio nel mondo
            att_world = R_pay @ p.attach_vecs[:, i]
            anchor_pos = pay_pos[:, k] + att_world
            # Distanza euclidea tra drone e punto di aggancio
            L_actual_res[i, k] = np.linalg.norm(uav_pos[:, i, k] - anchor_pos)

    # Palette colori per i droni (simile all'immagine)
    colors = plt.cm.tab10(np.linspace(0, 0.9, p.N))

    fig = plt.figure(figsize=(14, 18))
    
    # Riduciamo il campionamento per disegni vettoriali più leggeri (skip)
    skip = max(1, steps // 1000)

    # ── Panel 1: Traiettoria Top-Down (X-Y) ───────────────────────────────────
    # ── Panel 1: Traiettoria Top-Down (X-Y) ───────────────────────────────────
    ax1 = fig.add_subplot(2, 1, 1)
    
    # Plot Droni
    for i in range(p.N):
        ax1.plot(uav_pos[0, i, ::skip], uav_pos[1, i, ::skip], 
                 color=colors[i], lw=1.5, alpha=0.5, label=f'UAV {i+1}')
        
    # Plot Payload
    ax1.plot(pay_pos[0, ::skip], pay_pos[1, ::skip], 
             color='black', linestyle='--', lw=2.5, label='Payload')
    
    # Start and Goal Markers
    ax1.plot(pay_pos[0, 0], pay_pos[1, 0], '^', color="green", markersize=10, label='Start')
    goal_x, goal_y = p.payload_goal_pos[0], p.payload_goal_pos[1]
    ax1.plot(goal_x, goal_y, '*', color='blue', markersize=10, label='Goal (XY)')

    ax1.set_xlabel('X Position [m]', fontsize=11)
    ax1.set_ylabel('Y Position [m]', fontsize=11)
    ax1.set_title('Top-Down Trajectory (X-Y Plane)', fontweight='bold', fontsize=13)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # ---> LA SOLUZIONE: Usa 'auto' invece di 'equal' per strecciare il grafico <---
    ax1.set_aspect('auto') 
    ax1.set_xlim(-10, 40)
    ax1.legend(loc='upper left', fontsize=10)

    # ── Panel 2: Traiettoria Laterale (X-Z) ───────────────────────────────────
    # ---> FONDAMENTALE: aggiungi sharex=ax1 qui per incollare l'asse X al Panel 1 <---
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1) 
    
    # Plot Droni
    for i in range(p.N):
        ax2.plot(uav_pos[0, i, ::skip], uav_pos[2, i, ::skip], 
                 color=colors[i], alpha=0.5, lw=1.5)
        
    # Plot Payload
    ax2.plot(pay_pos[0, ::skip], pay_pos[2, ::skip], 
             color='black', linestyle='--', lw=2.5)
    
    # Start, Goal e Linea di Terra
    ax2.plot(pay_pos[0, 0], pay_pos[2, 0], '^', color="green", markersize=10)
    ax2.plot(goal_x, p.payload_goal_pos[2], '*', color='blue', markersize=10, label='Goal (XZ)')
    ax2.axhline(p.floor_z, color='saddlebrown', lw=2, linestyle='-', label='Ground', zorder=0)

    ax2.set_xlabel('X Position [m]', fontsize=11)
    ax2.set_ylabel('Altitude Z [m]', fontsize=11)
    ax2.set_title('Side-View Trajectory (X-Z Plane)', fontweight='bold', fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=10)

    ## ── Panel 3: Cinematica dei Cavi (Lunghezza vs Tempo) ─────────────────────
    #ax3 = fig.add_subplot(3, 1, 3)
    #shade_phases(ax3, t_res, phase_res)
    #
    #for i in range(p.N):
    #    # Lunghezza fisica reale (solida)
    #    ax3.plot(t_res[::skip], L_actual_res[i, ::skip], 
    #             color=colors[i], lw=1.8, label=f'UAV {i+1} Actual L')
    #    # Lunghezza comandata dal Winch (tratteggiata e più leggera)
    #    ax3.plot(t_res[::skip], L_winch_cmd_res[i, ::skip], 
    #             color=colors[i], lw=1.0, linestyle='--', alpha=0.6)
#
    ## Evidenzia la lunghezza nominale di base
    #ax3.axhline(p.L, color='gray', linestyle=':', lw=1.5, label='Nominal Length')
#
    #ax3.set_xlabel('Time [s]', fontsize=11)
    #ax3.set_ylabel('Cable Length [m]', fontsize=11)
    #ax3.set_title('Winch Kinematics: Actual Length (Solid) vs Commanded (Dashed)', fontweight='bold', fontsize=13)
    #ax3.grid(True, linestyle=':', alpha=0.7)
    #
    ## Custom Legend per le fasi + cavi
    #present_phases = np.unique(phase_res)
    #legend_patches = [mpatches.Patch(color=PHASE_STYLES[ph]['color'], label=PHASE_STYLES[ph]['label'], alpha=0.5)
    #                  for ph in present_phases if ph in PHASE_STYLES]
    #
    #handles, labels = ax3.get_legend_handles_labels()
    ## Mostriamo solo le label delle lunghezze reali per non affollare la legenda
    #filtered_handles = [h for h, l in zip(handles, labels) if 'Actual' in l or 'Nominal' in l]
    #filtered_labels = [l for l in labels if 'Actual' in l or 'Nominal' in l]
    #
    #ax3.legend(filtered_handles + legend_patches, filtered_labels + [p.get_label() for p in legend_patches],
    #           loc='upper left', fontsize=9, ncol=3)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.35)
    plt.show()

if __name__ == "__main__":
    t_res, y_res, phase_res, L_winch_cmd_res, p = run_simulation()
    plot_trajectory_and_cables(t_res, y_res, phase_res, L_winch_cmd_res, p)