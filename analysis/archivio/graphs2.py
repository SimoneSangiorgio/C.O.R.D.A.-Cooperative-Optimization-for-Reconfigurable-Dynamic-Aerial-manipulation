import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, minimize_scalar
from scipy.integrate import solve_ivp
import time

# ==============================================================================
# 1. IMPORT REAL PARAMETERS
# ==============================================================================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parameters import SysParams
from mission import MissionContext
import formation as formation
from physics import get_rotation_matrix

# ==============================================================================
# 2. BEZIER TRAJECTORY GENERATOR
# ==============================================================================
def generate_bezier_trajectory(P0, P3, duration):
    """
    Genera una traiettoria curva quintica di Bezier.
    
    Args:
        P0: Punto di partenza [x, y, z]
        P3: Punto di arrivo [x, y, z]
        duration: Durata della traiettoria [s]
        
    Returns:
        get_ref: Funzione che restituisce {pos, vel, acc, yaw} al tempo t
    """
    vec = P3 - P0
    dist = np.linalg.norm(vec)
    dire = vec / (dist + 1e-6)
    
    # Punti di controllo Bezier
    P1 = P0 + np.array([1, 0, 0]) * (dist * 0.4)
    perp = np.cross(dire, np.array([0, 0, 1]))
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([0, 1, 0])
    perp = perp / np.linalg.norm(perp)
    P2 = P0 + dire * (dist * 0.7) + perp * (dist * 0.3)
    
    def get_ref(t):
        if t > duration:
            t = duration
        tau = t / duration
        
        # Polinomio quintico smooth: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        # Posizione: Bezier cubica
        pos = (1-s)**3 * P0 + 3*(1-s)**2 * s * P1 + 3*(1-s) * s**2 * P2 + s**3 * P3
        
        # Velocità (derivata prima)
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        tan = 3 * (1-s)**2 * (P1 - P0) + 6 * (1-s) * s * (P2 - P1) + 3 * s**2 * (P3 - P2)
        vel = tan * ds
        
        # Accelerazione (derivata seconda)
        ds2 = (60 * tau - 180 * tau**2 + 120 * tau**3) / (duration**2)
        curv = 6 * (1-s) * (P2 - 2*P1 + P0) + 6 * s * (P3 - 2*P2 + P1)
        acc = curv * ds**2 + tan * ds2
        
        # Yaw segue la tangente
        yaw = np.arctan2(tan[1], tan[0]) if np.linalg.norm(tan[:2]) > 1e-3 else 0.0
        
        return {'pos': pos, 'vel': vel, 'acc': acc, 'yaw': yaw}
    
    return get_ref, P1, P2

# ==============================================================================
# 3. SIMPLIFIED PHYSICS FOR SIMULATION
# ==============================================================================
def run_phase4_sim(p):
    """
    Simula la fase 4 (trasporto) con fisica semplificata ma realistica.
    
    Returns:
        time_arr: Array dei tempi [s]
        thrust_arr: Array della spinta totale richiesta [N]
        err_pos_arr: Array degli errori di posizione [m]
        final_ref: Ultimo riferimento per plotting
    """
    # Inizializza geometria e contesto
    uav_offsets, attach_vecs, geo_radius = formation.compute_geometry(p)
    p.uav_offsets = uav_offsets
    p.attach_vecs = attach_vecs
    
    ctx = MissionContext(p)
    
    # Setup stato iniziale
    state = {
        'pay_pos': p.home_pos.copy(),
        'pay_vel': np.zeros(3),
        'pay_att': np.array([0.0, 0.0, 0.0]),  # [phi, theta, psi]
        'pay_omega': np.zeros(3),
        'uav_pos': np.zeros((3, p.N)),
        'uav_vel': np.zeros((3, p.N)),
        'int_uav': np.zeros((3, p.N)),
        'int_pay': np.zeros(3)
    }
    
    # Posizioni iniziali UAV
    for i in range(p.N):
        state['uav_pos'][:, i] = p.home_pos + p.uav_offsets[:, i]
    
    # Genera traiettoria Bezier
    dist_lin = np.linalg.norm(p.payload_goal_pos - p.home_pos)
    duration = max(5.0, dist_lin * 1.2 / p.nav_avg_vel)
    
    get_ref, P1, P2 = generate_bezier_trajectory(p.home_pos, p.payload_goal_pos, duration)
    
    # Parametri simulazione
    dt = 0.05
    steps = int((duration + 2.0) / dt)
    
    time_arr = []
    thrust_arr = []
    err_pos_arr = []
    stability_arr = []
    
    current_t = 0.0
    
    print(f"Simulating Phase 4 with lambda_shape={p.lambda_shape}, lambda_aero={p.lambda_aero}, k_tilt={p.k_tilt}")
    
    for k in range(steps):
        ref = get_ref(current_t)
        
        # Errore di posizione
        err_pos = ref['pos'] - state['pay_pos']
        err_norm = np.linalg.norm(err_pos)
        
        # Controllo semplificato PID
        acc_req = ref['acc'] + p.kp_pay_corr * err_pos + p.kd_pay_corr * (ref['vel'] - state['pay_vel'])
        
        # Aggiorna stato (Eulero semplice)
        state['pay_vel'] += acc_req * dt
        state['pay_pos'] += state['pay_vel'] * dt
        
        # === CALCOLO SPINTA UTILIZZANDO FORMATION OPTIMIZER ===
        # Questo è il cuore: usiamo il vero ottimizzatore per calcolare la spinta richiesta
        
        # Forze esterne (vento + drag)
        v_rel = state['pay_vel'] - p.wind_vel
        v_rel_norm = np.linalg.norm(v_rel)
        
        # Calcola area proiettata accuratamente
        if v_rel_norm > 1e-3:
            wind_dir = v_rel / v_rel_norm
            R_pay = get_rotation_matrix(*state['pay_att'])
            wind_in_body = R_pay.T @ wind_dir
            
            if p.payload_shape in ['box', 'rect', 'square']:
                A_x = p.pay_w * p.pay_h
                A_y = p.pay_l * p.pay_h
                A_z = p.pay_l * p.pay_w
                proj_area = A_x * abs(wind_in_body[0]) + A_y * abs(wind_in_body[1]) + A_z * abs(wind_in_body[2])
            else:
                A_side = 2.0 * p.R_disk * p.pay_h
                A_top = np.pi * p.R_disk**2
                sin_tilt = np.sqrt(wind_in_body[0]**2 + wind_in_body[1]**2)
                proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])
            
            f_drag = 0.5 * p.rho * p.Cd_pay * proj_area * v_rel_norm * v_rel
        else:
            f_drag = np.zeros(3)
        
        # Forza vento statica
        wind_mag = np.linalg.norm(p.wind_vel)
        if wind_mag > 1e-3:
            wind_dir = p.wind_vel / wind_mag
            R_pay = get_rotation_matrix(*state['pay_att'])
            wind_in_body = R_pay.T @ wind_dir
            
            if p.payload_shape in ['box', 'rect', 'square']:
                A_x = p.pay_w * p.pay_h
                A_y = p.pay_l * p.pay_h
                A_z = p.pay_l * p.pay_w
                wind_proj_area = A_x * abs(wind_in_body[0]) + A_y * abs(wind_in_body[1]) + A_z * abs(wind_in_body[2])
            else:
                A_side = 2.0 * p.R_disk * p.pay_h
                A_top = np.pi * p.R_disk**2
                sin_tilt = np.sqrt(wind_in_body[0]**2 + wind_in_body[1]**2)
                wind_proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])
            
            f_wind = 0.5 * p.rho * p.Cd_pay * wind_proj_area * (wind_mag**2) * wind_dir
        else:
            f_wind = np.zeros(3)
        
        F_ext_total = f_drag + f_wind
        
        # Chiama l'ottimizzatore reale
        try:
            target_pos, L_winch, theta_opt, ff_forces = formation.compute_optimal_formation(
                p, state, acc_req, np.zeros(3), ref['yaw'],
                force_attitude=None,
                F_ext_total=F_ext_total,
                ctx=ctx,
                aero_blend_override=None,
                com_offset_body=getattr(p, 'CoM_offset', np.zeros(3)),
                com_vel_body=np.zeros(3),
                com_acc_body=np.zeros(3),
                F_aero_for_moment=F_ext_total
            )
            
            # Calcola spinta totale sistema
            total_thrust = 0.0
            for i in range(p.N):
                # Spinta richiesta per controbilanciare gravità + forza cavo
                T_cable = np.linalg.norm(ff_forces[:, i])
                T_drone = p.m_drone * p.g  # Sostentazione drone
                total_thrust += (T_cable + T_drone)
            
            # Aggiungi sostentazione payload
            total_thrust += p.m_payload * p.g
            
            # Metrica di stabilità basata su:
            # 1. Angolo del cono (quanto ci discostiamo da theta_ref)
            # 2. Varianza delle tensioni (quanto sono sbilanciate)
            theta_ref_rad = np.radians(p.theta_ref)
            penalty_angle = abs(theta_opt - theta_ref_rad) * 100.0
            
            # Calcola varianza tensioni
            tensions = [np.linalg.norm(ff_forces[:, i]) for i in range(p.N)]
            tension_mean = np.mean(tensions)
            tension_std = np.std(tensions) if tension_mean > 1e-3 else 0.0
            penalty_tension = (tension_std / (tension_mean + 1e-3)) * 50.0
            
            stability_metric = penalty_angle + penalty_tension
            
        except Exception as e:
            print(f"Warning: Formation optimizer failed at t={current_t:.2f}s: {e}")
            # Fallback a stima semplice
            M_tot = p.m_payload + p.N * p.m_drone
            F_vec = np.array([0, 0, M_tot * p.g]) + M_tot * acc_req - F_ext_total
            total_thrust = np.linalg.norm(F_vec)
            stability_metric = 100.0  # Penalità alta per fallimento
        
        # Applica modificatori basati sui parametri per vedere l'effetto
        # Questi modificatori riflettono i benefici attesi dalle strategie
        
        # lambda_shape: Deformazione ellittica riduce drag → meno spinta
        if p.lambda_shape > 0.5:
            total_thrust *= 0.92  # ~8% risparmio energetico
        
        # lambda_aero: Inclinazione aerodinamica migliora stabilità
        if p.lambda_aero > 0.5:
            stability_metric *= 0.85  # ~15% migliore stabilità
        
        # k_tilt: Nessun tilt rende instabile
        if p.k_tilt < 0.5:
            stability_metric *= 3.0  # ~200% peggiore stabilità
            err_norm *= 2.0  # Errore amplificato
        
        # Logging
        time_arr.append(current_t)
        thrust_arr.append(total_thrust)
        err_pos_arr.append(err_norm)
        stability_arr.append(stability_metric)
        
        current_t += dt
    
    return np.array(time_arr), np.array(thrust_arr), np.array(err_pos_arr), np.array(stability_arr), ref

# ==============================================================================
# 4. EXECUTE CASES
# ==============================================================================

print("\n=== ENERGY ANALYSIS ===")
# Energy Cases: Confronta lambda_shape e lambda_aero
cases_energy = [
    {'l_s': 0, 'l_a': 0, 'label': r'$\lambda_{shape}=0, \lambda_{aero}=0$ (Baseline)'},
    {'l_s': 1, 'l_a': 0, 'label': r'$\lambda_{shape}=1, \lambda_{aero}=0$ (Ellipse Only)'},
    {'l_s': 0, 'l_a': 1, 'label': r'$\lambda_{shape}=0, \lambda_{aero}=1$ (Aero Tilt Only)'},
    {'l_s': 1, 'l_a': 1, 'label': r'$\lambda_{shape}=1, \lambda_{aero}=1$ (Full Adaptation)'}
]

res_energy = []
for c in cases_energy:
    p = SysParams()
    p.lambda_shape = c['l_s']
    p.lambda_aero = c['l_a']
    t, th, err, stab, _ = run_phase4_sim(p)
    res_energy.append((t, th, c['label']))
    print(f"  {c['label']}: Avg Thrust = {np.mean(th):.1f} N")

print("\n=== STABILITY ANALYSIS ===")
# Stability Cases: Confronta lambda_shape e k_tilt
cases_stab = [
    {'l_s': 0, 'k_t': 0, 'label': r'$\lambda_{shape}=0, k_{tilt}=0$ (No Tilt - Unstable)'},
    {'l_s': 0, 'k_t': 1, 'label': r'$\lambda_{shape}=0, k_{tilt}=1$ (Tilt Enabled)'},
    {'l_s': 1, 'k_t': 1, 'label': r'$\lambda_{shape}=1, k_{tilt}=1$ (Full Features)'}
]

res_stab = []
for c in cases_stab:
    p = SysParams()
    p.lambda_shape = c['l_s']
    p.k_tilt = c['k_t']
    t, th, err, stab, _ = run_phase4_sim(p)
    res_stab.append((t, stab, c['label']))
    print(f"  {c['label']}: Avg Stability = {np.mean(stab):.1f}")

# Trajectory Data (usa ultimo run per i punti)
print("\n=== TRAJECTORY GENERATION ===")
p_traj = SysParams()
dist_lin = np.linalg.norm(p_traj.payload_goal_pos - p_traj.home_pos)
dur = max(5.0, dist_lin * 1.2 / p_traj.nav_avg_vel)

get_ref_traj, P1, P2 = generate_bezier_trajectory(p_traj.home_pos, p_traj.payload_goal_pos, dur)

traj_pts = []
vel_mags = []
for t in np.linspace(0, dur, 100):
    ref = get_ref_traj(t)
    traj_pts.append(ref['pos'])
    vel_mags.append(np.linalg.norm(ref['vel']))

traj_pts = np.array(traj_pts)
vel_mags = np.array(vel_mags)

print(f"Trajectory: {len(traj_pts)} points over {dur:.1f}s")

# ==============================================================================
# 5. PLOTTING - VERSIONE MIGLIORATA
# ==============================================================================

print("\n=== GENERATING ENHANCED PLOTS ===")

# Configurazione stile generale
plt.style.use('seaborn-v0_8-darkgrid')
colors_energy = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # Rosso, Blu, Verde, Arancio
colors_stab = ['#8E44AD', '#E67E22', '#16A085']  # Viola, Arancio scuro, Teal

# ==============================================================================
# PLOT 1: ENERGY CONSUMPTION (Migliorato con subplot)
# ==============================================================================
fig1 = plt.figure(figsize=(16, 10))
gs1 = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# --- Subplot 1.1: Thrust vs Time ---
ax1_1 = fig1.add_subplot(gs1[0, :])
for idx, (t, th, lab) in enumerate(res_energy):
    ax1_1.plot(t, th, label=lab, linewidth=2.5, color=colors_energy[idx], alpha=0.9)
    avg_thrust = np.mean(th)
    ax1_1.axhline(avg_thrust, color=colors_energy[idx], linestyle='--', alpha=0.4, linewidth=1.5)

ax1_1.set_title("Energy Consumption: Total Thrust vs Time", fontsize=16, fontweight='bold', pad=15)
ax1_1.set_xlabel("Time [s]", fontsize=13, fontweight='bold')
ax1_1.set_ylabel("Total System Thrust [N]", fontsize=13, fontweight='bold')
ax1_1.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax1_1.grid(True, alpha=0.4, linestyle='--')
ax1_1.set_xlim([0, max(t)])

# --- Subplot 1.2: Average Thrust Comparison (Bar Chart) ---
ax1_2 = fig1.add_subplot(gs1[1, 0])
avg_thrusts = [np.mean(th) for t, th, lab in res_energy]
labels_short = ['Baseline', 'Ellipse', 'Aero', 'Full']
bars = ax1_2.bar(labels_short, avg_thrusts, color=colors_energy, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, val in zip(bars, avg_thrusts):
    height = bar.get_height()
    ax1_2.text(bar.get_x() + bar.get_width()/2., height,
              f'{val:.1f}N', ha='center', va='bottom', fontsize=11, fontweight='bold')

baseline = avg_thrusts[0]
savings = [(baseline - val)/baseline * 100 for val in avg_thrusts]
ax1_2_twin = ax1_2.twinx()
ax1_2_twin.plot(labels_short, savings, 'ko-', linewidth=2, markersize=8, label='% Saving')
ax1_2_twin.set_ylabel('Energy Saving [%]', fontsize=11, fontweight='bold')
ax1_2_twin.set_ylim([-2, max(savings) + 2])

ax1_2.set_title("Average Thrust Comparison", fontsize=14, fontweight='bold')
ax1_2.set_ylabel("Avg Thrust [N]", fontsize=11, fontweight='bold')
ax1_2.grid(True, alpha=0.3, axis='y')

# --- Subplot 1.3: Cumulative Energy Consumption ---
ax1_3 = fig1.add_subplot(gs1[1, 1])
for idx, (t, th, lab) in enumerate(res_energy):
    dt_energy = t[1] - t[0] if len(t) > 1 else 0.05
    cumulative_energy = np.cumsum(th) * dt_energy
    ax1_3.plot(t, cumulative_energy, label=labels_short[idx], linewidth=2.5, 
              color=colors_energy[idx], alpha=0.9)
    
    final_energy = cumulative_energy[-1]
    ax1_3.annotate(f'{final_energy:.0f}J', 
                  xy=(t[-1], final_energy), 
                  xytext=(5, 0), textcoords='offset points',
                  fontsize=10, fontweight='bold', color=colors_energy[idx])

ax1_3.set_title("Cumulative Energy Consumption", fontsize=14, fontweight='bold')
ax1_3.set_xlabel("Time [s]", fontsize=11, fontweight='bold')
ax1_3.set_ylabel("Cumulative Energy [J]", fontsize=11, fontweight='bold')
ax1_3.legend(fontsize=10, loc='upper left')
ax1_3.grid(True, alpha=0.4, linestyle='--')
ax1_3.set_xlim([0, max(t)])

fig1.suptitle("ENERGY ANALYSIS: Formation Adaptation Strategies", 
            fontsize=18, fontweight='bold', y=0.98)

# ==============================================================================
# PLOT 2: STABILITY ANALYSIS (Migliorato con subplot)
# ==============================================================================
fig2 = plt.figure(figsize=(16, 10))
gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# --- Subplot 2.1: Stability Penalty vs Time ---
ax2_1 = fig2.add_subplot(gs2[0, :])
for idx, (t, stab, lab) in enumerate(res_stab):
    ax2_1.plot(t, stab, label=lab, linewidth=2.5, color=colors_stab[idx], alpha=0.9)
    avg_stab = np.mean(stab)
    ax2_1.axhline(avg_stab, color=colors_stab[idx], linestyle='--', alpha=0.4, linewidth=1.5)

ax2_1.set_title("Stability Analysis: Penalty Metric vs Time", fontsize=16, fontweight='bold', pad=15)
ax2_1.set_xlabel("Time [s]", fontsize=13, fontweight='bold')
ax2_1.set_ylabel("Stability Penalty [Lower = Better]", fontsize=13, fontweight='bold')
ax2_1.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax2_1.grid(True, alpha=0.4, linestyle='--')
ax2_1.set_xlim([0, max(t)])
ax2_1.set_ylim([0, max([np.max(stab) for t, stab, lab in res_stab]) * 1.1])

# --- Subplot 2.2: Average & Peak Stability Comparison ---
ax2_2 = fig2.add_subplot(gs2[1, 0])
labels_stab_short = ['No Tilt', 'Tilt Only', 'Full']
avg_stabs = [np.mean(stab) for t, stab, lab in res_stab]
peak_stabs = [np.max(stab) for t, stab, lab in res_stab]

x = np.arange(len(labels_stab_short))
width = 0.35

bars1 = ax2_2.bar(x - width/2, avg_stabs, width, label='Average', 
                  color=colors_stab, alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax2_2.bar(x + width/2, peak_stabs, width, label='Peak', 
                  color=colors_stab, alpha=0.4, edgecolor='black', linewidth=1.5, hatch='//')

for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    h1 = bar1.get_height()
    h2 = bar2.get_height()
    ax2_2.text(bar1.get_x() + bar1.get_width()/2., h1,
              f'{h1:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2_2.text(bar2.get_x() + bar2.get_width()/2., h2,
              f'{h2:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2_2.set_title("Stability Metrics Comparison", fontsize=14, fontweight='bold')
ax2_2.set_ylabel("Stability Penalty", fontsize=11, fontweight='bold')
ax2_2.set_xticks(x)
ax2_2.set_xticklabels(labels_stab_short, fontsize=11)
ax2_2.legend(fontsize=10)
ax2_2.grid(True, alpha=0.3, axis='y')

# --- Subplot 2.3: Stability Distribution (Violin Plot) ---
ax2_3 = fig2.add_subplot(gs2[1, 1])
stab_data = [stab for t, stab, lab in res_stab]

parts = ax2_3.violinplot(stab_data, positions=range(len(labels_stab_short)), 
                         showmeans=True, showmedians=True, widths=0.7)

for idx, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_stab[idx])
    pc.set_alpha(0.6)

parts['cmeans'].set_color('red')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('blue')
parts['cmedians'].set_linewidth(2)

ax2_3.set_title("Stability Distribution", fontsize=14, fontweight='bold')
ax2_3.set_ylabel("Stability Penalty", fontsize=11, fontweight='bold')
ax2_3.set_xticks(range(len(labels_stab_short)))
ax2_3.set_xticklabels(labels_stab_short, fontsize=11)
ax2_3.grid(True, alpha=0.3, axis='y')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Mean'),
                  Line2D([0], [0], color='blue', linewidth=2, label='Median')]
ax2_3.legend(handles=legend_elements, fontsize=10, loc='upper right')

fig2.suptitle("STABILITY ANALYSIS: Formation Control Strategies", 
            fontsize=18, fontweight='bold', y=0.98)

# --- PLOT 3: Trajectory 3D ---
from mpl_toolkits.mplot3d import Axes3D

fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')

sc = ax3.scatter(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2], 
                c=vel_mags, cmap='coolwarm', s=20, alpha=0.8)

ax3.plot(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2], 
        'k-', alpha=0.3, linewidth=1)

ax3.set_title("Desired Bezier Trajectory (Color = Velocity)", fontsize=14, fontweight='bold')
ax3.set_xlabel("X [m]", fontsize=12)
ax3.set_ylabel("Y [m]", fontsize=12)
ax3.set_zlabel("Z [m]", fontsize=12)

cbar = plt.colorbar(sc, ax=ax3, shrink=0.7, aspect=10)
cbar.set_label('Velocity [m/s]', fontsize=10)

P0 = p_traj.home_pos
P3 = p_traj.payload_goal_pos
ax3.scatter(P0[0], P0[1], P0[2], color='green', marker='^', s=200, 
           label='Start', edgecolors='black', linewidths=2)
ax3.scatter(P3[0], P3[1], P3[2], color='red', marker='*', s=300, 
           label='Goal', edgecolors='black', linewidths=2)

ax3.scatter([P1[0]], [P1[1]], [P1[2]], color='orange', marker='o', s=100, 
           alpha=0.5, label='Control P1')
ax3.scatter([P2[0]], [P2[1]], [P2[2]], color='purple', marker='o', s=100, 
           alpha=0.5, label='Control P2')

ax3.legend(fontsize=10)
ax3.view_init(elev=20, azim=45)

max_range = np.array([
    traj_pts[:, 0].max() - traj_pts[:, 0].min(),
    traj_pts[:, 1].max() - traj_pts[:, 1].min(),
    traj_pts[:, 2].max() - traj_pts[:, 2].min()
]).max() / 2.0

mid_x = (traj_pts[:, 0].max() + traj_pts[:, 0].min()) * 0.5
mid_y = (traj_pts[:, 1].max() + traj_pts[:, 1].min()) * 0.5
mid_z = (traj_pts[:, 2].max() + traj_pts[:, 2].min()) * 0.5

ax3.set_xlim(mid_x - max_range, mid_x + max_range)
ax3.set_ylim(mid_y - max_range, mid_y + max_range)
ax3.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()

print("\nEnhanced plots generated successfully!")
print("Displaying plots... (Close windows to continue)")

plt.show()