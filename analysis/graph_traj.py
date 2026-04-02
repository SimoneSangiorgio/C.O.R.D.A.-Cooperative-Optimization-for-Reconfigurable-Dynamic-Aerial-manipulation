import sys
import os

# Setup path per importare i moduli del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Import moduli esistenti
from parameters import SysParams
import mission
from mission import MissionContext, equations_of_motion
import formation as formation
import physics

# ==============================================================================
# 1. TRAJECTORY GENERATORS (RALLENTATI PER STABILITÀ)
# ==============================================================================
class TrajectoryGenerator:
    @staticmethod
    def circle(t, radius=20.0, period=80.0, height=6.0): # Periodo aumentato a 80s
        if t < 0: return TrajectoryGenerator.circle(0, radius, period, height)
        
        omega = 2 * np.pi / period
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = height
        
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        vz = 0
        
        return np.array([x, y, z]), np.array([vx, vy, vz]), np.arctan2(vy, vx)

    @staticmethod
    def figure_eight(t, scale=20.0, period=100.0, height=6.0): # Periodo aumentato a 100s
        if t < 0: return TrajectoryGenerator.figure_eight(0, scale, period, height)

        t_scaled = 2 * np.pi * t / period
        denom = 1 + np.sin(t_scaled)**2
        x = scale * np.cos(t_scaled) / denom
        y = scale * np.cos(t_scaled) * np.sin(t_scaled) / denom
        z = height
        
        dt = 0.01
        t_next = t_scaled + 2 * np.pi * dt / period
        denom_n = 1 + np.sin(t_next)**2
        xn = scale * np.cos(t_next) / denom_n
        yn = scale * np.cos(t_next) * np.sin(t_next) / denom_n
        
        vx = (xn - x) / dt
        vy = (yn - y) / dt
        vz = 0
        
        return np.array([x, y, z]), np.array([vx, vy, vz]), np.arctan2(vy, vx)

# ==============================================================================
# 2. SIMULATION RUNNER (CON PID SOFT TUNING)
# ==============================================================================
def run_fast_simulation(param_overrides, config_tuple, traj_func, duration=50.0):
    p = SysParams()
    
    # --- TUNING STABILITÀ PER I GRAFICI ---
    # Sovrascriviamo i PID per renderli più "morbidi" ed evitare vibrazioni grafiche
    p.kp_pos = 6.0      # Meno aggressivo (era 15)
    p.kv_pos = 8.0      # Più smorzamento (era 10)
    p.kp_roll = 8.0     
    p.kv_rot = 10.0     # Alto smorzamento angolare
    p.k_cable = 150.0   # Cavo più elastico per assorbire shock (era 500)
    p.gamma_damp = 10.0 # Smorzamento cavo alto
    
    # Override Parametri Utente
    for key, val in param_overrides.items():
        if hasattr(p, key):
            setattr(p, key, val)
    
    # Ricalcolo Inerzia
    if p.payload_shape == 'sphere':
        r = p.R_disk
        I = (2.0/5.0) * p.m_payload * (r**2)
        p.J = np.diag([I, I, I])
    else:
        Ixx = (1/12) * p.m_payload * (p.pay_w**2 + p.pay_h**2)
        Iyy = (1/12) * p.m_payload * (p.pay_l**2 + p.pay_h**2)
        Izz = (1/12) * p.m_payload * (p.pay_l**2 + p.pay_w**2)
        p.J = np.diag([Ixx, Iyy, Izz])

    # Geometria
    p.uav_offsets, p.attach_vecs, geo_r = formation.compute_geometry(p)
    dist_h = geo_r - (0.1 if p.payload_shape == 'sphere' else min(p.pay_l, p.pay_w)/2)
    p.safe_altitude_offset = np.sqrt(p.L**2 - dist_h**2) if p.L > dist_h else p.L

    # Configurazione Test
    p.lambda_shape = 0.0
    p.lambda_aero = 0.0
    p.lambda_traj = 0.0
    p.k_tilt = 1.0 
    
    if len(config_tuple) == 3:
        if 'test_type' in param_overrides and param_overrides['test_type'] == 'efficiency':
             p.lambda_shape, p.lambda_aero, p.lambda_traj = config_tuple
        elif 'test_type' in param_overrides and param_overrides['test_type'] == 'stability':
             p.k_tilt, p.lambda_shape, p.lambda_traj = config_tuple

    # Inizializzazione Contesto
    ctx = MissionContext(p)
    ctx.phase = 4 
    ctx.start_phase_time = 0.0
    
    # --- WARM-UP ESTESO ---
    warmup_duration = 5.0  # 5 secondi di assestamento
    
    pos_0, vel_traj_0, yaw_0 = traj_func(0.0)
    pos_0[2] = p.safe_altitude_offset
    ctx.nav_start_pos = pos_0.copy()
    
    uav_pos_0 = np.zeros((3, p.N))
    for i in range(p.N):
        uav_pos_0[:, i] = pos_0 + p.uav_offsets[:, i]
        
    x = np.concatenate([
        uav_pos_0.flatten('F'), pos_0, np.zeros(3), 
        np.zeros(3*p.N), np.zeros(3), np.zeros(3),
        np.zeros(3*p.N), np.zeros(3)
    ])
    
    # Timing
    graphic_dt = 0.05      # 20 Hz
    physics_dt = 0.002     # 500 Hz
    substeps = int(graphic_dt / physics_dt)
    
    total_duration = duration + warmup_duration
    steps = int(total_duration / graphic_dt)
    
    # Log Data
    total_energy_proxy = 0.0
    energy_hist = []  
    attitude_error_accum = 0.0
    
    traj_x, traj_y, traj_v = [], [], []
    attitude_history, t_hist = [], []

    current_sim_time = -warmup_duration 
    p.stop_after_phase = 6
    
    s_temp = physics.unpack_state(x, p)
    mission._init_hover_config(p, ctx, s_temp)

    try:
        for k in range(steps):
            # LOGICA WARMUP vs RUN
            if current_sim_time < 0:
                # Hover statico sul punto di start
                ref_pos, _, ref_yaw = traj_func(0.0)
                ref_vel = np.zeros(3)
                
                # Rampa di velocità nell'ultimo secondo
                if current_sim_time > -2.0:
                     _, v_final, _ = traj_func(0.0)
                     blend = (2.0 + current_sim_time) / 2.0
                     ref_vel = v_final * blend
            else:
                ref_pos, ref_vel, ref_yaw = traj_func(current_sim_time)
            
            custom_ref = {
                'pos': ref_pos, 'vel': ref_vel, 'acc': np.zeros(3), 
                'yaw': ref_yaw, 'z_rel': p.safe_altitude_offset
            }
            
            mission.update_dynamic_wind(max(0, current_sim_time), p)
            
            if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e4):
                break 

            s = physics.unpack_state(x, p)
            
            mission._run_optimization(current_sim_time, ctx, p, s, custom_ref)
            ctx.current_ref = custom_ref
            
            # Fisica Sub-Stepping
            for _ in range(substeps):
                dx = equations_of_motion(current_sim_time, x, p, ctx)
                x = x + dx * physics_dt
            
            current_sim_time += graphic_dt

            # Raccolta Dati (Solo se tempo positivo)
            if current_sim_time >= 0:
                thrust_sum_instant = 0.0
                if hasattr(ctx, 'last_u_acc'):
                    for i in range(p.N):
                        acc_cmd = ctx.last_u_acc[:, i]
                        thrust = np.linalg.norm(p.m_drone * (acc_cmd + np.array([0,0,9.81])))
                        thrust_sum_instant += thrust
                
                total_energy_proxy += thrust_sum_instant * graphic_dt
                energy_hist.append(thrust_sum_instant)
                
                # Misuriamo l'oscillazione rispetto all'assetto ideale (0 per Rigid, o Aero per altri)
                # Qui usiamo la velocità angolare come proxy di instabilità
                attitude_error_accum += np.linalg.norm(s['pay_omega']) * graphic_dt
                
                attitude_history.append(np.degrees(s['pay_att'][:2])) 
                t_hist.append(current_sim_time)
                traj_x.append(s['pay_pos'][0])
                traj_y.append(s['pay_pos'][1])
                traj_v.append(np.linalg.norm(s['pay_vel']))

    except Exception:
        pass

    return {
        'energy': total_energy_proxy,
        'energy_hist': np.array(energy_hist),
        'stability_score': attitude_error_accum,
        'traj_x': np.array(traj_x),
        'traj_y': np.array(traj_y),
        'traj_v': np.array(traj_v),
        'att_hist': np.array(attitude_history),
        'time': np.array(t_hist)
    }

# ==============================================================================
# 3. PLOTTING FUNCTIONS
# ==============================================================================

def plot_graph_1_efficiency():
    print("Generating Graph 1: Energy Efficiency...")
    
    configs = [
        (0,0,0, "Base (0,0,0)"),
        (0,0,1, "Traj Only (0,0,1)"),
        (0,1,1, "Aero+Traj (0,1,1)"),
        (1,0,1, "Shape+Traj (1,0,1)"),
        (1,1,1, "Full (1,1,1)")
    ]
    
    overrides = {
        'payload_shape': 'sphere',
        'theta_ref': 30.0,
        'L': 3.0,
        'm_payload': 3.0,
        'test_type': 'efficiency'
    }
    
    energies = []
    labels = []
    energy_hists = []
    time_hists = []
    
    for c in configs:
        conf_vals = c[:3]
        label = c[3]
        print(f"  Running: {label}")
        res = run_fast_simulation(overrides, conf_vals, TrajectoryGenerator.circle, duration=60.0)
        
        if len(res['energy_hist']) > 0:
            energies.append(res['energy'])
            energy_hists.append(res['energy_hist'])
            time_hists.append(res['time'])
            labels.append(label)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if len(energies) > 0 and energies[0] > 0:
        base_e = energies[0]
        energies_norm = [e/base_e * 100.0 for e in energies]
    else:
        energies_norm = []
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(labels)))
    bars = ax1.bar(labels, energies_norm, color=colors)
    
    ax1.set_title('Normalized Energy Consumption (Circular Trajectory)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy Cost [%]', fontsize=10)
    ax1.axhline(100, color='gray', linestyle='--')
    
    if energies_norm:
        min_y = min(energies_norm) * 0.95
        max_y = max(energies_norm) * 1.05
        ax1.set_ylim(min_y, max_y)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    for i, label in enumerate(labels):
        y_data = energy_hists[i]
        x_data = time_hists[i]
        
        # Strong smoothing for clean plots
        window = 30 
        if len(y_data) > window:
            y_smooth = np.convolve(y_data, np.ones(window)/window, mode='valid')
            x_smooth = x_data[:len(y_smooth)]
            ax2.plot(x_smooth, y_smooth, label=label, color=colors[i], linewidth=1.5)
        else:
             ax2.plot(x_data, y_data, label=label, color=colors[i], linewidth=1.5)
            
    ax2.set_title('Total Thrust over Time (Smoothed)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Thrust [N]', fontsize=10)
    ax2.set_xlabel('Time [s]', fontsize=10)
    ax2.legend()
    
    plt.suptitle('Graph 1: Energy Efficiency Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_graph_2_stability():
    print("Generating Graph 2: Stability...")
    
    configs = [
        (0.0, 0, 0, "Rigid (0,0,0)"),
        (1.0, 0, 0, "Tilt Only (1,0,0)"),
        (1.0, 0, 1, "Tilt+Traj (1,0,1)"),
        (1.0, 1, 1, "Full (1,1,1)")
    ]
    
    overrides = {
        'payload_shape': 'box',
        'pay_l': 0.8, 'pay_w': 0.8, 'pay_h': 0.4,
        'theta_ref': 25.0,
        'L': 4.0,
        'test_type': 'stability'
    }
    
    stability_scores = []
    labels = []
    time_hists = []
    roll_hists = []
    
    colors = ['#95a5a6', '#34495e', '#3498db', '#9b59b6']
    
    for c in configs:
        conf_vals = c[:3]
        label = c[3]
        print(f"  Running: {label}")
        res = run_fast_simulation(overrides, conf_vals, TrajectoryGenerator.figure_eight, duration=80.0)
        
        # Gestione Fallimenti (Rigid fallirà comunque, ma ora lo plottiamo)
        score = res['stability_score'] if len(res['time']) > 0 else 0
        stability_scores.append(score)
        labels.append(label)
        time_hists.append(res['time'])
        
        if len(res['att_hist']) > 0:
            roll_hists.append(res['att_hist'][:, 0]) 
        else:
            roll_hists.append(np.array([]))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Subplot 1
    ax1.bar(labels, stability_scores, color=colors)
    ax1.set_title('Stability Score (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Oscillation', fontsize=10)
    
    # Subplot 2
    for i, label in enumerate(labels):
        if len(time_hists[i]) > 0:
            # Smoothing del grafico di stabilità
            y = roll_hists[i]
            x = time_hists[i]
            if len(y) > 20:
                y = np.convolve(y, np.ones(10)/10, mode='valid')
                x = x[:len(y)]
            ax2.plot(x, y, label=label, color=colors[i], linewidth=1.5)
        else:
            ax2.plot([], [], label=f"{label} (Unstable)", color=colors[i], linestyle='--')
    
    ax2.set_title('Payload Roll Oscillation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Roll Angle [deg]', fontsize=10)
    ax2.set_xlabel('Time [s]', fontsize=10)
    ax2.legend()
    
    plt.suptitle('Graph 2: Stability Analysis (Figure 8)', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_graph_3_trajectory():
    print("Generating Graph 3: Trajectory...")
    
    overrides = {
        'payload_shape': 'box',
        'theta_ref': 30.0,
        'L': 3.0,
        'test_type': 'efficiency' 
    }
    config = (1.0, 1.0, 1.0) 
    
    res_8 = run_fast_simulation(overrides, config, TrajectoryGenerator.figure_eight, duration=80.0)
    res_c = run_fast_simulation(overrides, config, TrajectoryGenerator.circle, duration=60.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    def plot_colored_line(ax, x, y, v, title):
        if len(x) < 2: return None
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=np.min(v), vmax=np.max(v))
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(v)
        lc.set_linewidth(3)
        
        line = ax.add_collection(lc)
        ax.set_xlim(np.min(x)-5, np.max(x)+5)
        ax.set_ylim(np.min(y)-5, np.max(y)+5)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        return line

    cbar1 = plot_colored_line(ax1, res_8['traj_x'], res_8['traj_y'], res_8['traj_v'], "Figure-8 Trajectory")
    if cbar1: fig.colorbar(cbar1, ax=ax1, label='Velocity [m/s]')
    
    cbar2 = plot_colored_line(ax2, res_c['traj_x'], res_c['traj_y'], res_c['traj_v'], "Circular Trajectory")
    if cbar2: fig.colorbar(cbar2, ax=ax2, label='Velocity [m/s]')
    
    plt.suptitle('Graph 3: Trajectory & Velocity Profile', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_graph_1_efficiency()
    plot_graph_2_stability()
    plot_graph_3_trajectory()
    print("=== DONE ===")