import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

# Aggiungi il percorso radice per importare i moduli del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import formation as formation
from parameters import SysParams
import physics

# =============================================================================
# CONTEXT MOCK & FUNZIONI DI SUPPORTO
# =============================================================================
class MockCtx:
    """Contesto per l'ottimizzatore. Necessario per far funzionare la logica Yaw."""
    def __init__(self, p):
        self.last_theta_opt = np.radians(getattr(p, 'theta_ref', 30.0))
        self.last_stable_yaw = 0.0
        self.F_xy_filt = np.zeros(2)
        self.e_filt_despun = np.zeros(3)
        self.last_yaw_target_raw = 0.0
        self.last_yaw_abs_tgt = 0.0
        self.lambda_traj_effective = 0.0  # Sarà sovrascritto nel loop

def get_v_for_force(p, force_target_N):
    if force_target_N < 1e-3: return 0.0
    pay_proj_area = np.pi * (0.6**2) 
    k_drag = 0.5 * p.rho * p.Cd_pay * pay_proj_area
    return np.sqrt(force_target_N / k_drag)

def get_wind_force_and_area(p, v_wind_world, attitude, wind_dir_override=None):
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    
    # Se il Force Direction è 0, usa la direzione di override per calcolare l'area geometrica corretta
    if wind_mag >= 1e-3:
        wind_dir = v_wind_world / wind_mag
    else:
        wind_dir = wind_dir_override if wind_dir_override is not None else np.array([1.0, 0.0, 0.0])
        
    R_pay = physics.get_rotation_matrix(phi, theta, psi)
    wind_in_body = R_pay.T @ wind_dir

    if hasattr(p, 'payload_shape') and p.payload_shape in ['box', 'rect', 'square']:
        A_x, A_y, A_z = p.pay_w * p.pay_h, p.pay_l * p.pay_h, p.pay_l * p.pay_w
        pay_proj_area = A_x * abs(wind_in_body[0]) + A_y * abs(wind_in_body[1]) + A_z * abs(wind_in_body[2])
    else:
        if getattr(p, 'payload_shape', '') == 'sphere':
            pay_proj_area = np.pi * (p.R_disk**2)
        else: # Cylinder
            A_side = 2.0 * p.R_disk * p.pay_h
            A_top = np.pi * p.R_disk**2
            sin_tilt = np.sqrt(wind_in_body[0]**2 + wind_in_body[1]**2)
            pay_proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])

    if wind_mag < 1e-3: 
        return np.zeros(3), pay_proj_area

    f_wind = 0.5 * p.rho * p.Cd_pay * pay_proj_area * (wind_mag**2) * wind_dir
    return f_wind, pay_proj_area

def apply_shape(p, s_params):
    p.payload_shape = s_params['shape']
    if p.payload_shape == 'sphere':
        p.R_disk = s_params['R_disk']
        p.pay_h = s_params.get('pay_h', 0.3)
        I_val = (2.0/5.0) * (p.m_payload + p.m_liquid) * (p.R_disk**2)
        p.J = np.diag([I_val, I_val, I_val])
    elif p.payload_shape == 'cylinder':
        p.R_disk = s_params['R_disk']
        p.pay_h = s_params['pay_h']
        m_tot = p.m_payload + p.m_liquid
        Ixx = (1/12) * m_tot * (3*p.R_disk**2 + p.pay_h**2)
        Iyy = Ixx
        Izz = 0.5 * m_tot * p.R_disk**2
        p.J = np.diag([Ixx, Iyy, Izz])
    else: # Box/Rect
        p.pay_l = s_params['pay_l']
        p.pay_w = s_params['pay_w']
        p.pay_h = s_params['pay_h']
        m_tot = p.m_payload + p.m_liquid
        Ixx = (1/12) * m_tot * (p.pay_w**2 + p.pay_h**2)
        Iyy = (1/12) * m_tot * (p.pay_l**2 + p.pay_h**2)
        Izz = (1/12) * m_tot * (p.pay_l**2 + p.pay_w**2)
        p.J = np.diag([Ixx, Iyy, Izz])
    p.invJ = np.linalg.inv(p.J)

# =============================================================================
# MAIN SCRIPT
# =============================================================================
def plot_yaw_weathervane_comparison():
    print("Avvio simulazione statica per Weather-Vane (lambda_traj = 0 vs 1)...")
    
    p_base = SysParams()
    p_base.N = 4
    p_base.m_payload = 2.0
    p_base.m_liquid = 1.0
    
    p_base.lambda_aero = 0.0 
    p_base.lambda_shape = 0.0  
    p_base.w_smooth = 0.0               
    p_base.w_barrier = 0.0              
    
    wind_forces_N = np.linspace(0.0, 35.0, 50)
    
    # Geometrie da testare
    shapes = {
        'ref_box':  {'shape': 'rect', 'pay_l': 1.2, 'pay_w': 0.6, 'pay_h': 0.4},
        'sphere':   {'shape': 'sphere', 'R_disk': 0.42},
        'cylinder': {'shape': 'cylinder', 'R_disk': 0.6, 'pay_h': 0.3},
        'flat_box': {'shape': 'rect', 'pay_l': 1.2, 'pay_w': 0.6, 'pay_h': 0.4}
    }
    
    # -------------------------------------------------------------------------
    # PARTE 1: Sweep della Forza del Force Direction (45° costanti, lambda_traj = 0 vs 1)
    # -------------------------------------------------------------------------
    results = {sh: {0.0: {'wind_force': [], 'spread': [], 'power': [], 'max_thrust': [], 'area': [], 'imbalance': []},
                    1.0: {'wind_force': [], 'spread': [], 'power': [], 'max_thrust': [], 'area': [], 'imbalance': []}}
               for sh in shapes}
    
    for sh_name, sh_params in shapes.items():
        print(f"Calcolo sweep di forza per forma: {sh_name}...")
        for l_traj in [0.0, 1.0]:
            ctx = MockCtx(p_base)
            ctx.lambda_traj_effective = l_traj
            
            p = copy.deepcopy(p_base)
            apply_shape(p, sh_params)
            p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
            
            for f_target in wind_forces_N:
                v_mag = get_v_for_force(p_base, f_target)
                wind_dir_45 = np.array([0.7071, 0.7071, 0.0]) # Vento fisso a 45°
                v_wind_vec = v_mag * wind_dir_45 
                
                # Assicuriamoci che anche p sia aggiornato, nel caso la tua logica lo legga
                p.wind_vel = v_wind_vec 
                
                # Mini-loop di assestamento per il Weather-Vane dinamico
                for _ in range(25):
                    current_yaw = ctx.last_stable_yaw
                    attitude = (0.0, 0.0, current_yaw)
                    F_wind_actual, proj_area = get_wind_force_and_area(p, v_wind_vec, attitude, wind_dir_override=wind_dir_45)
                    
                    # --- NOVITÀ: Calcolo dinamico dell'attivazione basato sulle TUE soglie ---
                    if l_traj > 0.0:
                        F_mag = np.linalg.norm(F_wind_actual[:2])
                        f_min = 0.0  # Usa il valore che hai messo in SysParams
                        f_max = 0.0 # Usa il valore che hai messo in SysParams
                        
                        if f_max > f_min:
                            factor = np.clip((F_mag - f_min) / (f_max - f_min), 0.0, 1.0)
                        else:
                            factor = 1.0 if F_mag >= f_min else 0.0
                            
                        ctx.lambda_traj_effective = factor
                    else:
                        ctx.lambda_traj_effective = 0.0
                    # --------------------------------------------------------------------------
                    
                    state = {
                        'pay_pos': np.zeros(3), 
                        'pay_att': np.array(attitude), 
                        'pay_vel': np.zeros(3),
                        'pay_omega': np.zeros(3), 
                        'uav_vel': np.zeros((3, p.N))
                    }
                    
                    _, _, _, ff_forces = formation.compute_optimal_formation(
                        p, state, np.zeros(3), np.zeros(3), 0.0, 
                        force_attitude=(0.0, 0.0), 
                        F_ext_total=F_wind_actual,
                        com_offset_body=np.zeros(3),
                        ctx=ctx 
                    )
                
                tensions = np.linalg.norm(ff_forces, axis=0)
                spread = np.max(tensions) - np.min(tensions)
                imbalance = float(np.std(tensions))
                
                F_gravity_drones = np.array([0.0, 0.0, -p.m_drone * p.g])
                if v_mag > 1e-3:
                    wind_dir = v_wind_vec / v_mag
                    f_drag_single = 0.5 * p.rho * p.Cd_uav * p.A_uav * (v_mag**2)
                    F_wind_drone = f_drag_single * wind_dir
                else:
                    F_wind_drone = np.zeros(3)
                    
                thrust_mags = [np.linalg.norm(ff_forces[:, i] - F_gravity_drones - F_wind_drone) for i in range(p.N)]
                power = np.sum(np.array(thrust_mags)**1.5)
                max_thrust = np.max(thrust_mags)
                
                results[sh_name][l_traj]['wind_force'].append(f_target)
                results[sh_name][l_traj]['spread'].append(spread)
                results[sh_name][l_traj]['power'].append(power)
                results[sh_name][l_traj]['max_thrust'].append(max_thrust)
                results[sh_name][l_traj]['area'].append(proj_area)
                results[sh_name][l_traj]['imbalance'].append(imbalance)
                
    # -------------------------------------------------------------------------
    # PARTE 2: Sweep Direzionale (Velocità del Force Direction fissa a 10 m/s, SOLO STATICO)
    # -------------------------------------------------------------------------
    wind_speed_sweep = 10.0
    sweep_angles = np.linspace(-180, 180, 73)
    
    sweep_results = {sh: {'power': [], 'imbalance': []} for sh in ['sphere', 'cylinder', 'flat_box']}
                     
    print("\nCalcolo sweep direzionale (Velocità fissa = 10 m/s, solo statico) per Sphere, Cylinder, Flat Box...")
    for sh_name in ['sphere', 'cylinder', 'flat_box']:
        sh_params = shapes[sh_name]
        
        ctx = MockCtx(p_base)
        ctx.lambda_traj_effective = 0.0 # Forza calcolo statico
        
        p = copy.deepcopy(p_base)
        apply_shape(p, sh_params)
        p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
        
        for deg in sweep_angles:
            rad = np.radians(deg)
            # Force Direction ruota di 360°, payload rimane fisso (yaw = 0)
            v_wind_vec = wind_speed_sweep * np.array([np.cos(rad), np.sin(rad), 0.0])
            
            # Condizione statica: calcolo immediato (nessun ciclo necessario)
            attitude = (0.0, 0.0, 0.0) 
            F_wind_actual, _ = get_wind_force_and_area(p, v_wind_vec, attitude)
            
            state = {
                'pay_pos': np.zeros(3), 
                'pay_att': np.array(attitude), 
                'pay_vel': np.zeros(3),
                'pay_omega': np.zeros(3), 
                'uav_vel': np.zeros((3, p.N))
            }
            
            _, _, _, ff_forces = formation.compute_optimal_formation(
                p, state, np.zeros(3), np.zeros(3), 0.0, 
                force_attitude=(0.0, 0.0), 
                F_ext_total=F_wind_actual,
                com_offset_body=np.zeros(3),
                ctx=ctx 
            )
            
            tensions = np.linalg.norm(ff_forces, axis=0)
            imbalance = float(np.std(tensions))
            
            F_gravity_drones = np.array([0.0, 0.0, -p.m_drone * p.g])
            f_drag_single = 0.5 * p.rho * p.Cd_uav * p.A_uav * (wind_speed_sweep**2)
            wind_dir = v_wind_vec / wind_speed_sweep if wind_speed_sweep > 1e-3 else np.array([1, 0, 0])
            F_wind_drone = f_drag_single * wind_dir
            
            thrust_mags = [np.linalg.norm(ff_forces[:, i] - F_gravity_drones - F_wind_drone) for i in range(p.N)]
            power = np.sum(np.array(thrust_mags)**1.5)
            
            sweep_results[sh_name]['power'].append(power)
            sweep_results[sh_name]['imbalance'].append(imbalance)

    print("Dati calcolati con successo. Generazione grafici...")

    # =========================================================================
    # PLOTTING
    # =========================================================================
    C_0 = '#1D7EC2' # Blu per Statico
    C_1 = '#E63946' # Rosso per Adaptive
    C_2 = '#2A9D8F' # Verde per Flat Box
    C_3 = '#F4A261' # Arancione per Cylinder

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    #fig.suptitle('Analisi Weather-Vane: $\lambda_{traj} = 0$ vs $\lambda_{traj} = 1$', 
    #             fontweight='bold', fontsize=16)
    
    x_val = results['ref_box'][0.0]['wind_force']

    # --- RIGA 1: REFERENCE BOX (1.2 x 0.6 x 0.3) - Sweep di Forza ---
    ax = axs[0, 0]
    ax.plot(x_val, results['ref_box'][0.0]['spread'], label=r'$\lambda_{traj} = 0$', color=C_0, lw=2.5)
    ax.plot(x_val, results['ref_box'][1.0]['spread'], label=r'$\lambda_{traj} = 1$', color=C_1, lw=2.5)
    ax.set_title('Cable Tension Spread (Ref. Box - Force Direction 45°)', fontweight='bold')
    ax.set_ylabel('$T_{max} - T_{min}$ [N]')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(x_val, results['ref_box'][0.0]['power'], label=r'$\lambda_{traj} = 0$', color=C_0, lw=2.5)
    ax.plot(x_val, results['ref_box'][1.0]['power'], label=r'$\lambda_{traj} = 1$', color=C_1, lw=2.5)
    ax.set_title('Average System Power (Ref. Box - Force Direction 45°)', fontweight='bold')
    ax.set_ylabel('Power ($\sum T^{1.5}$)')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    ax = axs[0, 2]
    ax.plot(x_val, results['ref_box'][0.0]['max_thrust'], label=r'$\lambda_{traj} = 0$', color=C_0, lw=2.5)
    ax.plot(x_val, results['ref_box'][1.0]['max_thrust'], label=r'$\lambda_{traj} = 1$', color=C_1, lw=2.5)
    ax.set_title('Max Single Drone Thrust (Ref. Box - Force Direction 45°)', fontweight='bold')
    ax.set_ylabel('Max Thrust [N]')
    ax.set_xlabel('Horizontal Force [N]')
    ax.axhline(p_base.F_max_thrust, color='gray', linestyle=':', lw=2, label='Hardware Limit')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # --- RIGA 2: CONFRONTO SHAPES ---
    
    # 2.1 Projected Area (Rispetto alla Forza del Force Direction a 45°)
    ax = axs[1, 0]
    ax.plot(x_val, results['sphere'][0.0]['area'], color=C_0, lw=2, linestyle='--', alpha=0.5, label=r'Sphere ($\lambda_{traj}=0$)')
    ax.plot(x_val, results['cylinder'][0.0]['area'], color=C_3, lw=2, linestyle='--', alpha=0.5, label=r'Cylinder ($\lambda_{traj}=0$)')
    ax.plot(x_val, results['flat_box'][0.0]['area'], color=C_2, lw=2, linestyle='--', alpha=0.5, label=r'Flat Box ($\lambda_{traj}=0$)')
    
    ax.plot(x_val, results['sphere'][1.0]['area'], color=C_0, lw=2.5, label=r'Sphere ($\lambda_{traj}=1$)')
    ax.plot(x_val, results['cylinder'][1.0]['area'], color=C_3, lw=2.5, label=r'Cylinder ($\lambda_{traj}=1$)')
    ax.plot(x_val, results['flat_box'][1.0]['area'], color=C_2, lw=2.5, label=r'Flat Box ($\lambda_{traj}=1$)')
    ax.set_title('Exposed Projected Area (Force Direction 45°)', fontweight='bold')
    ax.set_ylabel('Projected Area [m²]')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)

    # 2.2 Energetic Efficiency vs Wind Direction (SOLO STATICO)
    ax = axs[1, 1]
    ax.plot(sweep_angles, sweep_results['sphere']['power'], color=C_0, lw=2.5, label='Sphere')
    ax.plot(sweep_angles, sweep_results['cylinder']['power'], color=C_3, lw=2.5, label='Cylinder')
    ax.plot(sweep_angles, sweep_results['flat_box']['power'], color=C_2, lw=2.5, label='Flat Box')
    ax.set_title('Energetic Efficiency vs Force Dir. (Static)', fontweight='bold')
    ax.set_ylabel('Power ($\sum T^{1.5}$)')
    ax.set_xlabel('Horizontal Force Direction [deg]')
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    # 2.3 Formation Load Imbalance vs Wind Direction (SOLO STATICO)
    ax = axs[1, 2]
    ax.plot(sweep_angles, sweep_results['sphere']['imbalance'], color=C_0, lw=2.5, label='Sphere')
    ax.plot(sweep_angles, sweep_results['cylinder']['imbalance'], color=C_3, lw=2.5, label='Cylinder')
    ax.plot(sweep_angles, sweep_results['flat_box']['imbalance'], color=C_2, lw=2.5, label='Flat Box')
    ax.set_title('Formation Load Imbalance vs Force Dir. (Static)', fontweight='bold')
    ax.set_ylabel('Tension Std. Dev. [N]')
    ax.set_xlabel('Horizontal Force Direction [deg]')
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_yaw_weathervane_comparison()