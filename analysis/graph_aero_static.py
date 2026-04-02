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
# FUNZIONI DI SUPPORTO E FISICA
# =============================================================================
def get_v_for_force(p, force_target_N):
    if force_target_N < 1e-3: return 0.0
    pay_proj_area = np.pi * (0.6**2) 
    k_drag = 0.5 * p.rho * p.Cd_pay * pay_proj_area
    return np.sqrt(force_target_N / k_drag)

def get_wind_force_and_area(p, v_wind_world, attitude):
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    
    # Se il vento è zero, usiamo una direzione fittizia solo per poter
    # proiettare l'area geometrica, ma restituiremo forza 0.
    wind_dir = v_wind_world / wind_mag if wind_mag >= 1e-3 else np.array([1.0, 0.0, 0.0])

    R_pay = physics.get_rotation_matrix(phi, theta, psi)
    wind_in_body = R_pay.T @ wind_dir

    # Calcolo Area
    if hasattr(p, 'payload_shape') and p.payload_shape in ['box', 'rect', 'square']:
        A_x, A_y, A_z = p.pay_w * p.pay_h, p.pay_l * p.pay_h, p.pay_l * p.pay_w
        pay_proj_area = A_x * abs(wind_in_body[0]) + A_y * abs(wind_in_body[1]) + A_z * abs(wind_in_body[2])
    else:
        if getattr(p, 'payload_shape', '') == 'sphere':
            pay_proj_area = np.pi * (p.R_disk**2)
        else:
            A_side = 2.0 * p.R_disk * p.pay_h
            A_top = np.pi * p.R_disk**2
            sin_tilt = np.sqrt(wind_in_body[0]**2 + wind_in_body[1]**2)
            pay_proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])

    # Se non c'è vento, ritorniamo forza 0 ma manteniamo l'AREA REALE
    if wind_mag < 1e-3: 
        return np.zeros(3), pay_proj_area

    f_wind = 0.5 * p.rho * p.Cd_pay * pay_proj_area * (wind_mag**2) * wind_dir
    return f_wind, pay_proj_area

def compute_aerodynamic_equilibrium(p, v_wind_world):
    phi, theta, psi = 0.0, 0.0, 0.0
    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)

    for _ in range(15):
        f_wind, _ = get_wind_force_and_area(p, v_wind_world, (phi, theta, psi))
        F_tot = np.array([0.0, 0.0, -m_tot * p.g]) + f_wind
        F_req = -F_tot
        theta = np.arctan2(F_req[0], abs(F_req[2]))
        phi = np.arctan2(-F_req[1], np.sqrt(F_req[0]**2 + F_req[2]**2))
    return phi, theta

def apply_shape(p, s_params):
    p.payload_shape = s_params['shape']
    if p.payload_shape == 'sphere':
        p.R_disk = s_params['R_disk']
        p.pay_h = s_params.get('pay_h', 0.3) # Ignorato per inerzia
        I_val = (2.0/5.0) * (p.m_payload + p.m_liquid) * (p.R_disk**2)
        p.J = np.diag([I_val, I_val, I_val])
    else:
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
def plot_aero_lambda_comparison():
    print("Avvio simulazione statica fluida per il confronto lambda_aero...")
    
    p_base = SysParams()
    p_base.N = 4
    p_base.m_payload = 2.0
    p_base.m_liquid = 1.0
    
    wind_forces_N = np.linspace(0.0, 35.0, 50)
    
    shapes = {
        'sphere': {'shape': 'sphere', 'R_disk': 0.6}, # Rimosso pay_h ininfluente
        'flat_box': {'shape': 'rect', 'pay_l': 1.3, 'pay_w': 1.1, 'pay_h': 0.3},
        'tall_box': {'shape': 'rect', 'pay_l': 0.3, 'pay_w': 1.0, 'pay_h': 1.3}
    }
    
    results = {sh: {0.0: {'wind_force': [], 'spread': [], 'power': [], 'max_thrust': [], 'area': [], 'drag': []},
                    1.0: {'wind_force': [], 'spread': [], 'power': [], 'max_thrust': [], 'area': [], 'drag': []}}
               for sh in shapes}
    
    for sh_name, sh_params in shapes.items():
        for l_aero in [0.0, 1.0]:
            
            p = copy.deepcopy(p_base)
            p.lambda_aero = l_aero
            p.lambda_shape = 0.0  
            
            # --- CORREZIONI CHIAVE PER ANALISI STATICA PURA ---
            p.k_tilt = 1.0                 
            p.min_drone_z_rel = -10.0      
            p.lambda_twist = 0.0           
            p.w_smooth = 0.0               
            p.w_barrier = 0.0              
            # -------------------------------------------------
            
            apply_shape(p, sh_params)
            p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
            
            for f_target in wind_forces_N:
                v_mag = get_v_for_force(p_base, f_target)
                v_wind_vec = np.array([v_mag, 0.0, 0.0])
                
                phi_eq, theta_eq = compute_aerodynamic_equilibrium(p, v_wind_vec)
                phi_tgt = phi_eq * l_aero
                theta_tgt = theta_eq * l_aero
                
                F_wind_actual, proj_area = get_wind_force_and_area(p, v_wind_vec, (phi_tgt, theta_tgt, 0.0))
                drag_xy = np.linalg.norm(F_wind_actual[:2])
                
                state = {
                    'pay_pos': np.zeros(3), 
                    'pay_att': np.array([phi_tgt, theta_tgt, 0.0]), 
                    'pay_vel': np.zeros(3),
                    'pay_omega': np.zeros(3), 
                    'uav_vel': np.zeros((3, p.N))
                }
                
                _, _, _, ff_forces = formation.compute_optimal_formation(
                    p, state, np.zeros(3), np.zeros(3), 0.0, 
                    force_attitude=(phi_tgt, theta_tgt), 
                    F_ext_total=F_wind_actual,
                    com_offset_body=getattr(p, 'CoM_offset', np.zeros(3)),
                    ctx=None 
                )
                
                tensions = np.linalg.norm(ff_forces, axis=0)
                spread = np.max(tensions) - np.min(tensions)
                
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
                
                results[sh_name][l_aero]['wind_force'].append(f_target)
                results[sh_name][l_aero]['spread'].append(spread)
                results[sh_name][l_aero]['power'].append(power)
                results[sh_name][l_aero]['max_thrust'].append(max_thrust)
                results[sh_name][l_aero]['area'].append(proj_area)
                results[sh_name][l_aero]['drag'].append(drag_xy)
                
    print("Dati calcolati con successo. Generazione grafici...")

    # =========================================================================
    # PLOTTING
    # =========================================================================
    C_0 = '#1D7EC2' 
    C_1 = '#E63946' 
    C_2 = '#2A9D8F' 
    C_3 = '#F4A261' 

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    #fig.suptitle('Analisi Aerodinamica: $\lambda_{aero} = 0$ vs $\lambda_{aero} = 1$ ($\lambda_{shape} = 0$ fissa)', 
    #             fontweight='bold', fontsize=16)
    
    x_val = results['sphere'][0.0]['wind_force']

    # --- RIGA 1: BASE SPHERE ---
    ax = axs[0, 0]
    ax.plot(x_val, results['sphere'][0.0]['spread'], label=r'$\lambda_{aero} = 0$', color=C_0, lw=2.5)
    ax.plot(x_val, results['sphere'][1.0]['spread'], label=r'$\lambda_{aero} = 1$', color=C_1, lw=2.5)
    ax.set_title('Cable Tension Spread (Sphere)', fontweight='bold')
    ax.set_ylabel('$T_{max} - T_{min}$ [N]')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(x_val, results['sphere'][0.0]['power'], label=r'$\lambda_{aero} = 0$', color=C_0, lw=2.5)
    ax.plot(x_val, results['sphere'][1.0]['power'], label=r'$\lambda_{aero} = 1$', color=C_1, lw=2.5)
    ax.set_title('Average System Power (Sphere)', fontweight='bold')
    ax.set_ylabel('Power ($\sum T^{1.5}$)')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    ax = axs[0, 2]
    ax.plot(x_val, results['sphere'][0.0]['max_thrust'], label=r'$\lambda_{aero} = 0$', color=C_0, lw=2.5)
    ax.plot(x_val, results['sphere'][1.0]['max_thrust'], label=r'$\lambda_{aero} = 1$', color=C_1, lw=2.5)
    ax.set_title('Max Single Drone Thrust (Sphere)', fontweight='bold')
    ax.set_ylabel('Max Thrust [N]')
    ax.set_xlabel('Horizontal Force [N]')
    ax.axhline(p_base.F_max_thrust, color='gray', linestyle=':', lw=2, label='Hardware Limit')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # --- RIGA 2: CONFRONTO SHAPES ---
    ax = axs[1, 0]
    ax.plot(x_val, results['sphere'][0.0]['area'], color=C_0, lw=2, linestyle='--', alpha=0.3, label=r'Sphere ($\lambda_{aero}=0$)')
    ax.plot(x_val, results['flat_box'][0.0]['area'], color=C_2, lw=2, linestyle='--', alpha=0.3, label=r'Flat Box ($\lambda_{aero}=0$)')
    ax.plot(x_val, results['tall_box'][0.0]['area'], color=C_3, lw=2, linestyle='--', alpha=0.3, label=r'Tall Box ($\lambda_{aero}=0$)')
    ax.plot(x_val, results['sphere'][1.0]['area'], color=C_0, lw=2.5, label=r'Sphere ($\lambda_{aero}=1$)')
    ax.plot(x_val, results['flat_box'][1.0]['area'], color=C_2, lw=2.5, label=r'Flat Box ($\lambda_{aero}=1$)')
    ax.plot(x_val, results['tall_box'][1.0]['area'], color=C_3, lw=2.5, label=r'Tall Box ($\lambda_{aero}=1$)')
    ax.set_title('Exposed Projected Area', fontweight='bold')
    ax.set_ylabel('Projected Area [m²]')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(x_val, results['sphere'][0.0]['drag'], color=C_0, lw=2, linestyle='--', alpha=0.3)
    ax.plot(x_val, results['flat_box'][0.0]['drag'], color=C_2, lw=2, linestyle='--', alpha=0.3)
    ax.plot(x_val, results['tall_box'][0.0]['drag'], color=C_3, lw=2, linestyle='--', alpha=0.3)
    ax.plot(x_val, results['sphere'][1.0]['drag'], color=C_0, lw=2.5, label='Sphere')
    ax.plot(x_val, results['flat_box'][1.0]['drag'], color=C_2, lw=2.5, label='Flat Box')
    ax.plot(x_val, results['tall_box'][1.0]['drag'], color=C_3, lw=2.5, label='Tall Box')
    ax.set_title('Payload Disturbance Force (Drag)', fontweight='bold')
    ax.set_ylabel('Horizontal Drag [N]')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    ax = axs[1, 2]
    ax.plot(x_val, results['sphere'][0.0]['power'], color=C_0, lw=2, linestyle='--', alpha=0.3)
    ax.plot(x_val, results['flat_box'][0.0]['power'], color=C_2, lw=2, linestyle='--', alpha=0.3)
    ax.plot(x_val, results['tall_box'][0.0]['power'], color=C_3, lw=2, linestyle='--', alpha=0.3)
    ax.plot(x_val, results['sphere'][1.0]['power'], color=C_0, lw=2.5, label='Sphere')
    ax.plot(x_val, results['flat_box'][1.0]['power'], color=C_2, lw=2.5, label='Flat Box')
    ax.plot(x_val, results['tall_box'][1.0]['power'], color=C_3, lw=2.5, label='Tall Box')
    ax.set_title('Average System Power (Shape Comparison)', fontweight='bold')
    ax.set_ylabel('Power ($\sum T^{1.5}$)')
    ax.set_xlabel('Horizontal Force [N]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_aero_lambda_comparison()