"""
graph_wind_attack_angle.py
--------------------------
Static analysis: Formation response to wind hitting the Vertex vs the Side.

This script sweeps the wind speed from 0 to 25 m/s and evaluates the C.O.R.D.A.
optimizer under two specific attack angles:
  1. Vertex (0°)  - Wind hits a drone head-on.
  2. Side   (45°) - Wind hits exactly between two drones (perpendicular to the edge).

Metrics plotted (4 Panels):
  Panel A - Energy (Power Index)
  Panel B - Stability: Load Imbalance (Tension Std. Dev.)
  Panel C - Stability: Maximum Tension (Saturation margin)
  Panel D - Stability: Minimum Tension (Slack margin)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from parameters import SysParams
import formation as formation
import physics

# ==============================================================================
# PHYSICAL HELPERS (from graph_best_yaw.py)
# ==============================================================================
def get_wind_force_at_attitude(p, v_wind_world: np.ndarray, attitude: tuple) -> np.ndarray:
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    if wind_mag < 1e-3:
        return np.zeros(3)

    wind_dir      = v_wind_world / wind_mag
    R_pay         = physics.get_rotation_matrix(phi, theta, psi)
    wind_in_body  = R_pay.T @ wind_dir

    if getattr(p, 'payload_shape', '') in ['box', 'rect', 'square']:
        A_x = p.pay_w * p.pay_h
        A_y = p.pay_l * p.pay_h
        A_z = p.pay_l * p.pay_w
        proj_area = (A_x * abs(wind_in_body[0])
                   + A_y * abs(wind_in_body[1])
                   + A_z * abs(wind_in_body[2]))
    elif getattr(p, 'payload_shape', '') == 'sphere':
        proj_area = np.pi * p.R_disk ** 2          
    else:                                           
        A_side    = 2.0 * p.R_disk * p.pay_h
        A_top     = np.pi * p.R_disk ** 2
        sin_tilt  = np.sqrt(wind_in_body[0] ** 2 + wind_in_body[1] ** 2)
        proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])

    f_wind = (0.5 * p.rho * p.Cd_pay * proj_area * wind_mag ** 2) * wind_dir
    return f_wind

def compute_aerodynamic_equilibrium(p, v_wind_world: np.ndarray, base_yaw: float = 0.0):
    phi, theta = 0.0, 0.0
    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)
    c_psi, s_psi = np.cos(base_yaw), np.sin(base_yaw)
    R_z_inv = np.array([[c_psi,  s_psi, 0], [-s_psi, c_psi, 0], [0, 0, 1]])

    for _ in range(15):
        f_wind = get_wind_force_at_attitude(p, v_wind_world, (phi, theta, base_yaw))
        F_tot  = np.array([0.0, 0.0, -m_tot * p.g]) + f_wind
        F_req  = -F_tot
        F_req_body = R_z_inv @ F_req
        theta = np.arctan2(F_req_body[0], abs(F_req_body[2]))
        phi   = np.arctan2(-F_req_body[1], np.sqrt(F_req_body[0] ** 2 + F_req_body[2] ** 2))
    return phi, theta, f_wind

def compute_power_index(ff_forces: np.ndarray, p, v_wind: np.ndarray) -> float:
    v_mag = np.linalg.norm(v_wind)
    if v_mag > 1e-3:
        f_aero_drone = (0.5 * p.rho * p.Cd_uav * p.A_uav * v_mag ** 2 * (v_wind / v_mag))
    else:
        f_aero_drone = np.zeros(3)
    g_vec = np.array([0.0, 0.0, p.m_drone * p.g]) 
    p_total = 0.0
    for i in range(p.N):
        thrust_vec = ff_forces[:, i] + g_vec - f_aero_drone
        p_total   += np.linalg.norm(thrust_vec) ** 1.5
    return p_total

class MockCtx:
    def __init__(self, p):
        self.last_theta_opt       = np.radians(getattr(p, 'theta_ref', 30.0))
        self.last_stable_yaw      = 0.0
        self.last_damp_correction = np.zeros(p.N)
        self.last_T_final         = np.zeros(p.N) 
        self.dot_T_filt           = np.zeros(p.N)
        self.F_xy_filt            = np.zeros(2)
        self.e_filt_despun        = np.zeros(3)

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
def run_wind_attack_analysis():
    p = SysParams()

    # Per isolare l'effetto della FORMAZIONE, usiamo un payload simmetrico
    p.payload_shape = 'sphere'
    p.R_disk = 0.3
    p.m_payload = 3.0
    p.enable_sloshing = False

    p.w_T = 1.0; p.w_ref = 50.0; p.w_barrier = 100.0; p.w_cond = 100.0
    p.w_resid_f = 300.0; p.w_resid_m = 100.0
    p.w_smooth = 0.0; p.max_angle_variation = 20.0
    
    # Nessuna deformazione o tilt aerodinamico per isolare il comportamento rigido
    p.lambda_shape = 0.0
    p.lambda_aero  = 0.0
    p.lambda_static = 0.0
    p.lambda_twist  = 0.0

    p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)

    wind_speeds = np.linspace(0, 15.0, 45) # Sweep da 0 a 22 m/s
    area_sphere = np.pi * (p.R_disk ** 2)
    wind_forces = 0.5 * p.rho * p.Cd_pay * area_sphere * (wind_speeds ** 2)

    cases = [
        {"name": "Vertex Wind", "angle": 0.0, "color": "royalblue", "data": {"pow": [], "std": [], "tmax": [], "tmin": []}},
        {"name": "Side Wind", "angle": 45.0, "color": "crimson", "data": {"pow": [], "std": [], "tmax": [], "tmin": []}}
    ]

    print("Running Vertex vs Side analysis ...")
    
    for case in cases:
        angle_rad = np.radians(case["angle"])
        ctx = MockCtx(p)
        
        for v in wind_speeds:
            # Calcolo del vettore vento
            wind_vec_world = np.array([v * np.cos(angle_rad), v * np.sin(angle_rad), 0.0])
            p.wind_vel = wind_vec_world
            
            # Equilibrio
            phi_eq, theta_eq, _ = compute_aerodynamic_equilibrium(p, wind_vec_world, base_yaw=0.0)
            f_wind = get_wind_force_at_attitude(p, wind_vec_world, (0.0, 0.0, 0.0))

            state = {
                'uav_pos': np.zeros((3, p.N)), 'uav_vel': np.zeros((3, p.N)),
                'pay_pos': np.zeros(3), 'pay_vel': np.zeros(3),
                'pay_att': np.array([0.0, 0.0, 0.0]), 'pay_omega': np.zeros(3),
                'int_uav': np.zeros((3, p.N))
            }

            _, _, _, ff_forces = formation.compute_optimal_formation(
                p, state, acc_cmd_pay=np.zeros(3), acc_ang_cmd_pay=np.zeros(3),
                ref_yaw=0.0, force_attitude=(0.0, 0.0),
                F_ext_total=f_wind, F_aero_for_moment=f_wind,
                ctx=ctx, com_offset_body=p.CoM_offset
            )

            tensions = np.linalg.norm(ff_forces, axis=0)

            case["data"]["pow"].append(compute_power_index(ff_forces, p, wind_vec_world))
            case["data"]["std"].append(float(np.std(tensions)))
            case["data"]["tmax"].append(float(np.max(tensions)))
            case["data"]["tmin"].append(float(np.min(tensions)))

    print("Analysis complete. Plotting...")

    # --------------------------------------------------------------------------
    # Plotting (4 Panels)
    # --------------------------------------------------------------------------
    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)
    T_safe_base = p.k_safe * (m_tot * p.g / p.N)
    T_max_limit = (p.F_max_thrust - p.m_drone * p.g) * p.k_limit

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Panel A: Power
    for case in cases:
        axs[0].plot(wind_forces, case["data"]["pow"], color=case["color"], lw=2, label=case["name"])
    axs[0].set_title('Energetic Efficiency vs. External Force', fontweight='bold', fontsize=11)
    axs[0].set_ylabel(r'Power Index ($\sum T^{1.5}$)')
    axs[0].legend(fontsize=10)

    # Panel B: Load Imbalance
    for case in cases:
        axs[1].plot(wind_forces, case["data"]["std"], color=case["color"], lw=2)
    axs[1].set_title('Formation Load Imbalance (Stability)', fontweight='bold', fontsize=11)
    axs[1].set_ylabel('Tension Std. Dev. [N]')

    # Panel C: Max Tension
    for case in cases:
        axs[2].plot(wind_forces, case["data"]["tmax"], color=case["color"], lw=2, label=case["name"])
    axs[2].axhline(T_max_limit, color='black', linestyle=':', lw=1.5, label=f'Saturation limit')
    axs[2].fill_between(wind_forces, T_max_limit, T_max_limit + 10, color='red', alpha=0.1)
    axs[2].set_title('Maximum Cable Tension (Risk of Motor Saturation)', fontweight='bold', fontsize=11)
    axs[2].set_ylabel('Max Tension [N]')
    axs[2].legend(fontsize=9, loc='upper left')

    # Panel D: Min Tension
    ax = axs[3]
    for case in cases:
        ax.plot(wind_forces, case["data"]["tmin"], color=case["color"], lw=2, label=case["name"])
    
    # Calcolo della tensione di hovering (Punto di partenza delle curve)
    # m_tot è la massa del payload (2.5kg) + liquido (1.5kg) = 4kg -> T_hover ~ 9.81 N
    T_hover = (m_tot * p.g) / p.N 
    
    ax.axhline(T_safe_base, color='black', linestyle=':', lw=1.5, label=f'Slack limit')
    ax.axhline(0.0, color='gray', lw=1.0)
    ax.fill_between(wind_forces, -5, T_safe_base, color='orange', alpha=0.1)
    
    ax.set_title('Minimum Cable Tension (Risk of Slack Cables)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Min Tension [N]')
    ax.set_xlabel('External Force [N]', fontsize=11)
    
    # Spostiamo la legenda in alto a destra così non copre le curve che scendono
    ax.legend(fontsize=9, loc='upper right') 

    # REGOLAZIONE ZOOM: Impostiamo il tetto al 20% sopra il valore di hovering
    # Così vedrai chiaramente la curva che parte da ~9.8 N e scende verso lo zero.
    ax.set_ylim(-1.0, T_hover * 1.4)

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
    run_wind_attack_analysis()