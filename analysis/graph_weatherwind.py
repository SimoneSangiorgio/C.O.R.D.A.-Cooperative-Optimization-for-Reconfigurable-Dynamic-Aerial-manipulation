import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa i moduli del progetto
import formation as formation
from parameters import SysParams
import physics
import graph_ellisswind as ge

# Mock del Context per gestire lo stato della formazione (yaw e filtri)
class MockCtx:
    def __init__(self, p):
        self.last_theta_opt = np.radians(getattr(p, 'theta_ref', 30.0))
        self.last_stable_yaw = 0.0
        self.F_xy_filt = np.zeros(2)
        self.e_filt_despun = np.zeros(3)
        self.last_yaw_target_raw = 0.0
        self.last_yaw_abs_tgt = 0.0
        self.lambda_traj_effective = 1.0 # Forza l'attivazione del Weather-Vane

def get_wind_force_and_area(p, v_wind_world, attitude):
    """Calcola la forza del vento usando l'area proiettata."""
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    wind_dir = v_wind_world / wind_mag if wind_mag > 1e-3 else np.array([1.0, 0.0, 0.0])
    R_pay = physics.get_rotation_matrix(phi, theta, psi)
    wind_in_body = R_pay.T @ wind_dir

    A_x = p.pay_w * p.pay_h
    A_y = p.pay_l * p.pay_h
    A_z = p.pay_l * p.pay_w
    pay_proj_area = A_x * abs(wind_in_body[0]) + A_y * abs(wind_in_body[1]) + A_z * abs(wind_in_body[2])
    
    f_wind = 0.5 * p.rho * getattr(p, 'Cd_pay', 0.8) * pay_proj_area * (wind_mag**2) * wind_dir
    return f_wind

def custom_plot_xy_view(ax, p, pos_adapt, att_vecs_adapt, att_target, v_wind_vec, limit=3.0):
    """Disegna esplicitamente il carico come un rettangolo ruotato sul piano 2D."""
    ix, iy, iz = 0, 1, 2
    PAYLOAD_ZORDER = 10
    phi, theta, yaw = att_target

    # 1. Disegna il Vettore Vento
    w_mag = np.linalg.norm(v_wind_vec[:2])
    if w_mag > 0.1:
        w_dir = v_wind_vec[:2] / w_mag
        start_pt = -w_dir * (limit * 0.8)
        ax.arrow(start_pt[0]-0.5, start_pt[1]-0.5, w_dir[0]*limit*0.4, w_dir[1]*limit*0.4,
                 width=limit*0.025 if limit else 0.1, color="#0092BE", alpha=0.6, zorder=0)
        ax.text(start_pt[0]*1.06-0.5, start_pt[1]*1.06-0.5, f"{w_mag:.1f} m/s", color='#0092BE', fontweight='bold', ha='left', va='center_baseline', fontsize=9)

    # 2. Disegna il Payload in 2D in maniera infallibile
    lx, ly = p.pay_l / 2, p.pay_w / 2
    # Vertici del rettangolo nel frame del corpo (Z=0)
    corners_body = np.array([
        [lx, ly, 0], [-lx, ly, 0], [-lx, -ly, 0], [lx, -ly, 0]
    ]).T
    
    # Ruota i vertici nel mondo in base allo yaw corrente
    R = physics.get_rotation_matrix(phi, theta, yaw)
    corners_world = R @ corners_body

    polygon = mpatches.Polygon(corners_world[:2, :].T, closed=True, 
                               facecolor='#FF9100', edgecolor='black', alpha=0.8, zorder=PAYLOAD_ZORDER)
    ax.add_patch(polygon)
    ax.scatter(0, 0, color='black', marker='*', zorder=PAYLOAD_ZORDER+1)
    
    # 3. Disegna l'ellisse rossa del perimetro della formazione
    ge.draw_perimeter(ax, pos_adapt, 'red', 0.5, style='--', zorder=PAYLOAD_ZORDER, view='XY')

    # 4. Disegna Droni e Cavi
    for i in range(p.N):
        depth_val = att_vecs_adapt[iz, i]
        cable_zorder = PAYLOAD_ZORDER + 1 if depth_val > 0 else PAYLOAD_ZORDER - 1
        ax.plot([att_vecs_adapt[ix, i], pos_adapt[ix, i]], [att_vecs_adapt[iy, i], pos_adapt[iy, i]], 
                color='black', lw=1.5, zorder=cable_zorder)
        ax.scatter(pos_adapt[ix, i], pos_adapt[iy, i], c='black', s=60, marker='h', zorder=cable_zorder+1)

    # 5. Calcola e disegna il centroide della formazione
    centroid_pos = np.mean(pos_adapt, axis=1)
    ax.scatter(centroid_pos[ix], centroid_pos[iy], color='black', marker='+', s=50, zorder=26)

    # Impostazioni grafiche
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

def plot_weathervane_sequence():
    print("Avvio simulazione dinamica Weather Vaning...")
    
    p_base = SysParams()
    p_base.N = 4
    p_base.m_payload = 2.0
    p_base.m_liquid = 1.0

    # Usiamo 'rect' per evitare bug con le funzioni fisiche e di calcolo d'area
    p_base.payload_shape = 'rect'
    p_base.pay_l = 1.2
    p_base.pay_w = 0.6
    p_base.pay_h = 0.4
    p_base.R_disk = 0.6 # Fallback di sicurezza

    p_base.uav_offsets, p_base.attach_vecs, p_base.geo_radius = formation.compute_geometry(p_base)

    p_base.lambda_shape = 0.0  
    p_base.lambda_aero = 0.0 
    p_base.lambda_static = 1.0 
    p_base.lambda_CoM = 1.0
    p_base.lambda_twist = 1.0

    # Vento a 45 gradi costanti
    wind_speed = 5.0
    wind_dir = np.array([0.7071, 0.7071, 0.0]) 
    v_wind_vec = wind_speed * wind_dir
    p_base.wind_vel = v_wind_vec

    ctx = MockCtx(p_base)

    state = {
        'pay_pos': np.zeros(3), 'pay_att': np.zeros(3), 'pay_vel': np.zeros(3),
        'pay_omega': np.zeros(3), 'uav_vel': np.zeros((3, p_base.N))
    }

    snapshots = []
    
    # 90 step di simulazione per mostrare l'effetto "molla" fino al raggiungimento dello Steady State
    for step in range(90):
        attitude = (0.0, 0.0, ctx.last_stable_yaw)
        F_wind_actual = get_wind_force_and_area(p_base, v_wind_vec, attitude)
        state['pay_att'] = np.array(attitude)
        
        pos_adapt, _, _, _ = formation.compute_optimal_formation(
            p_base, state, np.zeros(3), np.zeros(3), 0.0, 
            force_attitude=(0.0, 0.0), F_ext_total=F_wind_actual, M_ext_total=np.zeros(3),
            com_offset_body=np.zeros(3), ctx=ctx
        )
        
        # Stampa su console l'angolo per verificare che il sistema stia girando!
        if step % 15 == 0:
            print(f"Step {step}: Yaw attuale = {np.degrees(ctx.last_stable_yaw):.2f}°")

        if step == 0 or step == 25 or step == 85:
            snapshots.append({
                'pos': pos_adapt.copy(),
                'yaw': ctx.last_stable_yaw,
                'time': step
            })

    # Creazione della Figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('', fontweight='bold', fontsize=16)
    labels = ["Initial State (t=0)", "Transient State", "Steady State (Aligned)"]

    for i, snap in enumerate(snapshots):
        ax = axes[i]
        yaw = snap['yaw']
        pos = snap['pos']
        
        att_target = (0.0, 0.0, yaw)
        att_vecs_adapt = physics.get_rotation_matrix(*att_target) @ p_base.attach_vecs
        
        # Applichiamo la nostra funzione 2D infallibile

        max_uav_dist = np.max(np.abs(pos)) 
        dynamic_limit = max_uav_dist * 1.2

        custom_plot_xy_view(ax, p_base, pos, att_vecs_adapt, att_target, v_wind_vec, limit=dynamic_limit)
        ax.set_title(f"{labels[i]}\nYaw: {np.degrees(yaw):.1f}°", fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_weathervane_sequence()