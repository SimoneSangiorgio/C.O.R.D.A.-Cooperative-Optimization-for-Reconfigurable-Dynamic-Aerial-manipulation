import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import copy
from matplotlib.patches import Ellipse, Rectangle, Circle, Polygon
from scipy.spatial import ConvexHull

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa i moduli del progetto
import formation as formation
from parameters import SysParams
import physics 

def get_wind_force_at_attitude(p, v_wind_world, attitude):
    """
    Calcola la forza del vento per un DATO assetto.
    Replica esattamente la logica di physics.py / compute_derivatives.
    """
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    if wind_mag < 1e-3: return np.zeros(3)
    
    wind_dir = v_wind_world / wind_mag
    R_pay = physics.get_rotation_matrix(phi, theta, psi)
    wind_in_body = R_pay.T @ wind_dir
    
    if hasattr(p, 'payload_shape') and p.payload_shape in ['box', 'rect', 'square']:
        A_x, A_y, A_z = p.pay_w * p.pay_h, p.pay_l * p.pay_h, p.pay_l * p.pay_w
        pay_proj_area = A_x * abs(wind_in_body[0]) + A_y * abs(wind_in_body[1]) + A_z * abs(wind_in_body[2])
    else:
        # CYLINDER Logic (Area Side + Area Top proiettata)
        A_side = 2.0 * p.R_disk * p.pay_h
        A_top = np.pi * p.R_disk**2
        # sin_tilt è la componente del vento perpendicolare all'asse Z del cilindro
        sin_tilt = np.sqrt(wind_in_body[0]**2 + wind_in_body[1]**2)
        pay_proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])
        
    f_wind = 0.5 * p.rho * p.Cd_pay * pay_proj_area * (wind_mag**2) * wind_dir
    return f_wind

def compute_aerodynamic_equilibrium(p, v_wind_world):
    """
    Trova l'angolo di equilibrio 'naturale' (Lambda=1.0) iterando.
    """
    phi, theta, psi = 0.0, 0.0, 0.0
    
    # Iteriamo per trovare l'assetto dove il momento è nullo (cavo allineato alla forza)
    for _ in range(15): 
        f_wind = get_wind_force_at_attitude(p, v_wind_world, (phi, theta, psi))
        
        # Forza Totale Apparente (Gravità + Vento)
        # Nota: L'equilibrio statico si ha quando l'asse Z del payload è opposto alla forza totale
        F_tot = np.array([0.0, 0.0, -p.m_payload * p.g]) + f_wind
        
        F_req = -F_tot # La direzione in cui i cavi devono tirare
        
        # Aggiorna angoli per allineare Z a F_req
        theta = np.arctan2(F_req[0], abs(F_req[2]))
        phi = np.arctan2(-F_req[1], np.sqrt(F_req[0]**2 + F_req[2]**2))
    
    return phi, theta, f_wind

def generate_static_comparison():
    # 1. SETUP PARAMETRI
    p_base = SysParams()
    p_base.uav_offsets, p_base.attach_vecs, p_base.geo_radius = formation.compute_geometry(p_base)
    
    # --- CONFIGURAZIONE VENTO (Stesso della Simulazione) ---
    # In Simulation: initial_wind_vec = [3.5, 3.5, 0] -> Mag ~4.95
    v_wind_vec = np.array([-3, -3, 0.0])
    v_wind_ms = np.linalg.norm(v_wind_vec)
    dir_norm = v_wind_vec / v_wind_ms

    arrow_len = 2.0    
    start_dist = 3.0   

    # --- CALCOLO SCENARI ---
    
    # A. Caso RIGIDO (Reference) - Assetto forzato a 0
    # Calcoliamo la forza che agisce se il carico è PIATTO.
    F_wind_rigid = get_wind_force_at_attitude(p_base, v_wind_vec, (0,0,0))
    
    # B. Caso EQUILIBRIO (Lambda=1.0)
    # Calcoliamo l'assetto di equilibrio e la forza risultante (che sarà MAGGIORE per il cilindro piatto!)
    phi_eq, theta_eq, F_wind_eq_calc = compute_aerodynamic_equilibrium(p_base, v_wind_vec)
    
    print(f"Wind Speed: {v_wind_ms:.2f} m/s")
    print(f"Rigid Force (Flat): {np.linalg.norm(F_wind_rigid):.2f} N")
    print(f"Adaptive Force (Tilted): {np.linalg.norm(F_wind_eq_calc):.2f} N (Increased area exposure!)")
    print(f"Equilibrium Angles -> Pitch: {np.degrees(theta_eq):.2f}°, Roll: {np.degrees(phi_eq):.2f}°")

    state = {
        'pay_pos': np.zeros(3), 'pay_att': np.zeros(3), 'pay_vel': np.zeros(3),
        'pay_omega': np.zeros(3), 'uav_vel': np.zeros((3, p_base.N))
    }
    
    # ---------------------------------------------------------
    # SCENARIO 1: Rigido (Nominal)
    # ---------------------------------------------------------
    p_rigid = copy.deepcopy(p_base)
    p_rigid.lambda_shape = 0.0
    p_rigid.lambda_aero = 0.0
    
    state_rigid = copy.deepcopy(state)
    state_rigid['pay_att'] = np.zeros(3)
    
    # Passiamo F_wind_rigid. L'ottimizzatore vedrà una forza MINORE.
    pos_ref, _, _, _ = formation.compute_optimal_formation(
        p_rigid, state_rigid, np.zeros(3), np.zeros(3), 0.0, 
        force_attitude=(0.0, 0.0), 
        F_ext_total=F_wind_rigid 
    )

    # ---------------------------------------------------------
    # SCENARIO 2: Adattivo (Corrected Logic)
    # ---------------------------------------------------------
    p_adapt = copy.deepcopy(p_base)
    
    # === IMPOSTAZIONI ===
    p_adapt.lambda_shape = 0.0   # Deformazione Ellittica DISABILITATA (come da prompt)
    p_adapt.lambda_aero = 1.0    # Tilt ABILITATO (1.0 = Allineamento completo)
    # ====================

    p_adapt.yaw_force_min = 1e9  
    p_adapt.yaw_force_max = 1e9 + 10.0

    # 1. Calcola l'assetto target in base a lambda_aero
    phi_target = phi_eq * p_adapt.lambda_aero
    theta_target = theta_eq * p_adapt.lambda_aero
    
    # 2. Ricalcola la forza ESATTA per questo assetto
    F_wind_actual = get_wind_force_at_attitude(p_adapt, v_wind_vec, (phi_target, theta_target, 0.0))
    
    state_adapt = copy.deepcopy(state)
    state_adapt['pay_att'] = np.array([phi_target, theta_target, 0.0])
    
    # Passiamo F_wind_actual. L'ottimizzatore vedrà una forza MAGGIORE -> Allargherà la formazione.
    pos_adapt, _, _, _ = formation.compute_optimal_formation(
        p_adapt, state_adapt, np.zeros(3), np.zeros(3), 0.0, 
        force_attitude=(phi_target, theta_target),
        F_ext_total=F_wind_actual
    )

    # 3. PLOTTING
    fig = plt.figure(figsize=(15, 8))
    
    # Ripeto le funzioni di disegno per completezza

    def draw_payload_geometry(ax, p, view='XY', att=(0,0,0), style='solid', alpha=1.0, 
                            color="#FF7B00", att_color="black", zorder=1):
        
        phi, theta, psi = att
        R = physics.get_rotation_matrix(phi, theta, psi) 

        # Configurazione Bordi (come richiesto: chiari)
        edge_rgba = (0.5, 0.5, 0.5, 0.2) 

        # 1. Calcolo Punti di Attacco
        att_vecs_rotated = R @ p.attach_vecs

        # 2. Generazione Geometria
        shape = getattr(p, 'payload_shape', 'box')

        if shape == 'cylinder':
            radius = getattr(p, 'R_disk', 0.1)
            height = getattr(p, 'pay_h', 0.1)
            
            t = np.linspace(0, 2*np.pi, 60)
            x_c, y_c = radius * np.cos(t), radius * np.sin(t)
            
            # Genera i punti 3D dei tappi (World Frame)
            verts_top_world = R @ np.vstack([x_c, y_c, np.full_like(t, height/2)])
            verts_bot_world = R @ np.vstack([x_c, y_c, np.full_like(t, -height/2)])
            
            # --- ALGORITMO DI VISIBILITÀ (DEPTH SORTING) ---
            if view == 'XY':
                idx_x, idx_y = 0, 1
                z_avg_top = np.mean(verts_top_world[2, :])
                z_avg_bot = np.mean(verts_bot_world[2, :])
                is_top_front = z_avg_top > z_avg_bot
                
            elif view == 'XZ':
                idx_x, idx_y = 0, 2
                y_avg_top = np.mean(verts_top_world[1, :])
                y_avg_bot = np.mean(verts_bot_world[1, :])
                is_top_front = y_avg_top < y_avg_bot 
                
            else: 
                idx_x, idx_y = 1, 2
                is_top_front = True

            # Proiezione 2D
            vx_top, vy_top = verts_top_world[idx_x, :], verts_top_world[idx_y, :]
            vx_bot, vy_bot = verts_bot_world[idx_x, :], verts_bot_world[idx_y, :]

            # Funzione helper di disegno
            def _draw_part(vx, vy):
                ax.fill(vx, vy, facecolor=color, alpha=alpha, 
                        edgecolor=edge_rgba, linewidth=0.5, zorder=zorder)

            # --- CALCOLO MANTELLO (Silhouette) ---
            # Uniamo tutti i punti e calcoliamo l'involucro convesso
            all_x = np.concatenate([vx_top, vx_bot])
            all_y = np.concatenate([vy_top, vy_bot])
            points_2d = np.column_stack([all_x, all_y])
            
            # Calcola Hull (richiede scipy.spatial.ConvexHull)
            hull = ConvexHull(points_2d)
            hull_x = points_2d[hull.vertices, 0]
            hull_y = points_2d[hull.vertices, 1]

            # Disegna in ordine: Faccia Dietro -> Mantello -> Faccia Davanti
            if is_top_front:
                _draw_part(vx_bot, vy_bot) # Fondo (Dietro)
                _draw_part(hull_x, hull_y) # Mantello (Corpo)
                _draw_part(vx_top, vy_top) # Top (Davanti)
            else:
                _draw_part(vx_top, vy_top) # Top (Dietro)
                _draw_part(hull_x, hull_y) # Mantello (Corpo)
                _draw_part(vx_bot, vy_bot) # Fondo (Davanti)

        else: 
            # --- BOX / RECT ---
            # (Codice invariato, ma applicando edge_rgba corretto)
            lx, ly, h = p.pay_l/2, p.pay_w/2, p.pay_h/2
            corners = np.array([[lx,ly,h], [-lx,ly,h], [-lx,-ly,h], [lx,-ly,h],
                                [lx,ly,-h], [-lx,ly,-h], [-lx,-ly,-h], [lx,-ly,-h]]).T
            corn_w = R @ corners
            
            # Mappatura assi vista
            if view == 'XY':   v_idx, idx_x, idx_y, sign = 2, 0, 1, 1  # Usa Z
            elif view == 'XZ': v_idx, idx_x, idx_y, sign = 1, 0, 2, -1 # Usa Y (invertito)
            else:              v_idx, idx_x, idx_y, sign = 0, 1, 2, 1
            
            faces = [[0,1,2,3], [4,7,6,5], [0,3,7,4], [1,5,6,2], [0,4,5,1], [3,2,6,7]]
            
            # Ordinamento facce per profondità (dal più lontano al più vicino)
            face_depths = []
            for f_idx in faces:
                # Calcola profondità media della faccia lungo l'asse di vista
                depth = np.mean(corn_w[v_idx, f_idx]) * sign
                face_depths.append((depth, f_idx))
            
            # Ordina crescente (Disegna prima i lontani/piccoli, poi i vicini/grandi)
            face_depths.sort(key=lambda x: x[0])
            
            for _, f_idx in face_depths:
                ax.fill(corn_w[idx_x, f_idx], corn_w[idx_y, f_idx], 
                        facecolor=color, alpha=alpha, edgecolor=edge_rgba, linewidth=0.5, zorder=zorder)

        # 3. Disegno Attacchi (Occlusione corretta basata su assi cartesiani)
        if view == 'XY': view_ax, view_dir = 2, 1   # Z axis, looking from +
        elif view == 'XZ': view_ax, view_dir = 1, -1 # Y axis, looking from -
        else: view_ax, view_dir = 0, 1

        for i in range(att_vecs_rotated.shape[1]):
            pos = att_vecs_rotated[:, i]
            # Se la componente lungo l'asse di vista "punta" verso la camera rispetto al centro (0)
            # Semplificazione: se siamo nel "lato" vicino della figura
            if pos[view_ax] * view_dir > -getattr(p, 'R_disk', 0.1)*0.2:  
                # ZORDER Attacchi leggermente sopra il corpo (zorder + 1)
                ax.scatter(pos[idx_x], pos[idx_y], c=att_color, alpha=alpha, s=25, zorder=zorder+0.1, edgecolors='none')

    def draw_perimeter(ax, positions, color, alpha, label, style='--', zorder=1):
        x, y = positions[0, :], positions[1, :]
        cx, cy = np.mean(x), np.mean(y)
        points = np.vstack((x - cx, y - cy))
        cov = np.cov(points, bias=True)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort(); vals = vals[order]; vecs = vecs[:, order]
        theta_ell = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
        width, height = 2 * np.sqrt(2 * vals[1]), 2 * np.sqrt(2 * vals[0])
        ell = Ellipse(xy=(cx, cy), width=width, height=height, angle=theta_ell,
                      edgecolor=color, facecolor='none', linestyle=style, linewidth=1.5, label=label, alpha=alpha, zorder=zorder)
        ax.add_patch(ell)

    def draw_side_segment(ax, positions, color, alpha, label, style='--', zorder=1):
        mu = np.mean(positions, axis=1)

        P_centered = positions - mu[:, np.newaxis]

        U, s, Vh = np.linalg.svd(P_centered)

        P_proj = U.T @ P_centered # Proiezione nel sistema di coordinate locale (Principal Components)

        radii = np.linalg.norm(P_proj[:2, :], axis=0)
        r_avg = np.mean(radii)

        theta = np.linspace(0, 2*np.pi, 100)

        x_local = r_avg * np.cos(theta)
        y_local = r_avg * np.sin(theta)


        z_local = np.zeros_like(theta) # Siamo sul piano 2D locale
        circle_local = np.vstack([x_local, y_local, z_local])

        circle_world = U @ circle_local + mu[:, np.newaxis]

        ax.plot(circle_world[0, :], circle_world[2, :], 
                color=color, linestyle=style, linewidth=1.5, 
                alpha=alpha, label=label, zorder=zorder)


    com_x, com_y, _ = p_base.CoM_offset
    R_act = physics.get_rotation_matrix(phi_target, theta_target, 0.0)
    com_world = R_act @ p_base.CoM_offset
    F_tot_vis = F_wind_actual + np.array([0, 0, -p_base.m_payload * p_base.g])
    norm_F_tot = np.linalg.norm(F_tot_vis)
    att_vecs_rigid = p_base.attach_vecs
    att_vecs_adapt = R_act @ p_base.attach_vecs
    PAYLOAD_ZORDER = 10


    # VISTA TOP --------------------------------------------------------------------------------
    ax_top = fig.add_subplot(121)
    
    # --- RIGID (SFONDO) ---
    draw_payload_geometry(ax_top, p_base, view='XZ', att=(0,0,0), style=':', alpha=0.1, color='gray', zorder=1)
    draw_perimeter(ax_top, pos_ref, 'gray', 0.2, 'Nominal (Rigid)', style=':', zorder=1)

    for i in range(p_base.N):
        ax_top.plot([att_vecs_rigid[0, i], pos_ref[0, i]], [att_vecs_rigid[1, i], pos_ref[1, i]], 
                    color='gray', ls=':', alpha=0.1, zorder=1)

    # --- ADAPTIVE (PRIMO PIANO con OCCLUSIONE) ---
    draw_payload_geometry(ax_top, p_base, view='XY', att=(phi_target, theta_target, 0.0), alpha=1.0, zorder=PAYLOAD_ZORDER)
    draw_perimeter(ax_top, pos_adapt, 'red', 0.5, f'Adaptive)', style='--', zorder=PAYLOAD_ZORDER)
    
    for i in range(p_base.N):
        # Occlusione basata su Z (asse di profondità per vista XY)
        # Se Z > 0, il punto è "in alto" (verso la camera) -> SOPRA il payload
        # Se Z < 0, il punto è "in basso" (lontano dalla camera) -> SOTTO il payload
        att_z = att_vecs_adapt[2, i]
        
        if att_z < 0:
            cable_zorder = PAYLOAD_ZORDER - 1 # Dietro
        else:
            cable_zorder = PAYLOAD_ZORDER + 1 # Davanti
            
        ax_top.plot([att_vecs_adapt[0, i], pos_adapt[0, i]], [att_vecs_adapt[1, i], pos_adapt[1, i]], 
                    color='black', zorder=cable_zorder)
        ax_top.scatter(pos_adapt[0, i], pos_adapt[1, i], c='black', s=50, zorder=cable_zorder)
    
    # Decorazioni
    ax_top.scatter(com_x, com_y, color='black', marker='+', s=50, label='CoM', zorder=15)

    ax_top.arrow(-dir_norm[0]*start_dist, -dir_norm[1]*start_dist, dir_norm[0]*arrow_len, dir_norm[1]*arrow_len, 
             width=0.08, color="#0092BE", alpha=0.5, ec='None', zorder=0)
    
    ax_top.text(-dir_norm[0]*start_dist , -dir_norm[1]*start_dist , 
                f"{v_wind_ms:.1f} m/s", color="#0092BE", fontweight='bold')
    
    ax_top.arrow(com_world[0], com_world[1], F_tot_vis[0]*0.05, F_tot_vis[1]*0.05, 
             width=0.05, fc='green', ec='None', alpha=0.5, label='Total Force', zorder=0)
    
    ax_top.set_title(f"XY")
    ax_top.grid(True, alpha=0.3); ax_top.axis('equal')
    ax_top.legend(loc='upper right')

    # VISTA SIDE ------------------------------------------------------------------------------------------------------
    ax_side = fig.add_subplot(122)
    
    # 1. Disegna PRIMA il sistema Rigid (ZORDER BASSO = Sfondo totale)
    draw_payload_geometry(ax_side, p_base, view='XZ', att=(0,0,0), style=':', alpha=0.1, color='gray', zorder=1)
    draw_side_segment(ax_side, pos_ref, 'gray', 0.2, 'Formation (Rigid)', style=':', zorder=1)

    # Disegna cavi Rigid (tutti sullo sfondo)
    for i in range(p_base.N):
        ax_side.plot([att_vecs_rigid[0, i], pos_ref[0, i]], [att_vecs_rigid[2, i], pos_ref[2, i]], 
                     color='grey', alpha=0.1, ls=':', zorder=1)

    # 2. Disegna il Payload Adapt (ZORDER MEDIO)
    # Impostiamo un zorder fisso, ad esempio 10
    draw_payload_geometry(ax_side, p_base, view='XZ', att=(phi_target, theta_target, 0.0), zorder=PAYLOAD_ZORDER)
    draw_side_segment(ax_side, pos_adapt, 'red', 0.6, 'Formation (Adaptive)', style='--', zorder=PAYLOAD_ZORDER)

    # 3. Disegna i Cavi Adapt con ORDINAMENTO DI PROFONDITÀ (Depth Sorting)
    for i in range(p_base.N):

        att_y = att_vecs_adapt[1, i]

        if att_y > 0:
            cable_zorder = PAYLOAD_ZORDER - 1  # DIETRO al cilindro (es. 9)
        else:
            cable_zorder = PAYLOAD_ZORDER + 1  # DAVANTI al cilindro (es. 11)

        # Disegna Cavo
        ax_side.plot([att_vecs_adapt[0, i], pos_adapt[0, i]], 
                     [att_vecs_adapt[2, i], pos_adapt[2, i]], 
                     color='black', zorder=cable_zorder)
        
        # Disegna Drone (Opzionale: puoi usare lo stesso zorder del cavo o metterlo sempre in primo piano)
        ax_side.scatter(pos_adapt[0, i], pos_adapt[2, i], c='black', s=50, zorder=cable_zorder)

    ax_side.arrow(-dir_norm[0]*start_dist, 1.5, dir_norm[0]*arrow_len, dir_norm[2]*arrow_len, 
              width=0.08, color="#0092BE", alpha=0.5, label='Wind',ec='None', zorder = 0)
    
    ax_side.scatter(com_world[0], com_world[2], color='black', marker='+', s=50, label='CoM', zorder=25)
    
    # Visualizza la forza totale
    ax_side.arrow(com_world[0], com_world[2], F_tot_vis[0]*0.05, F_tot_vis[2]*0.05, 
                  width=0.05, fc='green', ec='None', alpha=0.5, label='Total Force', zorder=0)
    
    ax_side.text(com_world[0] + F_tot_vis[0]*0.05 + 0.2, com_world[2] + F_tot_vis[2]*0.041, 
                 f"{norm_F_tot:.1f} N", color="green", fontweight='bold')

    ax_side.set_title(f"XZ")
    ax_side.grid(True, alpha=0.3); ax_side.axis('equal')
    ax_side.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_static_comparison()

