import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
import copy
from matplotlib.patches import FancyArrowPatch

# Simulazione moduli esterni
try:
    import formation as formation
    from parameters import SysParams
    import physics 
except ImportError:
    print("ATTENZIONE: Moduli 'formation', 'parameters' o 'physics' non trovati.")
    sys.exit(1)

def get_wind_force_at_attitude(p, v_wind_world, attitude):
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
        if getattr(p, 'payload_shape', '') == 'sphere':
            pay_proj_area = np.pi * (p.R_disk**2)
        else:
            # Cylinder
            A_side = 2.0 * p.R_disk * p.pay_h
            A_top = np.pi * p.R_disk**2
            sin_tilt = np.sqrt(wind_in_body[0]**2 + wind_in_body[1]**2)
            pay_proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])
        
    f_wind = 0.5 * p.rho * p.Cd_pay * pay_proj_area * (wind_mag**2) * wind_dir
    return f_wind

def compute_force_equilibrium(p, v_wind_world, acc_pay_world):
    """
    Calcola l'assetto di equilibrio allineando l'asse Z del payload 
    con la forza totale richiesta (Gravità + Vento + Inerzia).
    """
    phi, theta, psi = 0.0, 0.0, 0.0
    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)
    
    for _ in range(15): 
        # Calcolo forza del vento all'assetto attuale
        f_wind = get_wind_force_at_attitude(p, v_wind_world, (phi, theta, psi))
        
        # Forza peso (vettore verso il basso)
        f_g = np.array([0.0, 0.0, -m_tot * p.g])
        
        # Forza richiesta ai cavi (F_req = m*a - F_g - F_wind)
        # Deve contrastare gravità e vento E fornire l'accelerazione desiderata
        f_req = m_tot * acc_pay_world - f_g - f_wind
        
        # Calcolo angoli di inclinazione per allinearsi a f_req
        theta = np.arctan2(f_req[0], abs(f_req[2]))
        phi = np.arctan2(-f_req[1], np.sqrt(f_req[0]**2 + f_req[2]**2))
        
    return phi, theta, f_wind

# ---------------------------------------------------------
# UTILITY FUNCTIONS FOR PLOTTING
# ---------------------------------------------------------

def draw_payload_geometry(ax, p, view='XY', att=(0,0,0), alpha=1.0, 
                        color="#FF9100", att_color="black", zorder=1, style='-'): #8B372C #FF9100
    
    phi, theta, psi = att
    R = physics.get_rotation_matrix(phi, theta, psi) 
    edge_rgba = (0.5, 0.5, 0.5, 0.2) 
    att_vecs_rotated = R @ p.attach_vecs
    shape = getattr(p, 'payload_shape', 'box')

    if view == 'XY':   v_idx, idx_x, idx_y, sign = 2, 0, 1, 1 
    elif view == 'XZ': v_idx, idx_x, idx_y, sign = 1, 0, 2, -1 
    elif view == 'YZ': v_idx, idx_x, idx_y, sign = 0, 1, 2, 1 
    else:              v_idx, idx_x, idx_y, sign = 2, 0, 1, 1

    # Logica disegno box/cilindro invariata...
    if shape == 'sphere':
        # --- SPHERE (Wireframe Rings) ---
        radius = getattr(p, 'R_disk', 0.5)
        # 1. Outline (cerchio esterno sempre visibile in proiezione)
        circle_patch = mpatches.Circle((0,0), radius, color=color, alpha=alpha*0.3, zorder=zorder)
        # Nota: il cerchio 2D non ruota, ma fornisce il riempimento di base.
        # Per posizionarlo correttamente dobbiamo capire dove è il centro proiettato.
        # Poiché draw_payload_geometry disegna rispetto all'origine locale (che poi il plot sposta),
        # possiamo aggiungere il patch direttamente.
        
        # Ma per coerenza con il resto del codice che usa ax.fill su vertici ruotati,
        # generiamo dei "meridiani" e "paralleli" e li ruotiamo.
        
        phi_angs = np.linspace(0, 2*np.pi, 30)
        theta_angs = np.linspace(-np.pi/2, np.pi/2, 15)
        
        # Generiamo 3 cerchi ortogonali per dare il senso della rotazione
        t = np.linspace(0, 2*np.pi, 50)
        # Cerchio XY
        c1 = np.vstack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)])
        # Cerchio XZ
        c2 = np.vstack([radius*np.cos(t), np.zeros_like(t), radius*np.sin(t)])
        # Cerchio YZ
        c3 = np.vstack([np.zeros_like(t), radius*np.cos(t), radius*np.sin(t)])
        
        # Ruotiamo
        c1_w = R @ c1
        c2_w = R @ c2
        c3_w = R @ c3
        
        # Disegno riempimento "hull" approssimato (un cerchio semplice di sfondo)
        # Usiamo tanti punti per fare un poligono pieno che copra i cavi dietro
        fill_t = np.linspace(0, 2*np.pi, 50)
        fill_x = radius * np.cos(fill_t)
        fill_y = radius * np.sin(fill_t)
        # Il fill non ruota (è una sfera!), ma deve essere disegnato sugli assi corretti
        ax.fill(fill_x, fill_y, facecolor=color, alpha=alpha, zorder=zorder, edgecolor='none')
        
        # Disegno wireframe
        ax.plot(c1_w[idx_x], c1_w[idx_y], color='k', alpha=0.3, lw=0.5, zorder=zorder+0.1)
        ax.plot(c2_w[idx_x], c2_w[idx_y], color='k', alpha=0.3, lw=0.5, zorder=zorder+0.1)
        ax.plot(c3_w[idx_x], c3_w[idx_y], color='k', alpha=0.3, lw=0.5, zorder=zorder+0.1)

    elif shape == 'cylinder':
        radius = getattr(p, 'R_disk', 0.1)
        height = getattr(p, 'pay_h', 0.1)
        t = np.linspace(0, 2*np.pi, 60)
        x_c, y_c = radius * np.cos(t), radius * np.sin(t)
        verts_top_world = R @ np.vstack([x_c, y_c, np.full_like(t, height/2)])
        verts_bot_world = R @ np.vstack([x_c, y_c, np.full_like(t, -height/2)])
        depth_top = np.mean(verts_top_world[v_idx, :]) * sign
        depth_bot = np.mean(verts_bot_world[v_idx, :]) * sign
        is_top_front = depth_top > depth_bot
        vx_top, vy_top = verts_top_world[idx_x, :], verts_top_world[idx_y, :]
        vx_bot, vy_bot = verts_bot_world[idx_x, :], verts_bot_world[idx_y, :]
        def _draw_part(vx, vy):
            ax.fill(vx, vy, facecolor=color, alpha=alpha, 
                    edgecolor=edge_rgba, linewidth=0.5, zorder=zorder, linestyle=style)
        all_x = np.concatenate([vx_top, vx_bot])
        all_y = np.concatenate([vy_top, vy_bot])
        points_2d = np.column_stack([all_x, all_y])
        hull = ConvexHull(points_2d)
        hull_x, hull_y = points_2d[hull.vertices, 0], points_2d[hull.vertices, 1]
        if is_top_front:
            _draw_part(vx_bot, vy_bot); _draw_part(hull_x, hull_y); _draw_part(vx_top, vy_top)
        else:
            _draw_part(vx_top, vy_top); _draw_part(hull_x, hull_y); _draw_part(vx_bot, vy_bot)
    else: 
        # --- BOX ---
        lx, ly, h = p.pay_l/2, p.pay_w/2, p.pay_h/2
        corners = np.array([[lx,ly,h], [-lx,ly,h], [-lx,-ly,h], [lx,-ly,h],
                            [lx,ly,-h], [-lx,ly,-h], [-lx,-ly,-h], [lx,-ly,-h]]).T
        corn_w = R @ corners
        faces = [[0,1,2,3], [4,7,6,5], [0,3,7,4], [1,5,6,2], [0,4,5,1], [3,2,6,7]]
        face_depths = []
        for f_idx in faces:
            depth = np.mean(corn_w[v_idx, f_idx]) * sign
            face_depths.append((depth, f_idx))
        face_depths.sort(key=lambda x: x[0])
        for _, f_idx in face_depths:
            ax.fill(corn_w[idx_x, f_idx], corn_w[idx_y, f_idx], 
                    facecolor=color, alpha=alpha, edgecolor=edge_rgba, 
                    linewidth=0.5, zorder=zorder, linestyle=style)

    # Attacchi
    view_dir = sign
    for i in range(att_vecs_rotated.shape[1]):
        pos = att_vecs_rotated[:, i]
        if pos[v_idx] * view_dir > -getattr(p, 'R_disk', 0.1)*0.2:  
            ax.scatter(pos[idx_x], pos[idx_y], c=att_color, alpha=alpha, s=25, marker='.', zorder=zorder+0.1, edgecolors='none')

from scipy.interpolate import make_interp_spline

def draw_perimeter(ax, positions, color, alpha, style='--', zorder=1, view='XY'):
    """
    Applica l'ellisse statistica per la vista XY e la spline interpolante per XZ/YZ.
    """
    if view == 'XY':
        # --- VERSIONE XY (Statistica/Ellipse Patch) ---
        x, y = positions[0, :], positions[1, :]
        cx, cy = np.mean(x), np.mean(y)
        
        # Calcolo covarianza per determinare l'ellisse di distribuzione
        points = np.vstack((x - cx, y - cy))
        cov = np.cov(points, bias=True)
        
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()
        vals, vecs = vals[order], vecs[:, order]
        
        theta_ell = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
        width = 2 * np.sqrt(2 * np.abs(vals[1]))
        height = 2 * np.sqrt(2 * np.abs(vals[0]))
        
        if height < 1e-6: height = width * 0.01

        ell = Ellipse(xy=(cx, cy), width=width, height=height, angle=theta_ell,
                      edgecolor=color, facecolor='none', linestyle=style, 
                      linewidth=1.5, alpha=alpha, zorder=zorder)
        ax.add_patch(ell)
        
    else:
        # --- VERSIONE XZ/YZ (Spline Interpolante) ---
        if view == 'XZ': ix, iy = 0, 2
        else:            ix, iy = 1, 2 # YZ
        
        x_drones = positions[ix, :]
        y_drones = positions[iy, :]
        num_uavs = positions.shape[1]

        # Chiudiamo il cerchio aggiungendo il primo punto alla fine
        x_pts = np.append(x_drones, x_drones[0])
        y_pts = np.append(y_drones, y_drones[0])
        
        # Parametrizzazione da 0 a 1
        t = np.linspace(0, 1, num_uavs + 1)
        t_fine = np.linspace(0, 1, 150)
        
        # Creazione della Spline Cubica Periodica (passa esattamente per i droni)
        spline_x = make_interp_spline(t, x_pts, k=3, bc_type='periodic')(t_fine)
        spline_y = make_interp_spline(t, y_pts, k=3, bc_type='periodic')(t_fine)
        
        ax.plot(spline_x, spline_y, color=color, linestyle=style, 
                linewidth=1.8, alpha=alpha, zorder=zorder)


def draw_side_segment_generic(ax, positions, ix, iy, color, alpha, style, zorder):
    mu = np.mean(positions, axis=1)
    P_centered = positions - mu[:, np.newaxis]
    U, s, Vh = np.linalg.svd(P_centered)
    radii = np.linalg.norm(P_centered[:2, :], axis=0) 
    r_avg = np.mean(radii)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_local = np.vstack([r_avg * np.cos(theta), r_avg * np.sin(theta), np.zeros_like(theta)])
    circle_on_plane = U @ circle_local 
    circle_world = circle_on_plane + mu[:, np.newaxis]
    ax.plot(circle_world[ix, :], circle_world[iy, :], 
            color=color, linestyle=style, linewidth=1.5, alpha=alpha, zorder=zorder)

# ---------------------------------------------------------
# VIEW-SPECIFIC PLOTTING FUNCTIONS (UPDATED)
# ---------------------------------------------------------

def configure_axis_common(ax, limit, title):
    """Configura assi comuni per tutte le viste"""
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.1)
    ax.set_aspect('equal', 'box') # Forza aspetto quadrato
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if limit:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
    # Rimuovi label assi per pulizia, o lasciali se preferisci

def plot_xy_view(ax, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                 phi_target, theta_target, v_wind_vec, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                 centroid_pos, thrust_vec, limit=None, M_ext=None):
    
    ix, iy, iz = 0, 1, 2
    view_name = 'Top View (XY) - Kinematics'
    
    # 1. RIGID
    #draw_payload_geometry(ax, p_base, view='XY', att=(0,0,0), style=':', alpha=0.1, color='gray', zorder=1)
    #draw_perimeter(ax, pos_ref, 'gray', 0.2, style=':', zorder=1, view='XY')
    #for i in range(p_base.N):
    #    ax.plot([att_vecs_rigid[ix, i], pos_ref[ix, i]], [att_vecs_rigid[iy, i], pos_ref[iy, i]], 
    #            color='gray', ls=':', alpha=0.1, zorder=1)

    # 2. ADAPTIVE
    draw_payload_geometry(ax, p_base, view='XY', att=(phi_target, theta_target, 0.0), alpha=1.0, zorder=PAYLOAD_ZORDER)
    draw_perimeter(ax, pos_adapt, 'red', 0.5, style='--', zorder=PAYLOAD_ZORDER, view='XY')
    for i in range(p_base.N):
        depth_val = att_vecs_adapt[iz, i]
        is_front = (depth_val > 0)
        cable_zorder = PAYLOAD_ZORDER + 1 if is_front else PAYLOAD_ZORDER - 1
        ax.plot([att_vecs_adapt[ix, i], pos_adapt[ix, i]], [att_vecs_adapt[iy, i], pos_adapt[iy, i]], 
                color='black', zorder=cable_zorder)
        ax.scatter(pos_adapt[ix, i], pos_adapt[iy, i], c='black', s=50, marker='h', zorder=cable_zorder)

    c_y, c_z = centroid_pos[ix], centroid_pos[iy]

    ax.scatter(c_y, c_z, color='black', marker='+', s=20, zorder=26)

    # Vettori

    w_vec_2d = np.array([v_wind_vec[0], v_wind_vec[1]])
    if np.linalg.norm(w_vec_2d) > 0.1:
        direction = w_vec_2d / np.linalg.norm(w_vec_2d)
        start_pt = -direction * (limit * 0.8 if limit else 4)
        ax.arrow(start_pt[0], start_pt[1], direction[0], direction[1],
                 width=limit*0.025 if limit else 0.1, color="#0092BE", alpha=0.6, ec='black', zorder=0)
    
        wind_speed = np.linalg.norm(v_wind_vec)
        ax.text(start_pt[0]*1.06 -0.7, start_pt[1]*1.06, 
                f"{wind_speed:.1f} m/s", 
                color='#0092BE', fontsize=9, fontweight='bold', 
                ha='left', va='center_baseline', zorder=30)

    com_2d = np.array([com_world[ix], com_world[iy]])
    ax.scatter(com_2d[0], com_2d[1], color='black', marker='*', s=20, zorder=25)

    if M_ext is not None and abs(M_ext[2]) > 0.1:
        from matplotlib.patches import FancyArrowPatch
        # Disegna una freccia curva per indicare la rotazione sul piano XY
        sign = np.sign(M_ext[2])
        r_arc = limit * 0.5 if limit else 1.0
        # Arco di rotazione intorno al CoM
        arc = FancyArrowPatch(
            posA=(com_2d[0] + r_arc - 0.5, com_2d[1]- 0.5), 
            posB=(com_2d[0]- 0.5, com_2d[1] + sign * r_arc- 0.5),
            connectionstyle=f"arc3,rad={sign * 0.5}",
            arrowstyle="Simple,head_width=5,head_length=8",
            color='#0092BE', lw=3, zorder=0, label="Torque Z")
        ax.add_patch(arc)
        ax.text(com_2d[0] + r_arc- 1.0 , com_2d[1] + sign*r_arc- 0.5, f" {M_ext[2]:.1f}Nm", color='#0092BE', fontweight='bold', fontsize = 8)


    f_in_2d = np.array([F_inertial[ix], F_inertial[iy]])
    norm_in_2d = np.linalg.norm(f_in_2d)

    if norm_in_2d > 1e-6:
        in_unit = f_in_2d / norm_in_2d
        # Scaliamo la lunghezza per mantenerla proporzionata al grafico
        fixed_length_in = limit * 0.25 if limit else 0.8

        ax.arrow(com_2d[0], com_2d[1], 
                 in_unit[0] * fixed_length_in, in_unit[1] * fixed_length_in,
                 width=limit*0.02 if limit else 0.05, 
                 head_width=limit*0.06 if limit else 0.15, 
                 fc='purple', ec='black', alpha=0.6, zorder=21)
        
        # Testo del valore (Accelerazione in XY: m/s²)
        norm_in_3d = np.linalg.norm(F_inertial)
        m_tot = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
        acc_3d = norm_in_3d / m_tot  # a = F/m
        
        tip_x_in = com_2d[0] + in_unit[0] * fixed_length_in *1.5
        tip_y_in = com_2d[1] + in_unit[1] * fixed_length_in *1.5
        
        ax.text(tip_x_in, tip_y_in+0.1 , 
                f" {acc_3d:.1f} m/s²", 
                color='purple', fontsize=8, fontweight='bold', 
                ha='left', va='top', zorder=30)
    
    configure_axis_common(ax, limit, view_name)
    

def plot_xz_view(ax, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                 phi_target, theta_target, v_wind_vec, F_wind_actual, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                 centroid_pos, thrust_vec, limit=None, ff_forces=None):
    
    ix, iy, iz = 0, 2, 1
    view_name = 'Frontal View (XZ) - D\'Alembert\'s balance'
    
    # 1. RIGID
    #draw_payload_geometry(ax, p_base, view='XZ', att=(0,0,0), style=':', alpha=0.1, color='gray', zorder=1)
    #draw_perimeter(ax, pos_ref, 'gray', 0.2, style=':', zorder=1, view='XZ')
    #for i in range(p_base.N):
    #    ax.plot([att_vecs_rigid[ix, i], pos_ref[ix, i]], [att_vecs_rigid[iy, i], pos_ref[iy, i]], 
    #            color='gray', ls=':', alpha=0.1, zorder=1)

    # 2. ADAPTIVE
    FORCE_SCALE = 0.05
    draw_payload_geometry(ax, p_base, view='XZ', att=(phi_target, theta_target, 0.0), alpha=1.0, zorder=PAYLOAD_ZORDER)
    draw_perimeter(ax, pos_adapt, 'red', 0.5, style='--', zorder=PAYLOAD_ZORDER, view='XZ')
    m_tot = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
    for i in range(p_base.N):
        depth_val = att_vecs_adapt[iz, i]
        is_front = (depth_val < 0)
        cable_zorder = PAYLOAD_ZORDER + 1 if is_front else PAYLOAD_ZORDER - 1
        
        # Disegno Cavi e Droni (esistente)
        ax.plot([att_vecs_adapt[ix, i], pos_adapt[ix, i]], [att_vecs_adapt[iy, i], pos_adapt[iy, i]], 
                color='black', zorder=cable_zorder)
        ax.scatter(pos_adapt[ix, i], pos_adapt[iy, i], c='black', s=50, marker='h', zorder=cable_zorder)

        # --- AGGIUNTA: Freccia Forza Peso Drone ---
        # Calcolo della forza peso del singolo drone (F = m * g)
        f_g_drone = p_base.m_drone * p_base.g
        dy_g = -f_g_drone * FORCE_SCALE*1.5
        
        # Disegno della freccia verso il basso (asse Z del mondo, che è iy nel plot)
        ax.arrow(pos_adapt[ix, i], pos_adapt[iy, i], 
                 0, dy_g,
                 width=limit*0.01 if limit else 0.02, 
                 head_width=limit*0.03 if limit else 0.07,
                 color="#4C45B4", alpha=0.4, zorder=cable_zorder,
                 length_includes_head=True)

        # --- AGGIUNTA: Valore numerico della forza ---
        # Posizioniamo il testo leggermente sotto la punta della freccia
        ax.text(pos_adapt[ix, i], pos_adapt[iy, i] + dy_g - (limit*0.05 if limit else 0.1), 
                f"{f_g_drone:.2f}N", 
                color='#4C45B4', 
                fontsize=7, 
                fontweight='bold', alpha=0.4,
                ha='center', 
                va='top', 
                zorder=30)
        
        if ff_forces is not None:
            # 1. Forza peso drone
            f_g_drone = np.array([0, 0, -p_base.m_drone * p_base.g])
            
            # 2. Forza vento drone
            v_w_mag = np.linalg.norm(v_wind_vec)
            f_w_drone = (0.5 * p_base.rho * p_base.Cd_uav * p_base.A_uav * (v_w_mag**2) * (v_wind_vec/v_w_mag)) if v_w_mag > 1e-3 else np.zeros(3)
            
            # 3. Forza inerziale drone (quota parte dell'inerzia totale)
            f_in_drone = F_inertial * (p_base.m_drone / m_tot)
            
            # 4. Calcolo Thrust individuale: T_i = F_cavo_i - F_g_uav - F_w_uav - F_in_uav
            # (ff_forces[:, i] è la forza che il cavo i esercita sul payload)
            t_vec_i = ff_forces[:, i] - f_g_drone - f_w_drone - f_in_drone
            
            # Proiezione XZ (ix=0, iy=2)
            dx_t = t_vec_i[ix] * FORCE_SCALE * 0.8 
            dy_t = t_vec_i[iy] * FORCE_SCALE * 0.8
            
            # Disegno Freccia Thrust (Crimson)
            ax.arrow(pos_adapt[ix, i], pos_adapt[iy, i], 
                     dx_t, dy_t,
                     width=limit*0.01 if limit else 0.02, 
                     head_width=limit*0.03 if limit else 0.07,
                     color="crimson", alpha=0.4, zorder=cable_zorder + 1,
                     length_includes_head=True)

            # Valore numerico della spinta
            t_mag_i = np.linalg.norm(t_vec_i)
            ax.text(pos_adapt[ix, i] + dx_t-0.1, pos_adapt[iy, i] + dy_t + (limit*0.05 if limit else 0.1), 
                    f"{t_mag_i:.1f}N", 
                    color='crimson', alpha=0.4, fontsize=7, fontweight='bold',
                    ha='center', va='bottom', zorder=35)
        
        

    R_pay = physics.get_rotation_matrix(phi_target, theta_target, 0.0)
    z_local_world = R_pay @ np.array([0.0, 0.0, 1.0])
    
    line_len = (limit * 1.5) if limit else 5.0
    pt1 = com_world - z_local_world * line_len
    pt2 = com_world + z_local_world * line_len
    
    ax.plot([pt1[ix], pt2[ix]], [pt1[iy], pt2[iy]], 
            color='gray', linestyle=':', alpha=0.2, linewidth=1.5, zorder=PAYLOAD_ZORDER-1)

    
    # (Opzionale) Scrivi il valore numerico sopra la linea
    # -------------------------------------------------------------
    # DISEGNO QUOTA Δx PERFETTAMENTE ALLINEATA (LINEA + MARKER)
    # -------------------------------------------------------------
    # 1. Matrice di rotazione e assi locali
    R_pay = physics.get_rotation_matrix(phi_target, theta_target, 0.0)
    
    # Asse X locale (direzione della linea di quota)
    x_local = R_pay @ np.array([1.0, 0.0, 0.0])
    dir_x_2d = np.array([x_local[ix], x_local[iy]])
    dir_x_2d /= np.linalg.norm(dir_x_2d)
    
    # Asse Z locale (direzione delle stanghette/marker)
    z_local = R_pay @ np.array([0.0, 0.0, 1.0])
    dir_z_2d = np.array([z_local[ix], z_local[iy]])
    dir_z_2d /= np.linalg.norm(dir_z_2d)
    
    # 2. Calcolo distanza proiettata
    drone_centroid_world = np.array([np.mean(pos_adapt[0, :]), np.mean(pos_adapt[1, :]), np.mean(pos_adapt[2, :])])
    diff_vec = drone_centroid_world - com_world
    dist_x_local = np.dot(diff_vec, x_local)
    
    # 3. Posizionamento linea di quota (sollevata sopra i droni)
    offset_quota = dir_z_2d * (limit * 1.2 if limit else 1.0)
    ref_pt_line = np.array([com_world[ix], com_world[iy]]) + offset_quota
    
    pt1 = ref_pt_line
    pt2 = ref_pt_line + dir_x_2d * dist_x_local
    
    # 4. Disegno della linea principale
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='grey', lw=1.0, zorder=30)
    
    # 5. Disegno manuale delle stanghette (marker inclinati)
    m_len = (limit * 0.02 if limit else 0.08) # Lunghezza totale della stanghetta
    for pt in [pt1, pt2]:
        s_start = pt - dir_z_2d * (m_len / 2)
        s_end = pt + dir_z_2d * (m_len / 2)
        ax.plot([s_start[0], s_end[0]], [s_start[1], s_end[1]], color='grey', lw=1.0, zorder=30)
    
    # 6. Testo ruotato e posizionato
    angle_deg = np.degrees(np.arctan2(dir_x_2d[1], dir_x_2d[0]))
    if angle_deg > 90: angle_deg -= 180
    elif angle_deg < -90: angle_deg += 180
    
    text_pt = (pt1 + pt2) / 2.0 + dir_z_2d * (limit * 0.03 if limit else 0.1)
    ax.text(text_pt[0], text_pt[1], 
            f"Δx: {abs(dist_x_local):.2f}m", color='grey', alpha=0.8, 
            ha='center', va='bottom', fontsize=8, fontweight='bold', 
            rotation=angle_deg, rotation_mode='anchor')

    c_y, c_z = centroid_pos[ix], centroid_pos[iy]

    # Vettori
    com_2d = np.array([com_world[ix], com_world[iy]])
    
    # ---------------------------------------------------------
    # RICALCOLO FORZE (Per garantire coerenza matematica esatta)
    # ---------------------------------------------------------
    m_tot = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
    F_weight = np.array([0.0, 0.0, -m_tot * p_base.g])
    F_tot_vis = F_wind_actual + F_weight + F_inertial  # Somma vettoriale pura

    ## THRUST (Senza testo, alpha=0.3)
    #t_2d = np.array([thrust_vec[ix], thrust_vec[iy]])
    #norm_t_3d = np.linalg.norm(thrust_vec)
    #if norm_t_3d > 1e-3:
    #    dx_t = t_2d[0] * FORCE_SCALE * 0.5 
    #    dy_t = t_2d[1] * FORCE_SCALE * 0.5
    #    ax.arrow(c_y, c_z, dx_t, dy_t,
    #             width=limit*0.025 if limit else 0.05, 
    #             head_width=limit*0.06 if limit else 0.15,
    #             fc='crimson', ec="None", alpha=0.3, zorder=25,
    #             length_includes_head=True)

    # FORZA INERZIALE (Con testo, alpha=0.9)
    f_in_2d = np.array([F_inertial[ix], F_inertial[iy]])
    norm_in_3d = np.linalg.norm(F_inertial)
    if norm_in_3d > 1e-6:
        dx_in = f_in_2d[0] * FORCE_SCALE * 6
        dy_in = f_in_2d[1] * FORCE_SCALE * 6
        ax.arrow(com_2d[0], com_2d[1], dx_in, dy_in,
                 width=limit*0.015 if limit else 0.05, 
                 head_width=limit*0.04 if limit else 0.15, 
                 fc='purple', ec='None', alpha=0.9, zorder=21,
                 length_includes_head=True)
        ax.text(com_2d[0] + dx_in, com_2d[1] + dy_in, 
                f" {norm_in_3d:.1f} N", color='purple', fontsize=8, 
                fontweight='bold', ha='right', va='bottom', zorder=30)

    # VENTO (Con testo, alpha=0.9)
    fw_2d = np.array([F_wind_actual[ix], F_wind_actual[iy]])
    norm_fw_3d = np.linalg.norm(F_wind_actual)
    if norm_fw_3d > 1e-6:
        dx_w = fw_2d[0] * FORCE_SCALE * 7.5
        dy_w = fw_2d[1] * FORCE_SCALE * 7.5
        ax.arrow(com_2d[0], com_2d[1], dx_w, dy_w,
                 width=limit*0.015 if limit else 0.05, 
                 head_width=limit*0.04 if limit else 0.15,
                 color="#0092BE", alpha=0.9, ec='None', zorder=20,
                 length_includes_head=True)
        ax.text(com_2d[0] + dx_w, com_2d[1] + dy_w, 
                f" {norm_fw_3d:.1f} N", color='#0092BE', fontsize=8, 
                fontweight='bold', ha='left', va='bottom', zorder=30)
        
    # FORZA PESO (Con testo, alpha=0.9)
    fg_2d = np.array([F_weight[ix], F_weight[iy]])
    norm_fg_3d = np.linalg.norm(F_weight)
    if norm_fg_3d > 1e-6:
        dx_g = fg_2d[0] * FORCE_SCALE * 0.9
        dy_g = fg_2d[1] * FORCE_SCALE * 0.9
        ax.arrow(com_2d[0], com_2d[1], dx_g, dy_g,
                 width=limit*0.015 if limit else 0.05, 
                 head_width=limit*0.04 if limit else 0.15,
                 color="grey", alpha=0.9, ec='None', zorder=19,
                 length_includes_head=True)
        ax.text(com_2d[0] + dx_g, com_2d[1] + dy_g, 
                f" {norm_fg_3d:.1f} N", color="grey", fontsize=8, 
                fontweight='bold', ha='left', va='top', zorder=30)

    ## FORZA TOTALE (Senza testo, alpha=0.3)
    #f_tot_2d = np.array([F_tot_vis[ix], F_tot_vis[iy]])
    #norm_ft_3d = np.linalg.norm(F_tot_vis)
    #if norm_ft_3d > 1e-6:
    #    dx_tot = f_tot_2d[0] * FORCE_SCALE
    #    dy_tot = f_tot_2d[1] * FORCE_SCALE
    #    ax.arrow(com_2d[0], com_2d[1], dx_tot, dy_tot,
    #             width=limit*0.025 if limit else 0.05, 
    #             head_width=limit*0.06 if limit else 0.15, 
    #             fc='green', alpha=0.3, ec='None', zorder=20,
    #             length_includes_head=True)

    ax.scatter(com_2d[0], com_2d[1], color='black', marker='*', s=20, zorder=25)
    ax.scatter(c_y, c_z, color='black', marker='+', s=20, zorder=26)

    configure_axis_common(ax, limit, view_name)
    if limit: ax.set_ylim(-limit + 1.5, limit + 1.5)
    
    
    ax.scatter(com_2d[0], com_2d[1], color='black', marker='*', s=20, zorder=25)
    ax.scatter(c_y, c_z, color='black', marker='+', s=20, zorder=26)

    configure_axis_common(ax, limit, view_name)
    if limit: ax.set_ylim(-limit + 1.5, limit + 1.5)

def plot_yz_view(ax, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                 phi_target, theta_target, v_wind_vec, F_wind_actual, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                 centroid_pos, thrust_vec, limit=None): # <--- Argomenti aggiunti
    
    ix, iy, iz = 1, 2, 0
    view_name = 'Lateral View (YZ) - Resultant Forces'
    
    ## 1. RIGID
    #draw_payload_geometry(ax, p_base, view='YZ', att=(0,0,0), style=':', alpha=0.1, color='gray', zorder=1)
    #draw_perimeter(ax, pos_ref, 'gray', 0.2, style=':', zorder=1, view='YZ')
    #for i in range(p_base.N):
    #    ax.plot([att_vecs_rigid[ix, i], pos_ref[ix, i]], [att_vecs_rigid[iy, i], pos_ref[iy, i]], 
    #            color='gray', ls=':', alpha=0.1, zorder=1)

    # 2. ADAPTIVE
    draw_payload_geometry(ax, p_base, view='YZ', att=(phi_target, theta_target, 0.0), alpha=1.0, zorder=PAYLOAD_ZORDER)
    draw_perimeter(ax, pos_adapt, 'red', 0.5, style='--', zorder=PAYLOAD_ZORDER, view='YZ')
    for i in range(p_base.N):
        depth_val = att_vecs_adapt[iz, i]
        is_front = (depth_val > 0)
        cable_zorder = PAYLOAD_ZORDER + 1 if is_front else PAYLOAD_ZORDER - 1
        ax.plot([att_vecs_adapt[ix, i], pos_adapt[ix, i]], [att_vecs_adapt[iy, i], pos_adapt[iy, i]], 
                color='black', zorder=cable_zorder)
        ax.scatter(pos_adapt[ix, i], pos_adapt[iy, i], c='black', s=50, marker='h', zorder=cable_zorder)


    R_pay = physics.get_rotation_matrix(phi_target, theta_target, 0.0)
    z_local_world = R_pay @ np.array([0.0, 0.0, 1.0])
    
    line_len = (limit * 1.5) if limit else 5.0
    pt1 = com_world - z_local_world * line_len
    pt2 = com_world + z_local_world * line_len
    
    ax.plot([pt1[ix], pt2[ix]], [pt1[iy], pt2[iy]], 
            color='gray', linestyle=':', alpha=0.2, linewidth=1.5, zorder=PAYLOAD_ZORDER-1)
    
    # -------------------------------------------------------------
    # DISEGNO QUOTA Δy PERFETTAMENTE ALLINEATA (VISTA LATERALE YZ)
    # -------------------------------------------------------------
    # 1. Matrice di rotazione e assi locali
    R_pay = physics.get_rotation_matrix(phi_target, theta_target, 0.0)
    
    # Asse Y locale (direzione longitudinale nella vista laterale)
    y_local_vec = R_pay @ np.array([0.0, 1.0, 0.0])
    # Mappatura YZ: orizzontale = Y mondo (index 1), verticale = Z mondo (index 2)
    dir_y_2d = np.array([y_local_vec[1], y_local_vec[2]]) 
    dir_y_2d /= np.linalg.norm(dir_y_2d)
    
    # Asse Z locale (per stanghette e sollevamento quota)
    z_local_vec = R_pay @ np.array([0.0, 0.0, 1.0])
    dir_z_2d = np.array([z_local_vec[1], z_local_vec[2]])
    dir_z_2d /= np.linalg.norm(dir_z_2d)
    
    # 2. Calcolo distanza proiettata lungo l'asse Y del payload
    drone_centroid_world = np.array([np.mean(pos_adapt[0, :]), np.mean(pos_adapt[1, :]), np.mean(pos_adapt[2, :])])
    diff_vec = drone_centroid_world - com_world
    dist_y_local = np.dot(diff_vec, y_local_vec)
    
    # 3. Posizionamento linea di quota (sollevata lungo lo Z locale)
    # Usiamo lo stesso moltiplicatore (es. 1.2) per coerenza con la vista XZ
    offset_quota = dir_z_2d * (limit * 1.2 if limit else 1.0)
    ref_pt_line = np.array([com_world[1], com_world[2]]) + offset_quota
    
    pt1 = ref_pt_line
    pt2 = ref_pt_line + dir_y_2d * dist_y_local
    
    # 4. Disegno della linea principale
    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='grey', lw=1.0, zorder=30)
    
    # 5. Disegno stanghette perpendicolari (marker inclinati)
    m_len = (limit * 0.02 if limit else 0.08)
    for pt in [pt1, pt2]:
        s_start = pt - dir_z_2d * (m_len / 2)
        s_end = pt + dir_z_2d * (m_len / 2)
        ax.plot([s_start[0], s_end[0]], [s_start[1], s_end[1]], color='grey', lw=1.0, zorder=30)
    
    # 6. Testo ruotato Δy
    angle_deg = -np.degrees(np.arctan2(dir_y_2d[1], dir_y_2d[0]))
    
    # Raddrizza il testo solo se è fisicamente sottosopra (oltre i 90 gradi)
    if angle_deg > 90: angle_deg -= 180
    elif angle_deg < -90: angle_deg += 180
    
    text_pt = (pt1 + pt2) / 2.0 + dir_z_2d * (limit * 0.03 if limit else 0.1)
    ax.text(text_pt[0], text_pt[1], 
            f"Δy: {abs(dist_y_local):.2f}m", color='grey', alpha=0.8, 
            ha='center', va='bottom', fontsize=8, fontweight='bold', 
            rotation=angle_deg, rotation_mode='anchor')
    
    c_y, c_z = centroid_pos[ix], centroid_pos[iy]
    
    # Vettori
    com_2d = np.array([com_world[ix], com_world[iy]])
    FORCE_SCALE = 0.03
    
    # ---------------------------------------------------------
    # RICALCOLO FORZE (Per garantire coerenza matematica esatta)
    # ---------------------------------------------------------
    m_tot = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
    F_weight = np.array([0.0, 0.0, -m_tot * p_base.g])
    F_tot_vis = F_wind_actual + F_weight + F_inertial  # Somma vettoriale pura

    FORCE_SCALE_DRONE = 0.03
    
    # Calcolo del peso totale dei droni (N * m_drone * g)
    W_drones_total = p_base.N * p_base.m_drone * p_base.g
    dy_g_total = -W_drones_total * FORCE_SCALE_DRONE
    
    # In vista YZ: centroid_pos[1] è la Y, centroid_pos[2] è la Z
    # Disegno della freccia verso il basso
    ax.arrow(centroid_pos[1], centroid_pos[2], 
             0, dy_g_total,
             width=limit*0.010 if limit else 0.03, 
             head_width=limit*0.04 if limit else 0.1,
             color="#4C45B4", alpha=0.8, zorder=25,
             length_includes_head=True)

    # Inserimento del valore numerico sotto la punta della freccia
    ax.text(centroid_pos[1]-0.5, centroid_pos[2] + dy_g_total - (limit*0.06 if limit else 0.15), 
            f"{W_drones_total:.2f} N", 
            color='#4C45B4', fontsize=8, fontweight='bold',
            ha='center', va='top', zorder=30)

    # THRUST (Con testo, alpha=0.9)
    t_2d = np.array([thrust_vec[ix], thrust_vec[iy]])
    norm_t_3d = np.linalg.norm(thrust_vec)
    if norm_t_3d > 1e-3:
        dx_t = t_2d[0] * FORCE_SCALE * 0.5 
        dy_t = t_2d[1] * FORCE_SCALE * 0.5
        
        ax.arrow(c_y, c_z, dx_t, dy_t,
                 width=limit*0.015 if limit else 0.05, 
                 head_width=limit*0.04 if limit else 0.15,
                 fc='crimson', ec="None", alpha=0.9, zorder=25,
                 length_includes_head=True)
        
        ax.text(c_y + dx_t , c_z + dy_t, 
                f" {norm_t_3d:.1f} N", 
                color='crimson', fontsize=8, fontweight='bold', 
                ha='left', va='bottom', zorder=30)

    # FORZA TOTALE (Con testo, alpha=0.9)                   
    f_tot_2d = np.array([F_tot_vis[ix], F_tot_vis[iy]])
    norm_ft_3d = np.linalg.norm(F_tot_vis)
    if norm_ft_3d > 1e-6:
        dx_tot = f_tot_2d[0] * FORCE_SCALE
        dy_tot = f_tot_2d[1] * FORCE_SCALE
        ax.arrow(com_2d[0], com_2d[1], dx_tot, dy_tot,
                 width=limit*0.015 if limit else 0.05, 
                 head_width=limit*0.04 if limit else 0.15, 
                 fc='green', alpha=0.9, ec='None', zorder=20,
                 length_includes_head=True)

        ax.text(com_2d[0] + dx_tot, com_2d[1] + dy_tot, 
                f" {norm_ft_3d:.1f} N", color='green', fontsize=9, 
                fontweight='bold', ha='left', va='top', zorder=30)
    
    
    ax.scatter(com_2d[0], com_2d[1], color='black', marker='*', s=20, zorder=25)
    ax.scatter(c_y, c_z, color='black', marker='+', s=20, zorder=26)

    configure_axis_common(ax, limit, view_name)
    ax.invert_xaxis()
    if limit: ax.set_ylim(-limit + 1.5, limit + 1.5)

# ---------------------------------------------------------
# MAIN COMPARISON GENERATOR
# ---------------------------------------------------------

def generate_static_comparison():
    # ... setup parametri e calcoli (INVARIATI) ...
    p_base = SysParams()

    p_base.m_payload = 3 
    p_base.m_liquid = 0 

    p_base.CoM_offset = np.array([0.1, 0.3, 0.0])

    p_base.m_drone = 1.0

    p_base.uav_offsets, p_base.attach_vecs, p_base.geo_radius = formation.compute_geometry(p_base)
    
    v_wind_vec = np.array([3.0, 2.0, 0.0])
    acc_req = np.array([2.0, -1.5, 0.0])
    M_wind_twist = np.array([0.0, 0.0, 5])
    

    m_tot = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
    F_inertial = - m_tot * acc_req
    v_wind_ms = np.linalg.norm(v_wind_vec)
    dir_norm = v_wind_vec / v_wind_ms

    F_wind_rigid = get_wind_force_at_attitude(p_base, v_wind_vec, (0,0,0))
    phi_eq, theta_eq, F_wind_eq_calc = compute_force_equilibrium(p_base, v_wind_vec, acc_req)
    
    state = {
        'pay_pos': np.zeros(3), 'pay_att': np.zeros(3), 'pay_vel': np.zeros(3),
        'pay_omega': np.zeros(3), 'uav_vel': np.zeros((3, p_base.N))
    }
    
    p_rigid = copy.deepcopy(p_base)
    p_rigid.lambda_shape = 0.0
    p_rigid.lambda_aero = 0.0
    p_rigid.lambda_static = 0.0
    p_rigid.lambda_twist = 0
    state_rigid = copy.deepcopy(state)
    state_rigid['pay_att'] = np.zeros(3)
    
    pos_ref, _, _, ff_forces_ref = formation.compute_optimal_formation(
        p_rigid, state_rigid, np.zeros(3), np.zeros(3), 0.0, 
        force_attitude=(0.0, 0.0), F_ext_total=F_wind_rigid, M_ext_total=M_wind_twist,
        com_offset_body=p_rigid.CoM_offset
    )

    def compute_power_index(ff_forces, p_sys, v_wind):
        p_total = 0
        # Peso e vento per singolo drone
        f_g_drone = np.array([0, 0, -p_sys.m_drone * p_sys.g])
        v_mag = np.linalg.norm(v_wind)
        f_w_drone = 0.5 * p_sys.rho * p_sys.Cd_uav * p_sys.A_uav * (v_mag**2) * (v_wind/v_mag) if v_mag > 1e-3 else np.zeros(3)
        
        for i in range(p_sys.N):
            # La spinta T deve bilanciare la forza del cavo (ff_forces), il peso e il vento
            thrust_vec = ff_forces[:, i] - f_g_drone - f_w_drone
            thrust_mag = np.linalg.norm(thrust_vec)
            p_total += thrust_mag**1.5 # Indice di potenza
        return p_total


    p_adapt = copy.deepcopy(p_base)

    p_adapt.theta_ref = 30.0      

    p_adapt.w_T = 1.0
    p_adapt.w_ref = 200.0  #+
    p_adapt.w_smooth = 200.0 #+
    p_adapt.w_barrier = 0.0  #- picchi alti
    p_adapt.w_resid_m = 1000
    p_adapt.w_resid_f = 3000
    p_adapt.w_cond = 0

    p_adapt.F_ref = 15.0

    p_adapt.k_limit = 0.7 
    p_adapt.k_safe = 0.1   
    p_adapt.k_safe2 = 0.5 

    p_adapt.lambda_shape = 1  
    p_adapt.lambda_aero =  1 
    p_adapt.lambda_static = 1 
    p_adapt.lambda_CoM = 1
    p_adapt.lambda_twist = 1 
    p_adapt.yaw_force_min = 1e9  
    p_adapt.yaw_force_max = 1e9 + 10.0

    phi_aero = phi_eq * p_adapt.lambda_aero
    theta_aero = theta_eq * p_adapt.lambda_aero
    
    # 2. Contributo CoM (Sbilanciamento statico)
    # Calcoliamo l'angolo necessario affinché il CoM penda sotto i droni
    phi_com = np.arctan(p_base.CoM_offset[1] / p_base.L)
    theta_com = -np.arctan(p_base.CoM_offset[0] / p_base.L)
    
    # 3. Assetto finale (Somma pesata dai lambda)
    phi_target = phi_aero + (phi_com * p_adapt.lambda_CoM)
    theta_target = theta_aero + (theta_com * p_adapt.lambda_CoM)

    F_wind_actual = get_wind_force_at_attitude(p_adapt, v_wind_vec, (phi_target, theta_target, 0.0))
    state_adapt = copy.deepcopy(state)
    state_adapt['pay_att'] = np.array([phi_target, theta_target, 0.0])


    # MODIFICA: Recuperiamo ff_forces_adapt (le forze reali esercitate dai cavi sul payload)
    pos_adapt, _, _, ff_forces_adapt = formation.compute_optimal_formation(
        p_adapt, state_adapt, acc_req, np.zeros(3), 0.0, 
        force_attitude=(phi_target, theta_target), 
        F_ext_total=F_wind_actual,
        M_ext_total=M_wind_twist,
        com_offset_body=p_adapt.CoM_offset 
    )

    p_idx_nom = compute_power_index(ff_forces_ref, p_base, v_wind_vec)
    p_idx_adapt = compute_power_index(ff_forces_adapt, p_base, v_wind_vec)
    risparmio = (1 - p_idx_adapt/p_idx_nom) * 100

    # ---------------------------------------------------------
    # CALCOLO LIMITI GLOBALI (NUOVO)
    # ---------------------------------------------------------
    # Raccogliamo tutti i punti di interesse per calcolare la Bounding Box
    all_points = []
    all_points.append(pos_ref)      # Posizioni droni rigidi
    all_points.append(pos_adapt)    # Posizioni droni adattivi
    all_points.append(p_base.attach_vecs) # Attacchi payload
    
    # Stack tutto in un unico array (3, N_points)
    all_coords = np.hstack(all_points)
    
    # Troviamo il massimo valore assoluto su X, Y, Z
    max_coord = np.max(np.abs(all_coords))
    
    # Aggiungiamo un margine del 20%
    global_limit = max_coord * 1.2
    
    # ---------------------------------------------------------
    # PLOT SETUP
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5)) # Più quadrato per aiutare
    
    # Configurazione layout tecnico "Europeo/Americano" unificato
    ax_xz = axes[0, 0]  # Front
    ax_yz = axes[0, 1]  # Side
    ax_xy = axes[1, 0]  # Top
    ax_leg = axes[1, 1] # Legend

    com_world = physics.get_rotation_matrix(phi_target, theta_target, 0.0) @ p_base.CoM_offset
    
    # --- 3. CORREZIONE FORZA TOTALE GLOBALE ---
    m_tot_pay = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
    F_tot_vis = F_wind_actual + np.array([0, 0, -m_tot_pay * p_base.g]) + F_inertial
    norm_F_tot = np.linalg.norm(F_tot_vis)
    att_vecs_rigid = p_base.attach_vecs
    att_vecs_adapt = physics.get_rotation_matrix(phi_target, theta_target, 0.0) @ p_base.attach_vecs
    PAYLOAD_ZORDER = 10

    F_gravity_drones = np.array([0.0, 0.0, -p_base.N * p_base.m_drone * p_base.g])

    v_wind_mag = np.linalg.norm(v_wind_vec)
    if v_wind_mag > 1e-3:
        wind_dir = v_wind_vec / v_wind_mag
        f_drag_single = 0.5 * p_base.rho * p_base.Cd_uav * p_base.A_uav * (v_wind_mag**2)
        F_wind_drones = f_drag_single * wind_dir * p_base.N
    else:
        F_wind_drones = np.zeros(3)

    # MODIFICA: Calcolo Thrust Reale basato sull'output dell'ottimizzatore
    # Equilibrio Drone: T + F_g_drone + F_w_drone + F_cavo_su_drone = 0
    # F_cavo_su_drone = - F_cavo_su_payload (ff_forces_adapt)
    # T = F_cavo_su_payload - F_g_drone - F_w_drone

    F_cables_on_payload_sum = np.sum(ff_forces_adapt, axis=1)
    
    # --- 1. CORREZIONE SPINTA DRONI ---
    # I droni devono accelerare assieme al payload, quindi serve la loro forza d'inerzia
    F_inertial_drones = - (p_base.N * p_base.m_drone * acc_req)
    F_thrust_total = F_cables_on_payload_sum - F_gravity_drones - F_wind_drones - F_inertial_drones

    # ---------------------------------------------------------
    # DIAGNOSTICA DI HOVERING / DRIFTING
    # ---------------------------------------------------------
    # --- 2. CORREZIONE EQUILIBRIO DI D'ALEMBERT ---
    m_tot_pay = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
    F_weight_payload = np.array([0.0, 0.0, -m_tot_pay * p_base.g])
    
    # La somma netta ora include l'Inerzia per confermare che l'accelerazione è garantita
    F_net_dalembert = F_cables_on_payload_sum + F_wind_actual + F_weight_payload + F_inertial
    residual_force_mag = np.linalg.norm(F_net_dalembert)
    
    # Soglia di tolleranza (es. 1 Newton)
    TOLERANCE = 1.0 
    
    if residual_force_mag < TOLERANCE:
        # Il sistema è in equilibrio
        hover_status = "STATIC HOVERING"
        status_color = "green"
        drift_acc = 0.0
        drift_text = "Stable"
    else:
        # Il sistema non ce la fa e si sta muovendo
        hover_status = "DRIFTING (WIND BLOWN)"
        status_color = "red"
        
        # Calcola l'accelerazione di deriva (F = m*a -> a = F/m)
        # Usiamo la massa totale approssimata (Payload + Droni) per una stima realistica
        total_mass = p_base.m_payload + (p_base.N * p_base.m_drone)
        drift_acc = residual_force_mag / total_mass
        
        drift_text = f"Drift Acc: {drift_acc:.2f} m/s²"

    print(f"\n--- DIAGNOSTICA SISTEMA ---")
    print(f"Stato: {hover_status}")
    print(f"Forza Netta Residua: {residual_force_mag:.2f} N")
    if residual_force_mag >= TOLERANCE:
        print(f"ATTENZIONE: I motori sono saturi. Il sistema sta accelerando via.")
    print(f"---------------------------\n")

    

    drone_centroid = np.mean(pos_adapt, axis=1)

    # Passiamo global_limit a tutte le funzioni
    plot_xy_view(ax_xy, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                 phi_target, theta_target, v_wind_vec, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                 drone_centroid, F_thrust_total, limit=global_limit, M_ext=M_wind_twist)
    
    plot_xz_view(ax_xz, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                 phi_target, theta_target, v_wind_vec, F_wind_actual, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                 drone_centroid, F_thrust_total, limit=global_limit, ff_forces=ff_forces_adapt)
    
    plot_yz_view(ax_yz, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                 phi_target, theta_target, v_wind_vec, F_wind_actual, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                 drone_centroid, F_thrust_total, limit=global_limit)

    # ---------------------------------------------------------
    # LEGENDA E LAYOUT FINALE
    # ---------------------------------------------------------
    ax_leg.axis('off')
    
    # 1. Definizione degli elementi grafici (Proxy Artists)
    line_adapt = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, 
                               label=f'Adaptive Formation') #(λs={p_adapt.lambda_static})
    patch_pay = mpatches.Patch(facecolor='#FF7B00', edgecolor='gray', 
                               label=f'Payload') #(m={p_base.m_payload}Kg)
    line_cable = mlines.Line2D([], [], color='black', linewidth=2, label='Cables')
    dot_drone = mlines.Line2D([], [], color='black', marker='h', linestyle='None', 
                              markersize=8, label='UAV')
    
    # --- NUOVE AGGIUNTE RICHIESTE ---
    # Forza peso del payload (usando il colore rosso scuro #8B372C definito nei plot)
    arrow_weight_pay = mpatches.Patch(facecolor='gray', edgecolor='None', 
                                      label='Payload Weight Force')
    # Forza peso dei droni (colore grigio con alpha 0.6)
    arrow_weight_uav = mpatches.Patch(facecolor='#4C45B4', edgecolor='None', alpha=0.9, 
                                      label='UAVs Weight Force')
    # Linea tratteggiata passante per il CoM (asse centrale locale)
    line_com_axis = mlines.Line2D([], [], color='gray', linestyle=':', linewidth=1.5, 
                                  alpha=0.5, label='CoM Central Axis')
    # --------------------------------

    arrow_wind = mpatches.Patch(facecolor='#0092BE', edgecolor='None', label='Payload Aerodynamic Force')
    arrow_inertial = mpatches.Patch(facecolor='purple', edgecolor='None', label='Payload Inertial Force')
    arrow_wind_v = mpatches.Patch(facecolor='#0092BE', edgecolor='black', alpha = 0.6, label='Wind Velocity')
    arrow_inertial_v = mpatches.Patch(facecolor='purple', edgecolor='black', alpha = 0.6, label='Payload Acceleration')
    arrow_force = mpatches.Patch(facecolor='green', edgecolor='None', label='Payload Total Force')
    arrow_thrust = mpatches.Patch(facecolor='crimson', edgecolor='None', label='Thrust Force')
    
    marker_com = mlines.Line2D([], [], color='black', marker='*', linestyle='None', 
                               markersize=8, markeredgewidth=2, label='Center of Mass')
    marker_cof = mlines.Line2D([], [], color='black', marker='+', linestyle='None', 
                               markersize=8, markeredgewidth=2, label='Centroid of Formation')

    # 2. Creazione della lista ordinata dei handles
    handles_left = [
        line_adapt, 
        line_com_axis,
        line_cable,
        patch_pay,  
        dot_drone,
        marker_com, 
        marker_cof,
        arrow_wind_v,
        arrow_inertial_v,
    ]

    handles_right = [
        arrow_weight_uav,
        arrow_weight_pay, 
        
        arrow_wind, 
        arrow_inertial,   
        arrow_force, 
        arrow_thrust 
    ]


    handles = handles_left + handles_right

    # 3. Disegno della legenda
    ax_leg.legend(handles=handles, 
                  loc='center', 
                  ncol=1,                # Due colonne
                  columnspacing=1.5,     # Spazio tra le colonne
                  handletextpad=0.7,     # Spazio tra simbolo e testo
                  fontsize=11.89, 
                  frameon=True,
                  borderpad=1.2)

    

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_static_comparison()