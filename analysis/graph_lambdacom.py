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

def compute_aerodynamic_equilibrium(p, v_wind_world):
    phi, theta, psi = 0.0, 0.0, 0.0
    for _ in range(15): 
        f_wind = get_wind_force_at_attitude(p, v_wind_world, (phi, theta, psi))
        m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)
        F_tot = np.array([0.0, 0.0, -m_tot * p.g]) + f_wind
        F_req = -F_tot 
        theta = np.arctan2(F_req[0], abs(F_req[2]))
        phi = np.arctan2(-F_req[1], np.sqrt(F_req[0]**2 + F_req[2]**2))
    return phi, theta, f_wind

# ---------------------------------------------------------
# UTILITY FUNCTIONS FOR PLOTTING
# ---------------------------------------------------------

def draw_payload_geometry(ax, p, view='XY', att=(0,0,0), alpha=1.0, 
                        color="#FF9100", att_color="black", zorder=1, style='-'):
    
    phi, theta, psi = att
    R = physics.get_rotation_matrix(phi, theta, psi) 
    edge_rgba = (0.5, 0.5, 0.5, 0.2) 
    att_vecs_rotated = R @ p.attach_vecs
    shape = getattr(p, 'payload_shape', 'box')

    if view == 'XY':   v_idx, idx_x, idx_y, sign = 2, 0, 1, 1 
    elif view == 'XZ': v_idx, idx_x, idx_y, sign = 1, 0, 2, -1 
    elif view == 'YZ': v_idx, idx_x, idx_y, sign = 0, 1, 2, 1 
    else:              v_idx, idx_x, idx_y, sign = 2, 0, 1, 1

    if shape == 'sphere':
        radius = getattr(p, 'R_disk', 0.5)
        circle_patch = mpatches.Circle((0,0), radius, color=color, alpha=alpha*0.3, zorder=zorder)
        phi_angs = np.linspace(0, 2*np.pi, 30)
        theta_angs = np.linspace(-np.pi/2, np.pi/2, 15)
        t = np.linspace(0, 2*np.pi, 50)
        c1 = np.vstack([radius*np.cos(t), radius*np.sin(t), np.zeros_like(t)])
        c2 = np.vstack([radius*np.cos(t), np.zeros_like(t), radius*np.sin(t)])
        c3 = np.vstack([np.zeros_like(t), radius*np.cos(t), radius*np.sin(t)])
        c1_w = R @ c1
        c2_w = R @ c2
        c3_w = R @ c3
        fill_t = np.linspace(0, 2*np.pi, 50)
        fill_x = radius * np.cos(fill_t)
        fill_y = radius * np.sin(fill_t)
        ax.fill(fill_x, fill_y, facecolor=color, alpha=alpha, zorder=zorder, edgecolor='none')
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
    if view == 'XY':
        x, y = positions[0, :], positions[1, :]
        cx, cy = np.mean(x), np.mean(y)
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
        if view == 'XZ': ix, iy = 0, 2
        else:            ix, iy = 1, 2 
        
        x_drones = positions[ix, :]
        y_drones = positions[iy, :]
        num_uavs = positions.shape[1]
        x_pts = np.append(x_drones, x_drones[0])
        y_pts = np.append(y_drones, y_drones[0])
        t = np.linspace(0, 1, num_uavs + 1)
        t_fine = np.linspace(0, 1, 150)
        spline_x = make_interp_spline(t, x_pts, k=3, bc_type='periodic')(t_fine)
        spline_y = make_interp_spline(t, y_pts, k=3, bc_type='periodic')(t_fine)
        ax.plot(spline_x, spline_y, color=color, linestyle=style, 
                linewidth=1.8, alpha=alpha, zorder=zorder)

def configure_axis_common(ax, limit, title):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal', 'box') 
    if limit:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

def plot_xz_view(ax, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                 phi_target, theta_target, v_wind_vec, F_wind_actual, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                 centroid_pos, thrust_vec, limit=None):
    
    ix, iy, iz = 0, 2, 1  # In XZ view: x=0, y=2 (Z in world), z=1 (depth is Y in world)
    view_name = ''
    
    # 1. RIGID (Commentato per nascondere la traccia grigia)
    # draw_payload_geometry(ax, p_base, view='XZ', att=(0,0,0), style=':', alpha=0.1, color='gray', zorder=1)
    # draw_perimeter(ax, pos_ref, 'gray', 0.2, style=':', zorder=1, view='XZ')
    # for i in range(p_base.N):
    #     ax.plot([att_vecs_rigid[ix, i], pos_ref[ix, i]], [att_vecs_rigid[iy, i], pos_ref[iy, i]], 
    #             color='gray', ls=':', alpha=0.1, zorder=1)

    # 2. ADAPTIVE
    draw_payload_geometry(ax, p_base, view='XZ', att=(phi_target, theta_target, 0.0), alpha=1.0, zorder=PAYLOAD_ZORDER)
    draw_perimeter(ax, pos_adapt, 'red', 0.5, style='--', zorder=PAYLOAD_ZORDER, view='XZ')
    for i in range(p_base.N):
        depth_val = att_vecs_adapt[iz, i]
        is_front = (depth_val < 0)
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
    FORCE_SCALE = 0.03
    
    m_tot = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
    F_weight = np.array([0.0, 0.0, -m_tot * p_base.g])

    ## THRUST
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
#
    ## FORZA INERZIALE
    #f_in_2d = np.array([F_inertial[ix], F_inertial[iy]])
    #norm_in_3d = np.linalg.norm(F_inertial)
    #if norm_in_3d > 1e-6:
    #    dx_in = f_in_2d[0] * FORCE_SCALE
    #    dy_in = f_in_2d[1] * FORCE_SCALE
    #    ax.arrow(com_2d[0], com_2d[1], dx_in, dy_in,
    #             width=limit*0.015 if limit else 0.05, 
    #             head_width=limit*0.04 if limit else 0.15, 
    #             fc='purple', ec='None', alpha=0.9, zorder=21,
    #             length_includes_head=True)
    #    ax.text(com_2d[0] + dx_in, com_2d[1] + dy_in, 
    #            f" {norm_in_3d:.1f} N", color='purple', fontsize=8, 
    #            fontweight='bold', ha='left', va='top', zorder=30)
#
    # VENTO
    w_vec_2d = np.array([v_wind_vec[ix], v_wind_vec[iy]])
    wind_speed = np.linalg.norm(v_wind_vec)
    
    if wind_speed > 0.1:
        # Direzione 2D della freccia sul piano XZ
        direction = w_vec_2d / np.linalg.norm(w_vec_2d) if np.linalg.norm(w_vec_2d) > 1e-6 else np.array([1.0, 0.0])
        
        start_x = -2.0  # Partenza fissa a x = -2 come richiesto
        start_z = com_2d[1] # Altezza allineata al CoM per coerenza visiva
        
        # Freccia proporzionale alla scala del grafico
        arrow_len = limit * 0.25 if limit else 1.0
        dx_w = direction[0] * arrow_len
        dz_w = direction[1] * arrow_len
        
        ax.arrow(start_x, start_z, dx_w, dz_w,
                 width=limit*0.015 if limit else 0.05, 
                 head_width=limit*0.04 if limit else 0.15,
                 color="#0092BE", alpha=0.6, ec='None', zorder=0,
                 length_includes_head=True)
                 
        ax.text(start_x + dx_w/2, start_z + (limit*0.05 if limit else 0.15), 
                f"{wind_speed:.1f} m/s", color='#0092BE', fontsize=9, 
                fontweight='bold', ha='center', va='bottom', zorder=30)
    #    
    ## FORZA PESO
    #fg_2d = np.array([F_weight[ix], F_weight[iy]])
    #norm_fg_3d = np.linalg.norm(F_weight)
    #if norm_fg_3d > 1e-6:
    #    dx_g = fg_2d[0] * FORCE_SCALE
    #    dy_g = fg_2d[1] * FORCE_SCALE
    #    ax.arrow(com_2d[0], com_2d[1], dx_g, dy_g,
    #             width=limit*0.015 if limit else 0.05, 
    #             head_width=limit*0.04 if limit else 0.15,
    #             color="gray", alpha=0.9, ec='None', zorder=19,
    #             length_includes_head=True)
    #    ax.text(com_2d[0] + dx_g, com_2d[1] + dy_g, 
    #            f" {norm_fg_3d:.1f} N", color='gray', fontsize=8, 
    #            fontweight='bold', ha='left', va='top', zorder=30)
#
    ## FORZA TOTALE
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
#
    ax.scatter(com_2d[0], com_2d[1], color='black', marker='*', s=20, zorder=25)
    ax.scatter(c_y, c_z, color='black', marker='+', s=20, zorder=26)

    configure_axis_common(ax, limit, view_name)
    if limit: ax.set_ylim(-limit + 1.5, limit + 1.5)

# ---------------------------------------------------------
# NUOVA FUNZIONE COMPARATIVA CoM CRESCENTE (PIANO XZ)
# ---------------------------------------------------------

def generate_com_comparison_xz():
    p_base = SysParams()

    # Impostazione masse
    p_base.m_payload = 1.5 
    p_base.m_liquid = 1.5 

    p_base.uav_offsets, p_base.attach_vecs, p_base.geo_radius = formation.compute_geometry(p_base)
    
    # Valori di CoM Offset crescenti (spostamento lungo l'asse X)
    com_offsets = [
        np.array([0.0, 0.0, 0.0]),  # 1° grafico: CoM centrato
        np.array([0.2, 0.0, 0.0]),  # 2° grafico: CoM sbilanciato di 20 cm
        np.array([0.4, 0.0, 0.0])   # 3° grafico: CoM fortemente sbilanciato di 40 cm
    ]
    
    # Vento fisso a zero per isolare l'effetto del CoM
    v_wind_vec = np.array([0.0, 0.0, 0.0])
    
    # Creazione figura: 1 riga, 3 colonne
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    state = {
        'pay_pos': np.zeros(3), 'pay_att': np.zeros(3), 'pay_vel': np.zeros(3),
        'pay_omega': np.zeros(3), 'uav_vel': np.zeros((3, p_base.N))
    }
    
    for i, com_offset in enumerate(com_offsets):
        ax_xz = axes[i]
        
        # IMPOSTIAMO IL CoM PER QUESTA ITERAZIONE
        p_base.CoM_offset = com_offset
        
        wind_mag = np.linalg.norm(v_wind_vec)
        M_wind_twist = np.array([0.0, 0.0, 0])
        acc_req = np.array([0.0, -0.0, 0.0])

        m_tot = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
        F_inertial = - m_tot * acc_req

        F_wind_rigid = get_wind_force_at_attitude(p_base, v_wind_vec, (0,0,0))
        phi_eq, theta_eq, F_wind_eq_calc = compute_aerodynamic_equilibrium(p_base, v_wind_vec)
        
        # 1. Calcolo Formazione Rigida (riferimento grigio - ora nascosta)
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

        # 2. Calcolo Formazione Adattiva
        p_adapt = copy.deepcopy(p_base)
        p_adapt.theta_ref = 30.0      
        p_adapt.w_T = 1.0
        p_adapt.w_ref = 500.0  
        p_adapt.w_smooth = 200.0 
        p_adapt.w_barrier = 10.0  
        p_adapt.w_resid_m = 1000
        p_adapt.w_resid_f = 3000
        p_adapt.w_cond = 10
        p_adapt.F_ref = 15.0
        p_adapt.k_limit = 0.7 
        p_adapt.k_safe = 0.1   
        p_adapt.k_safe2 = 0.5 

        # IMPOSTAZIONE LAMBDA (CoM e Static attivi per compensare)
        p_adapt.lambda_shape = 0.0  
        p_adapt.lambda_aero = 0.0 
        p_adapt.lambda_static = 1.0 
        p_adapt.lambda_CoM = 1.0
        p_adapt.lambda_twist = 0.0
        
        p_adapt.yaw_force_min = 1e9  
        p_adapt.yaw_force_max = 1e9 + 10.0

        phi_aero = phi_eq * p_adapt.lambda_aero
        theta_aero = theta_eq * p_adapt.lambda_aero
        
        # Calcolo dell'assetto dovuto allo sbilanciamento del CoM
        phi_com = np.arctan(p_base.CoM_offset[1] / p_base.L)
        theta_com = -np.arctan(p_base.CoM_offset[0] / p_base.L)
        
        # L'assetto target finale è la somma dei contributi attivi
        phi_target = phi_aero + (phi_com * p_adapt.lambda_CoM)
        theta_target = theta_aero + (theta_com * p_adapt.lambda_CoM)

        F_wind_actual = get_wind_force_at_attitude(p_adapt, v_wind_vec, (phi_target, theta_target, 0.0))
        state_adapt = copy.deepcopy(state)
        state_adapt['pay_att'] = np.array([phi_target, theta_target, 0.0])

        pos_adapt, _, _, ff_forces_adapt = formation.compute_optimal_formation(
            p_adapt, state_adapt, acc_req, np.zeros(3), 0.0, 
            force_attitude=(phi_target, theta_target), 
            F_ext_total=F_wind_actual,
            M_ext_total=M_wind_twist,
            com_offset_body=p_adapt.CoM_offset 
        )

        # Ricalcolo limiti (usando i dati per trovare la Bounding Box)
        all_points = [pos_ref, pos_adapt, p_base.attach_vecs]
        all_coords = np.hstack(all_points)
        max_coord = np.max(np.abs(all_coords))
        global_limit = max_coord * 1.2
        
        com_world = physics.get_rotation_matrix(phi_target, theta_target, 0.0) @ p_base.CoM_offset
        m_tot_pay = p_base.m_payload + getattr(p_base, 'm_liquid', 0.0)
        F_tot_vis = F_wind_actual + np.array([0, 0, -m_tot_pay * p_base.g]) + F_inertial

        att_vecs_rigid = p_base.attach_vecs
        att_vecs_adapt = physics.get_rotation_matrix(phi_target, theta_target, 0.0) @ p_base.attach_vecs
        PAYLOAD_ZORDER = 10

        F_gravity_drones = np.array([0.0, 0.0, -p_base.N * p_base.m_drone * p_base.g])

        if wind_mag > 1e-3:
            wind_dir = v_wind_vec / wind_mag
            f_drag_single = 0.5 * p_base.rho * p_base.Cd_uav * p_base.A_uav * (wind_mag**2)
            F_wind_drones = f_drag_single * wind_dir * p_base.N
        else:
            F_wind_drones = np.zeros(3)

        F_cables_on_payload_sum = np.sum(ff_forces_adapt, axis=1)
        F_inertial_drones = - (p_base.N * p_base.m_drone * acc_req)
        F_thrust_total = F_cables_on_payload_sum - F_gravity_drones - F_wind_drones - F_inertial_drones

        drone_centroid = np.mean(pos_adapt, axis=1)

        # Plot su piano XZ 
        plot_xz_view(ax_xz, p_base, pos_ref, pos_adapt, att_vecs_rigid, att_vecs_adapt, 
                     phi_target, theta_target, v_wind_vec, F_wind_actual, F_tot_vis, F_inertial, com_world, PAYLOAD_ZORDER, 
                     drone_centroid, F_thrust_total, limit=global_limit)
                     
        #ax_xz.set_title(f"CoM Offset (X) = {com_offset[0]:.2f} m", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_com_comparison_xz()