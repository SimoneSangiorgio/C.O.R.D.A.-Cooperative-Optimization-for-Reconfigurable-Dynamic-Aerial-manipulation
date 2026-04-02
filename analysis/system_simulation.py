import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import make_interp_spline

def get_cylinder_geometry(r, h, n_faces=24):
    theta = np.linspace(0, 2*np.pi, n_faces+1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    vertices = []
    bottom = [np.array([x[i], y[i], -h/2]) for i in range(n_faces)]
    top = [np.array([x[i], y[i], h/2]) for i in range(n_faces)]
    vertices.append(bottom); vertices.append(top)
    for i in range(n_faces):
        p1 = np.array([x[i], y[i], -h/2]); p2 = np.array([x[(i+1)%n_faces], y[(i+1)%n_faces], -h/2])
        p3 = np.array([x[(i+1)%n_faces], y[(i+1)%n_faces], h/2]); p4 = np.array([x[i], y[i], h/2])
        vertices.append([p1, p2, p3, p4])
    return vertices

def get_box_geometry(L, W, H):
    x = [-L/2, L/2]; y = [-W/2, W/2]; z = [-H/2, H/2]
    v = []
    for k in z:
        for j in y:
            for i in x:
                v.append(np.array([i, j, k]))
    v = np.array(v)
    faces = [
        [v[0], v[1], v[3], v[2]], [v[4], v[5], v[7], v[6]], 
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]], 
        [v[0], v[2], v[6], v[4]], [v[1], v[3], v[7], v[5]]
    ]
    return faces

def get_sphere_geometry(r, n_faces=12):
    # Genera una sfera usando coordinate sferiche
    phi = np.linspace(0, np.pi, n_faces)
    theta = np.linspace(0, 2*np.pi, n_faces*2)
    phi, theta = np.meshgrid(phi, theta)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    vertices = []
    # Crea le facce (quadrilateri) per Poly3DCollection
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            p1 = np.array([x[i, j], y[i, j], z[i, j]])
            p2 = np.array([x[i+1, j], y[i+1, j], z[i+1, j]])
            p3 = np.array([x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]])
            p4 = np.array([x[i, j+1], y[i, j+1], z[i, j+1]])
            vertices.append([p1, p2, p3, p4])
    return vertices

def get_drone_geometry(r):
    return get_cylinder_geometry(r, h=0.03, n_faces=10)

def get_shadow_circle_verts(cx, cy, z, r, n=20):
    """Genera vertici per un cerchio piatto sul piano Z."""
    theta = np.linspace(0, 2*np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z_vals = np.full_like(x, z)
    return [list(zip(x, y, z_vals))]

def get_rotation_matrix(phi, theta, psi):
    cph, sph = np.cos(phi), np.sin(phi); cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        [cps*cth,  cps*sth*sph - sps*cph,  cps*sth*cph + sps*sph],
        [sps*cth,  sps*sth*sph + cps*cph,  sps*sth*cph - cps*sph],
        [-sth,     cth*sph,                cth*cph]
    ])

def get_drone_orientation(thrust_vec):
    norm_t = np.linalg.norm(thrust_vec)
    if norm_t < 1e-3: return np.eye(3)
    z_b = thrust_vec / norm_t
    ref = np.array([1, 0, 0])
    if abs(np.dot(z_b, ref)) > 0.9: ref = np.array([0, 1, 0])
    y_b = np.cross(z_b, ref); y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    # Return Rotation Matrix: [x_b, y_b, z_b]
    return np.column_stack((x_b, y_b, z_b))

def extract_roll_pitch(R):
    pitch = np.arcsin(-R[2, 0]) 
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.degrees(roll), np.degrees(pitch)

# --- ANIMATION ---

def animate(t, x, p, theta_log=None, thrust_log=None):
    print("Preparing 3D Animation (Full HUD & Flat Shadows)...")
    
    n_uav_pos = 3 * p.N
    uav_pos_flat = x[:n_uav_pos, :]
    pay_pos_flat = x[n_uav_pos : n_uav_pos+3, :]
    pay_att_flat = x[n_uav_pos+3 : n_uav_pos+6, :]
    uav_vel_flat = x[n_uav_pos+3+3 : n_uav_pos+3+3+3*p.N, :]

    idx_pay_vel = n_uav_pos + 3 + 3 + 3*p.N
    pay_vel_flat = x[idx_pay_vel : idx_pay_vel+3, :]
    
    t_plot = t
    
    theta_opt_vals = np.zeros_like(t_plot)
    if theta_log is not None and len(theta_log) > 0:
        log_t = np.array([val[0] for val in theta_log])
        log_th = np.array([val[1] for val in theta_log])
        theta_opt_vals = np.interp(t_plot, log_t, log_th)
    
    if len(t) > 1:
        dt_ms = (t[1] - t[0]) * 1000
        dt_sec = t[1] - t[0]
    else:
        dt_ms = 20; dt_sec = 0.02

    pay_acc_flat = np.gradient(pay_vel_flat, dt_sec, axis=1)
    acc_global = np.gradient(uav_vel_flat, dt_sec, axis=1)
    
    raw_thrusts = []
    
    # Vettore gravità (colonna 3x1 per broadcasting)
    g_vec = np.array([[0], [0], [-9.81]]) 

    if thrust_log is not None:
        # USA I DATI REALI (Magnitude corretta dal PID)
        acc_global = np.gradient(uav_vel_flat, dt_sec, axis=1)
        g_vec = np.array([[0], [0], [-9.81]])
        raw_thrusts = []
        
        for i in range(p.N):
            # Ricostruiamo la direzione basandoci sull'accelerazione cinematica
            idx = i*3
            acc_i = acc_global[idx:idx+3, :]
            dir_vec = p.m_drone * (acc_i - g_vec)
            
            # Normalizziamo la direzione
            mags = np.linalg.norm(dir_vec, axis=0)
            mags[mags < 1e-4] = 1.0
            dir_norm = dir_vec / mags
            
            # Applichiamo la magnitudo reale dal log
            real_mag = thrust_log[i, :]
            raw_thrusts.append(dir_norm * real_mag)
            
    else:
        # FALLBACK (Vecchio metodo euristico impreciso)
        pay_acc_flat = np.gradient(pay_vel_flat, dt_sec, axis=1)
        acc_global = np.gradient(uav_vel_flat, dt_sec, axis=1)
        raw_thrusts = []
        g_vec = np.array([[0], [0], [-9.81]]) 

        for i in range(p.N):
            idx = i*3
            acc_i = acc_global[idx:idx+3, :] 
            f_drone_req = p.m_drone * (acc_i - g_vec)
            f_pay_share = (p.m_payload / p.N) * (pay_acc_flat - g_vec)
            raw_thrusts.append(f_drone_req + f_pay_share)

    smooth_window = 50
    kernel = np.ones(smooth_window) / smooth_window
    
    smoothed_thrusts = []
    for i in range(p.N):
        t_smooth = np.zeros_like(raw_thrusts[i])
        for ax_j in range(3): 
            t_smooth[ax_j, :] = np.convolve(raw_thrusts[i][ax_j, :], kernel, mode='same')
        smoothed_thrusts.append(t_smooth)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    
    # Griglia ampia (Wireframe)
    grid_w = 200; step = 5
    xx, yy = np.meshgrid(np.arange(-grid_w, grid_w+1, step), np.arange(-grid_w, grid_w+1, step))
    ax.plot_wireframe(xx, yy, np.zeros_like(xx), color='lightgray', alpha=0.5, linewidth=0.8)

    shape_type = getattr(p, 'payload_shape', 'cylinder')
    pay_h = getattr(p, 'pay_h', 0.3)
    if shape_type == 'sphere':
        R_sphere = getattr(p, 'R_disk', 0.5)
        base_pay_faces = get_sphere_geometry(R_sphere, n_faces=14)
        pay_color = 'orange' # Colore distintivo per la sfera
    elif shape_type in ['box', 'square', 'rect']:
        L_box = getattr(p, 'pay_l', 0.8); W_box = getattr(p, 'pay_w', 0.8)
        base_pay_faces = get_box_geometry(L_box, W_box, pay_h)
        pay_color = 'sandybrown'
    else:
        R_cyl = getattr(p, 'R_disk', 0.5)
        base_pay_faces = get_cylinder_geometry(R_cyl, pay_h, n_faces=24)
        pay_color = 'orange'
        
    base_drone_faces = get_drone_geometry(0.15)
    colors = cm.get_cmap('tab10', p.N) 

    trail, = ax.plot([], [], [], ':', color='gray', alpha=0.5)
    ax.plot([p.payload_goal_pos[0]], [p.payload_goal_pos[1]], [p.payload_goal_pos[2]], 'g*', markersize=12, label='Goal')
    
    com_point, = ax.plot([], [], [], 'ko', ms=4, zorder=200, label='Payload CoM')

    anchors_scatter, = ax.plot([], [], [], 'k.', markersize=4, zorder=200)
    cables_lines = [ax.plot([], [], [], '-', lw=1)[0] for _ in range(p.N)]
    formation_ring, = ax.plot([], [], [], color='black', linestyle='--', linewidth=0.8, alpha=0.1)
    
    # --- OMBRE DEI CAVI ---
    cable_shadow_lines = [ax.plot([], [], [], ':', color='black', alpha=0.15, lw=1.5)[0] for _ in range(p.N)]
    
    wind_quiver = None
    title_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, fontweight='bold')
    hud_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes, fontsize=9, family='monospace', verticalalignment='top')
    
    actors = {
        'payload': None, 
        'drones': [None]*p.N, 
        'thrusts': [None]*p.N,
        'pay_shadow': None,
        'drone_shadows': [None]*p.N,
        'ftot_arrow': None
    }

    t_idle = 1
    t_ramp = getattr(p, 'motor_ramp_duration', 1.5)

    def update(frame):
        nonlocal wind_quiver
        
        # 1. TEMPO CORRENTE (Cruciale per Sloshing e Vento)
        curr_time = t_plot[frame]

        # 2. CLEANUP ATTORI PRECEDENTI
        if actors['payload']: 
            try: actors['payload'].remove()
            except ValueError: pass
        if actors['pay_shadow']:
            try: actors['pay_shadow'].remove()
            except ValueError: pass
        for i in range(p.N):
            if actors['drones'][i]: 
                try: actors['drones'][i].remove()
                except ValueError: pass
            if actors['thrusts'][i]: 
                try: actors['thrusts'][i].remove()
                except ValueError: pass 
            if actors['drone_shadows'][i]:
                try: actors['drone_shadows'][i].remove()
                except ValueError: pass
        if wind_quiver:
            try: wind_quiver.remove()
            except ValueError: pass
        if actors['ftot_arrow']:
            try: actors['ftot_arrow'].remove()
            except ValueError: pass

        # 3. STATO PAYLOAD BASE
        curr_p = pay_pos_flat[:, frame]
        curr_att = pay_att_flat[:, frame]
        R_pay = get_rotation_matrix(*curr_att)

        # 4. CALCOLO CoM DINAMICO (SLOSHING)
        if getattr(p, 'enable_sloshing', False):
            amp = getattr(p, 'slosh_amp', 0.3)
            freq = getattr(p, 'slosh_freq', 0.5)
            omega = 2.0 * np.pi * freq
            
            dx = amp * np.sin(omega * curr_time)
            dy = amp * np.cos(omega/ 2.0 * curr_time)
            current_com_offset = np.array([dx, dy, 0.0])
        else:
            current_com_offset = getattr(p, 'CoM_offset', np.zeros(3))

        # Posizione reale del CoM sincronizzata
        real_com_pos = curr_p + R_pay @ current_com_offset

        # ---------------------------------------------------------
        # 5. CALCOLO VENTO (LOGICA SOMMATIVA)
        # ---------------------------------------------------------
        # Base Wind (Fisso o Rotante)
        base_vec = getattr(p, 'initial_wind_vec', np.zeros(3)).copy()
        rot_vel = getattr(p, 'Rot_wind', np.zeros(3))
        rot_mag = np.linalg.norm(rot_vel)
        
        if rot_mag > 1e-6:
            u = rot_vel / rot_mag
            theta = rot_mag * curr_time
            # Formula di Rodrigues
            w_base = (base_vec * np.cos(theta) + 
                      np.cross(u, base_vec) * np.sin(theta) + 
                      u * np.dot(u, base_vec) * (1 - np.cos(theta)))
        else:
            w_base = base_vec

        # Raffiche (Gusts) - Si sommano alla base
        w_gust = np.zeros(3)
        if getattr(p, 'enable_gusts', False):
            for gust in p.gust_schedule:
                t_s = gust['t_start']; t_d = gust['duration']; ramp = gust.get('ramp', 0.5)
                if t_s <= curr_time <= (t_s + t_d):
                    scale = 1.0
                    if curr_time < (t_s + ramp): scale = (curr_time - t_s) / ramp
                    elif curr_time > (t_s + t_d - ramp): scale = ((t_s + t_d) - curr_time) / ramp
                    w_gust += (gust['vec'] * np.clip(scale, 0.0, 1.0))

        # Vento Totale (VELOCITA')
        w_curr_vec = w_base + w_gust
        w_mag = np.linalg.norm(w_curr_vec)

        # ---------------------------------------------------------
        # 6. CALCOLO FORZE E DISEGNO F_TOT
        # ---------------------------------------------------------
        m_tot_vis = p.m_payload + getattr(p, 'm_liquid', 0.0)
        F_gravity = np.array([0, 0, -m_tot_vis * p.g])
        
        # Stima Forza Aerodinamica (per visualizzazione F_TOT)
        # Usiamo l'area Z di riferimento per semplicità nel plot
        # Calcolo FISICO ESATTO della Forza Aerodinamica (Vento Relativo + Proiezione Area 3D)
        v_pay_curr = pay_vel_flat[:, frame] if frame > 0 else np.zeros(3)
        v_rel = v_pay_curr - w_curr_vec
        v_rel_norm = np.linalg.norm(v_rel)
        
        if v_rel_norm > 1e-3:
            v_rel_dir = v_rel / v_rel_norm
            v_rel_body = R_pay.T @ v_rel_dir
            
            if shape_type in ['box', 'rect', 'square']:
                A_x, A_y, A_z = getattr(p, 'pay_w', 0.6) * getattr(p, 'pay_h', 0.2), getattr(p, 'pay_l', 1.4) * getattr(p, 'pay_h', 0.2), getattr(p, 'pay_l', 1.4) * getattr(p, 'pay_w', 0.6)
                proj_area = A_x * abs(v_rel_body[0]) + A_y * abs(v_rel_body[1]) + A_z * abs(v_rel_body[2])
            else:
                A_side = 2.0 * getattr(p, 'R_disk', 0.6) * getattr(p, 'pay_h', 0.2)
                A_top = np.pi * getattr(p, 'R_disk', 0.6)**2
                sin_tilt = np.sqrt(v_rel_body[0]**2 + v_rel_body[1]**2)
                proj_area = A_side * sin_tilt + A_top * abs(v_rel_body[2])
                
            f_wind_vis = -0.5 * p.rho * getattr(p, 'Cd_pay', 1.0) * proj_area * (v_rel_norm**2) * v_rel_dir
        else:
            f_wind_vis = np.zeros(3)
        
        if frame < 2: curr_acc_pay = np.zeros(3)
        else: curr_acc_pay = pay_acc_flat[:, frame]

        F_inertial = -p.m_payload * curr_acc_pay
        F_ext = f_wind_vis + F_inertial
        F_tot_vec = F_gravity + F_ext
        
        f_tot_mag = np.linalg.norm(F_tot_vec)
        if frame > 5 and f_tot_mag > 0.5:
            scale_f = 0.05
            f_vis = F_tot_vec * scale_f
            
            actors['ftot_arrow'] = ax.quiver(
                real_com_pos[0], real_com_pos[1], real_com_pos[2],
                f_vis[0], f_vis[1], f_vis[2],
                color="#39488A", alpha=0.7, lw=1.5, arrow_length_ratio=0.2
            )

        # 7. DISEGNO ELLISSE (Anello Formazione)
        curr_uavs = np.zeros((3, p.N))
        for i in range(p.N):
            idx = i*3
            curr_uavs[:, i] = uav_pos_flat[idx:idx+3, frame]

        if p.N >= 3:
            # Centro geometrico
            center_pts = np.mean(curr_uavs, axis=1)
            # ALZO L'ELLISSE DI 30 CM SOPRA I DRONI
            center_pts[2] += 0.5
            
            vec_A = np.zeros(3)
            vec_B = np.zeros(3)
            for i in range(p.N):
                angle_i = 2 * np.pi * i / p.N
                diff = curr_uavs[:, i] - center_pts
                vec_A += diff * np.cos(angle_i)
                vec_B += diff * np.sin(angle_i)
            
            vec_A *= (2.0 / p.N)
            vec_B *= (2.0 / p.N)
            
            theta_loop = np.linspace(0, 2 * np.pi, 100)
            ring_vals = (center_pts[:, np.newaxis] + 
                         vec_A[:, np.newaxis] * np.cos(theta_loop) + 
                         vec_B[:, np.newaxis] * np.sin(theta_loop))
            
            formation_ring.set_data(ring_vals[0, :], ring_vals[1, :])
            formation_ring.set_3d_properties(ring_vals[2, :])
            
        # 8. ZOOM E CAMERA
        w = 4.5 
        ax.set_xlim(curr_p[0]-w, curr_p[0]+w)
        ax.set_ylim(curr_p[1]-w, curr_p[1]+w)
        ax.set_zlim(curr_p[2]-w, curr_p[2]+w)
        floor_level = getattr(p, 'floor_z', 0.0)

        # 9. OMBRA PAYLOAD
        r_shadow_pay = getattr(p, 'R_disk', 0.5)
        if shape_type in ['box', 'rect', 'square']:
            r_shadow_pay = max(getattr(p, 'pay_l', 0.5), getattr(p, 'pay_w', 0.5)) / 2.0
            
        sh_verts = get_shadow_circle_verts(curr_p[0], curr_p[1], floor_level, r_shadow_pay)
        poly_sh = Poly3DCollection(sh_verts, facecolors='black', alpha=0.2)
        ax.add_collection3d(poly_sh)
        actors['pay_shadow'] = poly_sh
        
        # Disegno Freccia VENTO
        if w_mag > 0.1:
            w_dir = w_curr_vec / w_mag
            w_start = curr_p + np.array([0, 0, 2.0]) 
            w_col = 'navy' if getattr(p, 'enable_gusts', False) else 'deepskyblue'
            wind_quiver = ax.quiver(w_start[0], w_start[1], w_start[2], 
                                     w_dir[0], w_dir[1], w_dir[2], 
                                     length=1.5 + (w_mag * 0.1), color=w_col, 
                                     arrow_length_ratio=0.3, linewidth=2, alpha=0.8)

        # 10. MOTORI E DRONI
        if curr_time < t_idle: motor_scale = 0.0
        elif curr_time < t_idle + t_ramp: motor_scale = (curr_time - t_idle) / t_ramp
        else: motor_scale = 1.0

        trans_faces = [[(R_pay @ pt + curr_p) for pt in face] for face in base_pay_faces]
        poly_pay = Poly3DCollection(trans_faces, facecolors=pay_color, edgecolors='darkorange', alpha=0.8)
        ax.add_collection3d(poly_pay)
        actors['payload'] = poly_pay
        
        # Aggiorna il punto nero nell'animazione (usa real_com_pos calcolato all'inizio)
        com_point.set_data([real_com_pos[0]], [real_com_pos[1]])
        com_point.set_3d_properties([real_com_pos[2]])
        
        anchor_coords = []
        
        # HUD Text setup
        roll_deg, pitch_deg = np.degrees(curr_att[0]), np.degrees(curr_att[1])
        yaw_deg = np.degrees(curr_att[2])
        theta_opt_curr = theta_opt_vals[frame] if len(theta_opt_vals) > frame else 0.0

        psi_curr = curr_att[2]
        c_p, s_p = np.cos(-psi_curr), np.sin(-psi_curr)
        R_inv_yaw = np.array([
            [c_p, -s_p, 0],
            [s_p,  c_p, 0],
            [0,    0,   1]
        ])
        R_abs = R_inv_yaw @ R_pay
        # Estraiamo Roll/Pitch relativi al mondo (assumendo Yaw=0)
        z_axis_world = R_pay[:, 2]
        pay_world_roll = np.degrees(np.arcsin(np.clip(-z_axis_world[1], -1.0, 1.0)))
        pay_world_pitch = np.degrees(np.arctan2(z_axis_world[0], z_axis_world[2]))

        # B. Formation RPY (Geometria reale dei droni)
        # Centro dei droni
        uav_center = np.mean(curr_uavs, axis=1)
        
        # Yaw Formazione (Vettore dal centro al Drone 0)
        v_head = curr_uavs[:, 0] - uav_center
        form_yaw = np.degrees(np.arctan2(v_head[1], v_head[0]))
        
        # Roll/Pitch Formazione (Vettore Normale al piano dei droni via SVD)
        uav_centered = curr_uavs - uav_center[:, None]
        # SVD è il metodo più robusto per trovare la normale di un insieme di punti planari
        u_svd, _, _ = np.linalg.svd(uav_centered)
        normal = u_svd[:, 2] # Il vettore singolare associato al valore più piccolo è la normale
        if normal[2] < 0: normal = -normal # Assicuriamo che punti verso l'alto (Z+)
        
        # Estrazione angoli dalla normale
        form_pitch = np.degrees(np.arcsin(-normal[0]))
        form_roll = np.degrees(np.arctan2(normal[1], normal[2]))
        
        hud_str = f"SYSTEM STATUS\n"
        hud_str += f"Time:    {curr_time:.2f} s\n"
        hud_str += f"Pay RP (World): {pay_world_roll:.1f}, {pay_world_pitch:.1f}\n"
        hud_str += f"Form RPY:      {form_roll:.1f}, {form_pitch:.1f}, {form_yaw:.1f}\n"
        hud_str += f"Theta Opt: {np.degrees(theta_opt_curr):.1f} deg\n"
        hud_str += f"Wind:    {w_mag:.1f} m/s\n" # Corretta unità misura
        hud_str += f"F_TOT:   {f_tot_mag:6.2f} N\n"
        hud_str += "-" * 25 + "\n"
        title_text.set_text(f"TIME: {curr_time:.2f} s")

        for i in range(p.N):
            u_pos = curr_uavs[:, i]
            
            # Ombra Drone
            sh_u_verts = get_shadow_circle_verts(u_pos[0], u_pos[1], floor_level, 0.2)
            poly_u_sh = Poly3DCollection(sh_u_verts, facecolors='black', alpha=0.2)
            ax.add_collection3d(poly_u_sh)
            actors['drone_shadows'][i] = poly_u_sh
            
            # Orientamento e Thrust
            T_vec_smooth = smoothed_thrusts[i][:, frame]
            T_vec_vis = T_vec_smooth * motor_scale
            thrust_mag = np.linalg.norm(T_vec_vis)
            
            R_uav_mat = np.eye(3)
            uav_roll, uav_pitch = 0.0, 0.0
            if np.linalg.norm(T_vec_vis) > 0.1: 
                R_uav_mat = get_drone_orientation(T_vec_vis)
                uav_roll, uav_pitch = extract_roll_pitch(R_uav_mat)
            
            drone_faces = [[(R_uav_mat @ pt + u_pos) for pt in face] for face in base_drone_faces]
            poly_uav = Poly3DCollection(drone_faces, facecolors=colors(i), edgecolors='k', alpha=1.0, linewidths=0.2)
            ax.add_collection3d(poly_uav)
            actors['drones'][i] = poly_uav
            
            # --- SCALING FRECCE THRUST (SQRT) ---
            k_vis = 0.2
            t_len = k_vis * np.sqrt(thrust_mag) # Scala non lineare

            if t_len > 0.05:
                t_dir = T_vec_vis / (thrust_mag + 1e-6) * t_len
                # Colore dinamico (Opzionale, qui messo fisso o scalato)
                # color_intensity = min(1.0, thrust_mag / 40.0) 
                
                q = ax.quiver(u_pos[0], u_pos[1], u_pos[2], t_dir[0], t_dir[1], t_dir[2], 
                              color='red', lw=1.5, arrow_length_ratio=0.25)
                actors['thrusts'][i] = q
            
            # Cavi
            r_arm = R_pay @ p.attach_vecs[:, i]
            anchor = curr_p + r_arm
            anchor_coords.append(anchor)
            dist = np.linalg.norm(u_pos - anchor)
            is_taut = dist >= (p.L * 0.98)
            
            cables_lines[i].set_data([u_pos[0], anchor[0]], [u_pos[1], anchor[1]])
            cables_lines[i].set_3d_properties([u_pos[2], anchor[2]])
            cables_lines[i].set_color('black' if is_taut else 'silver')
            cables_lines[i].set_linewidth(1.8 if is_taut else 1.0)
            
            # Ombra Cavo
            cable_shadow_lines[i].set_data([u_pos[0], anchor[0]], [u_pos[1], anchor[1]])
            cable_shadow_lines[i].set_3d_properties([floor_level, floor_level])
            
            hud_str += f"UAV {i+1}: T:{thrust_mag:.1f}N\n"

        anchors_np = np.array(anchor_coords)
        anchors_scatter.set_data(anchors_np[:, 0], anchors_np[:, 1])
        anchors_scatter.set_3d_properties(anchors_np[:, 2])
        hud_text.set_text(hud_str)

        st = max(0, frame-200)
        trail.set_data(pay_pos_flat[0, st:frame], pay_pos_flat[1, st:frame])
        trail.set_3d_properties(pay_pos_flat[2, st:frame])
        
        return [trail, title_text, com_point, anchors_scatter, hud_text, formation_ring] + cables_lines + cable_shadow_lines

    speed_factor = 10
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=range(0, len(t_plot), speed_factor),
        interval=dt_ms, 
        blit=False
    )
    
    plt.tight_layout()
    plt.show()