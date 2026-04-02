import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import copy

# Simulazione moduli esterni
try:
    import formation as formation
    from parameters import SysParams
    import physics 
except ImportError:
    print("ATTENZIONE: Moduli 'formation', 'parameters' o 'physics' non trovati.")
    sys.exit(1)

# =============================================================================
# FISICA E AERODINAMICA
# =============================================================================
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
    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0) 
    
    for _ in range(15): 
        f_wind = get_wind_force_at_attitude(p, v_wind_world, (phi, theta, psi))
        F_tot = np.array([0.0, 0.0, -m_tot * p.g]) + f_wind
        F_req = -F_tot 
        theta = np.arctan2(F_req[0], abs(F_req[2]))
        phi = np.arctan2(-F_req[1], np.sqrt(F_req[0]**2 + F_req[2]**2))
    return phi, theta, f_wind

# =============================================================================
# GEOMETRIA 3D
# =============================================================================
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
    phi = np.linspace(0, np.pi, n_faces)
    theta = np.linspace(0, 2*np.pi, n_faces*2)
    phi, theta = np.meshgrid(phi, theta)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    vertices = []
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
    theta = np.linspace(0, 2*np.pi, n)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z_vals = np.full_like(x, z)
    return [list(zip(x, y, z_vals))]

def get_drone_orientation(thrust_vec):
    norm_t = np.linalg.norm(thrust_vec)
    if norm_t < 1e-3: return np.eye(3)
    z_b = thrust_vec / norm_t
    ref = np.array([1, 0, 0])
    if abs(np.dot(z_b, ref)) > 0.9: ref = np.array([0, 1, 0])
    y_b = np.cross(z_b, ref); y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    return np.column_stack((x_b, y_b, z_b))

# =============================================================================
# MAIN RENDERER
# =============================================================================
def generate_3d_static_view():
    
    # ---------------------------------------------------------
    # MODIFICA: Fissiamo N a 4 e iteriamo sulle forme del payload
    # ---------------------------------------------------------
    shapes = ['sphere', 'cylinder', 'box']
    
    # Creazione della Figura unica (adeguata per 3 subplot orizzontali)
    fig = plt.figure(figsize=(18, 6))

    for idx, current_shape in enumerate(shapes):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.view_init(elev=20, azim=45)

        # =====================================================================
        # --- PANNELLO DI CONFIGURAZIONE UTENTE ---
        # =====================================================================
        p = SysParams()
        
        # 1. FORMA DEL PAYLOAD E DIMENSIONI
        p.N = 4  # Numero di droni fisso a 4
        p.L = 3.0
        p.theta_ref = 30.0
        p.payload_shape = current_shape  # Usa la forma corrente dell'iterazione
        p.pay_l = 1.8            # Lunghezza (asse X - usato per box)
        p.pay_w = 0.8            # Larghezza (asse Y - usato per box)
        p.pay_h = 0.2            # Altezza (asse Z - usato per box e cylinder)
        p.R_disk = 0.6           # Raggio (usato per cylinder o sphere)
        p.m_payload = 2.5        # Massa solida
        p.m_liquid = 1.5         # Massa liquida (influisce sul CoM/Inerzia)

        # 2. VENTO AMBIENTALE
        v_wind_vec = np.array([0.0, 0.0, 0.0]) # [m/s] Vento inclinato

        # 3. COST FUNCTION WEIGHTS (Pesi Ottimizzatore)
        p.w_T = 1.0           
        p.w_ref = 50.0        
        p.w_barrier = 100.0   
        p.w_cond = 10.0       
        p.w_resid_f = 3000.0  
        p.w_resid_m = 1000.0  
        p.w_smooth = 10.0     

        # 4. LAMBDA DI ADATTAMENTO DELLA FORMAZIONE
        p_adapt = copy.deepcopy(p)
        p_adapt.lambda_shape = 1.0   
        p_adapt.lambda_aero = 0.0    
        p_adapt.lambda_static = 1.0  
        p_adapt.lambda_twist = 1.0   
        # =====================================================================

        # Ricostruiamo la geometria base in base alla forma appena scelta
        p_adapt.uav_offsets, p_adapt.attach_vecs, p_adapt.geo_radius = formation.compute_geometry(p_adapt)
        
        w_mag = np.linalg.norm(v_wind_vec)
        floor_level = getattr(p_adapt, 'floor_z', 0.0)
        pay_hover_pos = np.array([0.0, 0.0, 0.0]) # Quota visiva

        # Calcolo Assetto Ottimale (Equilibrio gravità-vento)
        phi_eq, theta_eq, _ = compute_aerodynamic_equilibrium(p_adapt, v_wind_vec)
        
        phi_target = phi_eq * p_adapt.lambda_aero
        theta_target = theta_eq * p_adapt.lambda_aero
        
        state_adapt = {
            'pay_pos': np.zeros(3), 
            'pay_att': np.array([phi_target, theta_target, 0.0]), 
            'pay_vel': np.zeros(3),
            'pay_omega': np.zeros(3), 
            'uav_vel': np.zeros((3, p_adapt.N))
        }
        
        F_wind_actual = get_wind_force_at_attitude(p_adapt, v_wind_vec, (phi_target, theta_target, 0.0))
        
        # Chiamata all'ottimizzatore
        pos_adapt, _, _, ff_forces_adapt = formation.compute_optimal_formation(
            p_adapt, state_adapt, np.zeros(3), np.zeros(3), 0.0, 
            force_attitude=(phi_target, theta_target), 
            F_ext_total=F_wind_actual,
            com_offset_body=p_adapt.CoM_offset 
        )

        # Traslazione al mondo reale (Quota visiva)
        curr_uavs = pos_adapt + pay_hover_pos[:, np.newaxis]
        R_pay = physics.get_rotation_matrix(phi_target, theta_target, 0.0)
        real_com_pos = pay_hover_pos + (R_pay @ p_adapt.CoM_offset)

        # Calcolo Spinta Motori Reale
        F_gravity_drones = np.array([0.0, 0.0, -p_adapt.m_drone * p_adapt.g])
        if w_mag > 1e-3:
            wind_dir = v_wind_vec / w_mag
            f_drag_single = 0.5 * p_adapt.rho * p_adapt.Cd_uav * p_adapt.A_uav * (w_mag**2)
            F_wind_drone = f_drag_single * wind_dir
        else:
            F_wind_drone = np.zeros(3)

        thrust_vectors = []
        thrust_mags = []
        for i in range(p_adapt.N):
            t_vec = ff_forces_adapt[:, i] - F_gravity_drones - F_wind_drone
            thrust_vectors.append(t_vec)
            thrust_mags.append(np.linalg.norm(t_vec))

        # Geometria Payload Dinamica
        shape_type = getattr(p_adapt, 'payload_shape', 'box')
        pay_h = getattr(p_adapt, 'pay_h', 0.5)
        
        if shape_type == 'sphere':
            R_sphere = getattr(p_adapt, 'R_disk', 0.5)
            base_pay_faces = get_sphere_geometry(R_sphere, n_faces=14)
            pay_color = 'orange'
            r_shadow = R_sphere
        elif shape_type in ['box', 'square', 'rect']:
            L_box = getattr(p_adapt, 'pay_l', 0.8); W_box = getattr(p_adapt, 'pay_w', 0.8)
            base_pay_faces = get_box_geometry(L_box, W_box, pay_h)
            pay_color = 'sandybrown'
            r_shadow = max(L_box, W_box) / 2.0
        else: # cylinder
            R_cyl = getattr(p_adapt, 'R_disk', 0.5)
            base_pay_faces = get_cylinder_geometry(R_cyl, pay_h, n_faces=24)
            pay_color = 'orange'
            r_shadow = R_cyl

        # Mesh Payload
        trans_faces = [[(R_pay @ pt + pay_hover_pos) for pt in face] for face in base_pay_faces]
        ax.add_collection3d(Poly3DCollection(trans_faces, facecolors=pay_color, edgecolors='darkorange', alpha=0.8))

        base_drone_faces = get_drone_geometry(0.15)
        colors = cm.get_cmap('tab10', p_adapt.N)
        
        for i in range(p_adapt.N):
            u_pos = curr_uavs[:, i]
            t_vec = thrust_vectors[i]
            t_mag = thrust_mags[i]
            
            R_uav_mat = get_drone_orientation(t_vec)
            drone_faces = [[(R_uav_mat @ pt + u_pos) for pt in face] for face in base_drone_faces]
            ax.add_collection3d(Poly3DCollection(drone_faces, facecolors=colors(i), edgecolors='k', alpha=1.0, linewidths=0.2))
            
            r_arm = R_pay @ p_adapt.attach_vecs[:, i]
            anchor = pay_hover_pos + r_arm
            
            ax.plot3D([u_pos[0], anchor[0]], [u_pos[1], anchor[1]], [u_pos[2], anchor[2]], 
                      color='black', linewidth=1.8)

        # ---------------------------------------------------------------------
        # ZOOM E COMPRESSIONE 3D
        # ---------------------------------------------------------------------
        # Se si vuole spingere di più lo zoom (per avere immagini più grandi),
        # si può usare un moltiplicatore più piccolo tipo * 0.7 o * 0.8
        w_lim = max(p_adapt.L * 1.1, p_adapt.geo_radius * 1.1, 1.0)
        
        # 1. Definiamo i limiti esatti
        x_range = [-w_lim, w_lim]
        y_range = [-w_lim, w_lim]
        z_range = [pay_hover_pos[2], pay_hover_pos[2] + w_lim]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)

        # 2. Scala 1:1:1
        ax.set_box_aspect((
            x_range[1] - x_range[0], 
            y_range[1] - y_range[0], 
            z_range[1] - z_range[0]
        ))

        # 3. RIMOZIONE TOTALE ASSI E GRIGLIE
        ax.set_axis_off() 

    # Compatta i grafici riempiendo tutto lo spazio
    plt.subplots_adjust(left=-0.1, right=1.1, bottom=-0.2, top=1.2, wspace=-0.3)

    plt.show()

if __name__ == "__main__":
    generate_3d_static_view()