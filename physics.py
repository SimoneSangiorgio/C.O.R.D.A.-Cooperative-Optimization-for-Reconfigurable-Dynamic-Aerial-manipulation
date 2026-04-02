import numpy as np

def unpack_state(x, p):
    n_uav_pos = 3 * p.N
    n_pos = n_uav_pos + 3 + 3
    n_uav_vel = 3 * p.N
    n_vel = n_uav_vel + 3 + 3
    
    s = {}
    s['uav_pos'] = x[0:n_uav_pos].reshape((p.N, 3)).T
    s['pay_pos'] = x[n_uav_pos : n_uav_pos+3]
    s['pay_att'] = x[n_uav_pos+3 : n_pos]
    
    s['uav_vel'] = x[n_pos : n_pos+n_uav_vel].reshape((p.N, 3)).T
    s['pay_vel'] = x[n_pos+n_uav_vel : n_pos+n_uav_vel+3]
    s['pay_omega'] = x[n_pos+n_uav_vel+3 : n_pos+n_vel]
    
    idx_int = n_pos+n_vel
    s['int_uav'] = x[idx_int : idx_int+3*p.N].reshape((p.N, 3)).T
    s['int_pay'] = x[idx_int+3*p.N : idx_int+3*p.N+3]
    return s


def get_rotation_matrix(phi, theta, psi):
    """Rotation Matrix ZYX convention: R = R_z(psi) @ R_y(theta) @ R_x(phi)."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    
    R_x = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
    R_y = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    R_z = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
    
    return R_z @ R_y @ R_x


def compute_aerodynamics(v_rel, rho, Cd, Area):
    """
    Aerodynamic drag from relative velocity v_rel = v_body - v_wind.
    F_drag = -0.5 * rho * Cd * Area * ||v_rel||^2 * hat(v_rel)
    """
    v_norm = np.linalg.norm(v_rel)
    if v_norm < 1e-3:
        return np.zeros(3)
    dir_v = v_rel / v_norm
    f_mag = 0.5 * rho * Cd * Area * (v_norm**2)
    return -f_mag * dir_v


def compute_ground_effect(thrust_force_vec, h, prop_r):
    """
    Ground effect thrust enhancement for a UAV near the ground.
    Physically acts only on the vertical (z) thrust component.
    Returns additional force vector due to ground effect.
    """
    if h <= 0.0 or h > (4.0 * prop_r):
        return np.zeros(3)
    
    ratio = prop_r / (4.0 * h)
    if ratio > 0.6:
        ratio = 0.6
    k_ge = 1.0 / (1.0 - ratio**2)

    # Ground effect enhances only the vertical (upward) thrust component
    f_extra = np.zeros(3)
    f_extra[2] = max(0.0, thrust_force_vec[2]) * (k_ge - 1.0)
    return f_extra


def compute_downwash(pos_source, pos_target, thrust_source_mag, p):
    """
    Downwash force from a UAV rotor on a target below it.
    Uses actuator disk theory: v_0 = sqrt(T / (2 * rho * A_disk))
    where A_disk = pi * R_prop^2 (disk area, NOT pi*(2R)^2).
    """
    diff = pos_target - pos_source
    if diff[2] > 0:
        return np.zeros(3)
    
    dist_sq = np.dot(diff, diff)
    if dist_sq > 25.0 or dist_sq < 0.01:
        return np.zeros(3)
    
    dist = np.sqrt(dist_sq)
    down_vec = np.array([0.0, 0.0, -1.0])
    dir_target = diff / dist
    cos_angle = np.dot(down_vec, dir_target)
    
    if cos_angle > np.cos(p.downwash_cone_angle):
        # FIX: area disco = pi * R^2, NON pi * (2R)^2
        A_disk = np.pi * (p.prop_radius**2)
        v_0 = np.sqrt(thrust_source_mag / (2.0 * p.rho * A_disk))
        v_dist = v_0 * (1.0 / (1.0 + dist))
        f_mag = 0.5 * p.rho * p.Cd_pay * p.A_pay_z * (v_dist**2)
        return (f_mag * dir_target).astype(float)
    
    return np.zeros(3)


def compute_payload_aero(v_pay, wind_vec_world, R_pay, p):
    """
    Unified aerodynamic force on payload using the correct relative velocity:
        v_rel = v_payload - v_wind
        F_aero = -0.5 * rho * Cd * A_proj(v_rel) * ||v_rel||^2 * hat(v_rel)

    This correctly accounts for the cross-coupling between payload motion and wind,
    which is lost if the two contributions are computed and summed separately.
    """
    v_rel = v_pay - wind_vec_world
    v_rel_norm = np.linalg.norm(v_rel)

    if v_rel_norm < 1e-3:
        return np.zeros(3)

    v_rel_dir = v_rel / v_rel_norm
    # Project relative wind into body frame to compute orientation-dependent area
    v_rel_body = R_pay.T @ v_rel_dir

    payload_shape = getattr(p, 'payload_shape', 'box')

    if payload_shape in ['box', 'rect', 'square']:
        A_x = p.pay_w * p.pay_h
        A_y = p.pay_l * p.pay_h
        A_z = p.pay_l * p.pay_w
        A_proj = (A_x * abs(v_rel_body[0]) +
                  A_y * abs(v_rel_body[1]) +
                  A_z * abs(v_rel_body[2]))
    elif payload_shape == 'sphere':
        A_proj = np.pi * (p.R_disk**2)
    else:
        # Cylinder
        A_side = 2.0 * p.R_disk * p.pay_h
        A_top  = np.pi * p.R_disk**2
        sin_tilt = np.sqrt(v_rel_body[0]**2 + v_rel_body[1]**2)
        A_proj = A_side * sin_tilt + A_top * abs(v_rel_body[2])

    f_mag = 0.5 * p.rho * p.Cd_pay * A_proj * (v_rel_norm**2)
    return -f_mag * v_rel_dir


def compute_derivatives(t, s, u_acc, p, ctx, debug_forces):
    phi, theta, psi = s['pay_att']
    R_pay = get_rotation_matrix(phi, theta, psi)

    m_tot = p.m_payload + p.m_liquid

    # =========================================================================
    # 1. SLOSHING (Forza + Coppia)
    # =========================================================================
    # e_static: eccentricità strutturale statica del CoM del contenitore rigido
    # rispetto al suo centro geometrico GC, nel body frame.
    e_static = getattr(p, 'CoM_offset', np.zeros(3)).copy()

    # e_slosh: posizione del centroide del liquido rispetto al GC, nel body frame.
    # La sua contribuzione al CoM del SISTEMA è pesata da m_liquid / m_tot.
    e_slosh = np.zeros(3)
    F_slosh_reaction_body = np.zeros(3)

    if getattr(p, 'enable_sloshing', False):
        amp  = getattr(p, 'slosh_amp',  0.3)
        freq = getattr(p, 'slosh_freq', 0.5)
        omega_s = 2.0 * np.pi * freq

        # Posizione e accelerazione del centroide del liquido nel body frame
        e_slosh = np.array([
            amp * np.sin(omega_s * t),
            (amp / 2.0) * np.cos(omega_s * t),
            0.0
        ])
        a_slosh = np.array([
            -amp * (omega_s**2) * np.sin(omega_s * t),
            -(amp / 2.0) * (omega_s**2) * np.cos(omega_s * t),
            0.0
        ])
        # Per Newton III: il contenitore subisce la reazione dell'accelerazione del liquido
        F_slosh_reaction_body = -p.m_liquid * a_slosh

    # CoM del sistema rispetto al GC nel body frame:
    # e_CoM = e_static + (m_liquid / m_tot) * e_slosh
    # FIX: dividere per m_tot, non per m_payload
    e_com_body = e_static + (p.m_liquid / m_tot) * e_slosh

    # =========================================================================
    # 2. FORZE AERODINAMICHE SUL PAYLOAD
    # =========================================================================
    wind_vec_world = p.wind_vel

    # FIX: forza aerodinamica calcolata con velocità relativa unificata
    # v_rel = v_payload - v_wind (include cross-term quadratico correttamente)
    f_aero_pay = compute_payload_aero(s['pay_vel'], wind_vec_world, R_pay, p)

    # Coppia aerodinamica (weather-vane effect): CoP rispetto al CoM corrente
    r_cop_body  = getattr(p, 'CoP', np.zeros(3)) - e_com_body
    r_cop_world = R_pay @ r_cop_body
    tau_wind_aero = np.cross(r_cop_world, f_aero_pay)

    # =========================================================================
    # 3. FORZE CAVI E UAV
    # =========================================================================
    wind_mag = np.linalg.norm(wind_vec_world)
    if wind_mag > 1e-3:
        wind_dir = wind_vec_world / wind_mag
    else:
        wind_dir = np.zeros(3)

    f_cables_sum  = np.zeros(3)
    tau_cables_sum = np.zeros(3)
    f_repulsion   = np.zeros((p.N, 3))
    f_downwash_tot = np.zeros(3)

    # Repulsione inter-agente
    safe_dist = 0.6
    k_rep = 80.0
    for i in range(p.N):
        for j in range(i + 1, p.N):
            diff_ij = s['uav_pos'][:, i] - s['uav_pos'][:, j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < safe_dist and dist_ij > 0.01:
                f_rep_val = k_rep * (safe_dist - dist_ij) * (diff_ij / dist_ij)
                f_repulsion[i, :] += f_rep_val
                f_repulsion[j, :] -= f_rep_val

    d_uav_vel = np.zeros((3, p.N))
    g_vec = np.array([0.0, 0.0, p.g])

    for i in range(p.N):
        r_arm_body  = p.attach_vecs[:, i]
        r_arm_world = R_pay @ r_arm_body
        anchor_pt   = s['pay_pos'] + r_arm_world

        vec_cable = s['uav_pos'][:, i] - anchor_pt
        len_cable = np.linalg.norm(vec_cable)
        dir_cable = vec_cable / max(len_cable, 1e-4)
        delta_L   = len_cable - ctx.current_L_rest[i]

        vel_tan_body = np.cross(s['pay_omega'], r_arm_body)
        v_anchor = s['pay_vel'] + (R_pay @ vel_tan_body)
        l_dot    = np.dot(s['uav_vel'][:, i] - v_anchor, dir_cable)

        # Modello softplus (approssimazione smooth di max(0, delta_L))
        exponent = p.beta_smooth * delta_L
        if exponent > 50.0:
            sigmoid   = 1.0
            soft_dist = delta_L
        elif exponent < -50.0:
            sigmoid   = 0.0
            soft_dist = 0.0
        else:
            term = np.exp(-exponent)
            sigmoid = 1.0 / (1.0 + term)
            if exponent > 0:
                soft_dist = (exponent + np.log(1.0 + np.exp(-exponent))) / p.beta_smooth
            else:
                soft_dist = np.log(1.0 + np.exp(exponent)) / p.beta_smooth

        f_spring = p.k_cable * soft_dist

        if l_dot < 0:
            raw_damp = (p.gamma_damp * l_dot) * sigmoid
            # Clamp: lo smorzamento non può annullare più del 90% della forza elastica
            f_damp = max(-0.9 * f_spring, raw_damp)
        else:
            f_damp = (p.gamma_damp * l_dot) * sigmoid

        f_mag       = max(0.0, f_spring + f_damp)
        f_cable_vec = f_mag * dir_cable

        # UAV Dynamics
        F_thrust_req = p.m_drone * (u_acc[:, i] + g_vec)
        # Point-mass UAV: non può generare thrust verso il basso
        if F_thrust_req[2] < 0:
            F_thrust_req = np.zeros(3)
        thrust_mag = np.linalg.norm(F_thrust_req)

        # Ground effect (solo componente verticale, fisicamente corretto)
        f_ground_effect = compute_ground_effect(
            F_thrust_req, s['uav_pos'][2, i] - p.floor_z, p.prop_radius
        )

        # Drag aerodinamico UAV con velocità relativa corretta
        v_rel_uav = s['uav_vel'][:, i] - wind_vec_world
        f_aero_drag_uav = compute_aerodynamics(v_rel_uav, p.rho, p.Cd_uav, p.A_uav)

        F_total = (F_thrust_req
                   + f_aero_drag_uav
                   - f_cable_vec
                   + f_repulsion[i, :]
                   - p.m_drone * g_vec
                   + f_ground_effect)

        # Contatto con suolo (UAV)
        if s['uav_pos'][2, i] < p.floor_z + 0.05:
            if s['uav_vel'][2, i] < 0.01:
                F_down = F_total[2]
                if F_down < 0:
                    F_total[2]  -= F_down
                    F_total[:2] -= 200.0 * s['uav_vel'][:2, i]

        d_uav_vel[:, i] = F_total / p.m_drone

        f_dw = compute_downwash(s['uav_pos'][:, i], s['pay_pos'], thrust_mag, p)
        f_downwash_tot += f_dw
        f_cables_sum   += f_cable_vec
        tau_cables_sum += np.cross(r_arm_world, f_cable_vec)

    # =========================================================================
    # 4. PAYLOAD DYNAMICS
    # =========================================================================
    # Forza di reazione sloshing nel world frame
    F_slosh_world = R_pay @ F_slosh_reaction_body

    # Coppia sloshing: braccio = posizione del centroide del liquido nel world frame
    r_slosh_from_com_body = e_slosh - e_com_body
    r_slosh_world = R_pay @ r_slosh_from_com_body
    tau_slosh = np.cross(r_slosh_world, F_slosh_world)

    # Somma forze esterne sul sistema (payload + liquido)
    # FIX: la massa totale è m_tot = m_payload + m_liquid
    # F_slosh è una forza interna al sistema: non va sommata a f_sum per calcolare acc_pay,
    # ma è già inclusa implicitamente. La forza netta esterna sul sistema è:
    f_ext_sum = (f_cables_sum
                 + f_aero_pay
                 + f_downwash_tot
                 #+ F_slosh_world
                 )

    # FIX: secondo principio al sistema completo (contenitore + liquido)
    acc_pay = f_ext_sum / m_tot - g_vec

    # =========================================================================
    # 5. GROUND CONTACT (Payload)
    # =========================================================================
    floor_z    = p.floor_z
    k_correct  = 600.0
    d_dissip   = 80.0   # 0
    geo_points = []
    hh = p.pay_h / 2.0
    is_box = getattr(p, 'payload_shape', 'box') in ['box', 'rect', 'square']

    if is_box:
        hl = p.pay_l / 2.0
        hw = p.pay_w / 2.0
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    geo_points.append(np.array([sx * hw, sy * hl, sz * hh]))
        geo_points.append(np.array([0.0, 0.0, -hh]))
        geo_points.append(np.array([0.0, 0.0,  hh]))
    else:
        r_cyl  = getattr(p, 'R_disk', 0.5)
        n_rad  = 8
        for i in range(n_rad):
            ang = 2.0 * np.pi * i / n_rad
            geo_points.append(np.array([r_cyl * np.cos(ang), r_cyl * np.sin(ang), -hh]))
            geo_points.append(np.array([r_cyl * np.cos(ang), r_cyl * np.sin(ang),  hh]))
        geo_points.append(np.array([0.0, 0.0, -hh]))

    f_floor_reaction   = np.zeros(3)
    tau_floor_reaction = np.zeros(3)
    contact_points     = 0

    for pt_geo in geo_points:
        r_pt_body  = pt_geo   # Punti geometrici già relativi al GC
        r_pt_world = R_pay @ r_pt_body
        pos_pt     = s['pay_pos'] + r_pt_world
        penetration = floor_z - pos_pt[2]

        if penetration > -0.002:
            contact_points += 1
            vel_tan_world   = R_pay @ np.cross(s['pay_omega'], r_pt_body)
            vel_point_world = s['pay_vel'] + vel_tan_world
            vz_point        = vel_point_world[2]

            f_n = 0.0
            if penetration > 0:
                f_n += k_correct * penetration
            if vz_point < 0:
                f_n += -d_dissip * vz_point
            f_n_val = max(0.0, f_n)

            f_normal_vec = np.array([0.0, 0.0, f_n_val])
            f_floor_reaction   += f_normal_vec
            tau_floor_reaction += np.cross(r_pt_world, f_normal_vec)

    # FIX: anche la reazione del suolo va divisa per m_tot
    acc_pay += f_floor_reaction / m_tot

    if contact_points >= 3:
        if acc_pay[2] < 0:
            acc_pay[2] = 0.0
        if np.linalg.norm(s['pay_vel']) < 0.2:
            acc_pay -= 200.0 * s['pay_vel']
            tau_floor_reaction -= 50.0 * s['pay_omega']

    if contact_points > 0:
        mu = 1.8
        f_norm_mag = f_floor_reaction[2]
        vel_xy     = np.array([s['pay_vel'][0], s['pay_vel'][1], 0.0])
        vel_xy_norm = np.linalg.norm(vel_xy)
        if vel_xy_norm > 1e-4:
            # FIX: anche l'attrito va diviso per m_tot
            acc_pay += (-mu * f_norm_mag * (vel_xy / vel_xy_norm)) / m_tot

    # =========================================================================
    # 6. ANGULAR DYNAMICS
    # =========================================================================
    # Tutti i momenti sono già nel world frame; conversione al body frame
    tau_world_total = (tau_cables_sum
                       + tau_wind_aero
                       + tau_slosh
                       + tau_floor_reaction)

    tau_body_total = R_pay.T @ tau_world_total

    # Smorzamento angolare (nel body frame)
    tau_body_with_damp = tau_body_total - p.pay_damping_ang * s['pay_omega']

    # Correzione tensore di inerzia via teorema di Steiner
    # e_com_body: spostamento del CoM del sistema rispetto al GC (body frame)
    # I_CoM = I_GC - k_{I,offset} * m_tot * (|e|^2 I - e e^T)
    # (k_{I,offset} = 0.2 è un parametro di tuning conservativo)
    e_com_norm_sq = np.dot(e_com_body, e_com_body)
    I_corr = m_tot * (e_com_norm_sq * np.eye(3) - np.outer(e_com_body, e_com_body))
    k_J_offset = 1 #0.2
    J_corrected = p.J - k_J_offset * I_corr
    invJ_corrected = np.linalg.inv(J_corrected)

    # Equazione di Eulero
    gyroscopic = np.cross(s['pay_omega'], J_corrected @ s['pay_omega'])
    alpha_pay  = invJ_corrected @ (tau_body_with_damp - gyroscopic)

    # Cinematica Eulero ZYX: dot(Phi) = Omega(phi,theta) * omega_body
    tt = theta
    if abs(np.cos(tt)) < 1e-2:
        tt = np.sign(tt) * 1.56   # Regolarizzazione per theta ~ ±pi/2
    W = np.array([
        [1, np.sin(phi) * np.tan(tt), np.cos(phi) * np.tan(tt)],
        [0, np.cos(phi),              -np.sin(phi)             ],
        [0, np.sin(phi) / np.cos(tt),  np.cos(phi) / np.cos(tt)]
    ])
    d_att = W @ s['pay_omega']

    dx = np.concatenate([
        s['uav_vel'].T.flatten(),
        s['pay_vel'],
        d_att,
        d_uav_vel.T.flatten(),
        acc_pay,
        alpha_pay,
        debug_forces['d_int_uav'].T.flatten(),
        debug_forces['d_int_pay']
    ])

    return dx