"""
CORDA

This module implements the full optimization loop with:
- CoM offset corrections (Parallel Axis Theorem)
- Slosh filter for dynamic CoM
- Aerodynamic & CoM-based attitude blending
- Enhanced momentum shift calculation
- Complete winch damping with jerk compensation
- Robust tension optimization with barrier functions
"""

import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar
from physics import get_rotation_matrix


# ==============================================================================
# GEOMETRY INITIALIZATION
# ==============================================================================
def compute_geometry(p):
    """
    Initialize UAV offsets and attachment points based on payload geometry.
    
    Returns:
        uav_offsets: Initial drone positions relative to payload center
        attach_vecs: Attachment point vectors (vec_att_i in document)
        geo_radius: Geometric radius of formation
    """
    uav_offsets = np.zeros((3, p.N))
    attach_vecs = np.zeros((3, p.N))
    
    lx = getattr(p, 'pay_l', 1.0)
    ly = getattr(p, 'pay_w', 1.0)
    r_disk = getattr(p, 'R_disk', 0.5)
    is_box = (p.payload_shape in ['box', 'rect', 'square'])

    init_theta_deg = getattr(p, 'theta_ref', 30.0)

    for i in range(p.N):
        angle = i * (2 * np.pi / p.N)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Compute attachment points on payload geometry
        if is_box:
            hx, hy = lx/2.0, ly/2.0
            if abs(cos_a) < 1e-6:
                scale = hy/abs(sin_a)
            elif abs(sin_a) < 1e-6:
                scale = hx/abs(cos_a)
            else:
                scale = min(hx/abs(cos_a), hy/abs(sin_a))
            ax, ay = scale*cos_a, scale*sin_a
        else:
            ax, ay = r_disk*cos_a, r_disk*sin_a

        # vec_att_i - Attachment points relative to GC
        attach_vecs[:, i] = [ax, ay, 0.0]
        
        # Initial UAV positions
        r_proj = p.L * np.sin(np.radians(init_theta_deg))
        uav_offsets[:, i] = [(ax + r_proj*cos_a), (ay + r_proj*sin_a), 0.0]
        
    geo_radius = np.mean(np.linalg.norm(uav_offsets[:2, :], axis=0))
    return uav_offsets, attach_vecs, geo_radius


# ==============================================================================
# TENSION SOLVER
# ==============================================================================
def solve_tensions(A, W_req, T_min_safe, T_max_limit):
    """
    Solve for optimal cable tensions:
    vec_T^* = argmin (1/2) * vec_T^T * vec_T
    subject to: A * vec_T = W_req (equality)
                T_safe <= vec_T_i <= T_limit (inequality)
    
    Returns:
        T_opt: Optimal tension vector
        residual: Residual error from equilibrium equation
    """
    # Protection against NaN/Inf
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        return np.full(A.shape[1], T_min_safe), 1e9
    if np.any(np.isnan(W_req)) or np.any(np.isinf(W_req)):
        return np.full(A.shape[1], T_min_safe), 1e9

    if T_min_safe >= T_max_limit:
        T_min_safe = T_max_limit - 1e-3

    N_cables = A.shape[1]
    lb = np.full(N_cables, T_min_safe)
    ub = np.full(N_cables, T_max_limit)

    # Try unconstrained least squares first (faster if feasible)
    try:
        x_lstsq, residuals, _, _ = np.linalg.lstsq(A, W_req, rcond=None)
        
        tol = 1e-4
        if np.all(x_lstsq >= T_min_safe - tol) and np.all(x_lstsq <= T_max_limit + tol):
            x_lstsq = np.clip(x_lstsq, T_min_safe, T_max_limit)
            if residuals.size > 0:
                cost = residuals[0]
            else:
                cost = np.sum((A @ x_lstsq - W_req)**2)
            return x_lstsq, cost
    except Exception:
        pass

    # Constrained optimization
    try:
        res = lsq_linear(A, W_req, bounds=(lb, ub), method='bvls', verbose=0)
        return res.x, res.cost
    except Exception:
        return lb, 1e9


# ==============================================================================
# MAIN FORMATION OPTIMIZATION
# ==============================================================================
def compute_optimal_formation(p, state, acc_cmd_pay, acc_ang_cmd_pay, ref_yaw, 
                              force_attitude=None, F_ext_total=None,
                              w_T_override=None,
                              ctx=None, aero_blend_override=None, 
                              com_offset_body=None, com_vel_body=None, com_acc_body=None, e_slosh_body=None,
                              F_aero_for_moment=None, M_ext_total=None):
    """
    Complete formation optimization algorithm implementing the mathematical formulation.
    
    This function:
    1. Computes required forces and moments (wrench)
    2. Determines optimal formation attitude (aerodynamic + CoM blending)
    3. Optimizes cone angle (alpha) to minimize cost function
    4. Computes final drone positions and cable tensions
    5. Applies active winch damping with jerk compensation
    
    Args:
        p: System parameters
        state: Current system state
        acc_cmd_pay: Desired payload acceleration (vec_a_des)
        acc_ang_cmd_pay: Desired angular acceleration (vec_dot_omega_des)
        ref_yaw: Reference yaw angle (psi)
        force_attitude: Target (roll, pitch) if specified
        F_ext_total: External forces (F_ext)
        com_offset_body: CoM offset in body frame (vec_e_CoM)
        com_vel_body: CoM velocity in body frame (for feedforward)
        com_acc_body: CoM acceleration in body frame (for feedforward)
        F_aero_for_moment: Force vector for moment calculation
        
    Returns:
        target_pos: Desired drone positions (vec_D_i)
        L_winch_cmd: Commanded cable lengths (L_cmd,i)
        theta_opt: Optimal cone angle (alpha_opt)
        ff_forces: Feedforward forces for each drone
    """

    # =========================================================================
    # NORMALIZATION SCALES (Scale Factors)
    # =========================================================================
    # Forza nominale (es. peso del payload diviso numero droni, o totale)
    m_tot = p.m_payload + p.m_liquid
    F_nom = m_tot * p.g
    if F_nom < 1.0: F_nom = 1.0

    # Momento nominale (Braccio medio * Forza nominale)
    # Se il payload è largo 1m, il braccio è 0.5m
    arm_nom = getattr(p, 'pay_l', 1.0) * 0.5
    M_nom = F_nom * arm_nom
    if M_nom < 0.1: M_nom = 0.1

    # Deviazione angolare "grande" (es. 20 gradi)
    # Serve per dire: "un errore di 20 gradi vale quanto un errore di forza nominale"
    alpha_scale_sq = np.radians(20.0)**2 

    # Condizionamento "cattivo" (es. log10(100) = 2)
    cond_scale = 2.0
    
    # =========================================================================
    # PARAMETER INITIALIZATION
    # =========================================================================
    if com_offset_body is None:
        com_offset_body = np.zeros(3)
    if com_vel_body is None:
        com_vel_body = np.zeros(3)
    if com_acc_body is None:
        com_acc_body = np.zeros(3)
    
    # Aerodynamic blending factor (lambda_aero)
    if aero_blend_override is not None:
        lambda_aero = np.clip(aero_blend_override, 0.0, 1.0)
    else:
        lambda_aero = getattr(p, 'lambda_aero', 0.0)

    # Shape blending factor (lambda_shape)
    lambda_shape = getattr(p, 'lambda_shape', 0.0)
    
    # CoM stability blending factor (lambda_CoM)
    lambda_CoM = getattr(p, 'lambda_CoM', 1.0)
    
    # Formation tilt gain (k_tilt)
    k_tilt = getattr(p, 'k_tilt', 1.0)
    
    # Minimum drone z (z_min)
    z_min = getattr(p, 'min_drone_z_rel', 1.5)
    
    # Desired cable length (L_des)
    L_des = p.L
    
    # Current attitude
    phi_curr, theta_curr, psi_curr = state['pay_att']
    R_curr = get_rotation_matrix(phi_curr, theta_curr, psi_curr)
    
    
    # =========================================================================
    # SLOSH FILTER FOR DYNAMIC CoM (tau_slosh, beta_slosh)
    # =========================================================================
    # Yaw-despun CoM eccentricity: vec_e_despun = R_z(psi)^T * vec_e_CoM
    c_psi, s_psi = np.cos(ref_yaw), np.sin(ref_yaw)
    R_z_inv = np.array([
        [c_psi, s_psi, 0],
        [-s_psi, c_psi, 0],
        [0, 0, 1]
    ])
    e_despun = R_z_inv @ com_offset_body
    
    # Apply slosh filter
    if ctx is not None:
        if not hasattr(ctx, 'e_filt_despun'):
            ctx.e_filt_despun = e_despun.copy()
        
        tau_slosh = getattr(p, 'tau_slosh', 0.15)  # Time constant
        dt = getattr(p, 'optimization_dt', 0.04)
        beta_slosh = dt / (tau_slosh + dt)  # Filter coefficient
        
        # Filtered CoM: vec_e_filt_despun_k = (1 - beta) * vec_e_filt_k-1 + beta * vec_e_despun
        ctx.e_filt_despun = (1.0 - beta_slosh) * ctx.e_filt_despun + beta_slosh * e_despun
        e_filt_despun = ctx.e_filt_despun
    else:
        e_filt_despun = e_despun
    
    
    # =========================================================================
    # INERTIA TENSOR CORRECTION (Parallel Axis Theorem)
    # =========================================================================
    # I_CoM = I_GC - m_p * (||vec_CoM||^2 * I_3 - vec_CoM * vec_CoM^T)
    d = com_offset_body
    d_norm_sq = np.dot(d, d)
    k_J_offset = 1 #0.2
    J_correction = m_tot * (d_norm_sq * np.eye(3) - np.outer(d, d))
    I_CoM = p.J - (k_J_offset * J_correction)

    eigvals = np.linalg.eigvalsh(I_CoM)
    if np.any(eigvals < 1e-6):
        I_CoM = p.J
    
    
    # =========================================================================
    # REQUIRED FORCE CALCULATION (vec_F_req)
    # =========================================================================
    # Target attitude for rotation matrix
    if force_attitude is not None:
        phi_des_p, theta_des_p = force_attitude
    else:
        phi_des_p, theta_des_p = phi_curr, theta_curr

    # Rotation matrix for desired attitude (used for moment calculations)
    R_des_temp = get_rotation_matrix(phi_des_p, theta_des_p, ref_yaw)
    
    # Inertial force from CoM acceleration
    # F_inertial = -m_p * vec_a_cmd (from CoM acceleration inertial effect)
    # Transform CoM inertial force to world frame
    F_wind = F_ext_total if F_ext_total is not None else np.zeros(3)

    # Gravity force
    F_g_vec = np.array([0, 0, -m_tot * p.g])

    # CoM acceleration feedforward (sloshing).
    # com_acc_body è già l'accelerazione della CoM pesata (m_liquid/m_tot * ddot_e_slosh).
    # I cavi devono fornire la forza aggiuntiva per accelerare il CoM insieme al payload.
    # Segno positivo: se il CoM accelera in avanti, servono più forza dai cavi.
    F_com_ff_world = R_des_temp @ (m_tot * com_acc_body)

    # Required force: i cavi devono compensare gravità, vento E fornire acc_des + acc_CoM
    # F_req = m_tot*(a_des + ddot_e_CoM) - m_tot*g - F_wind
    F_req = m_tot * acc_cmd_pay + F_com_ff_world - F_g_vec - F_wind
    
    # Enforce minimum F_req_z as per document
    if F_req[2] < 0.1:
        F_req[2] = 0.1


   ## =========================================================================
    # 4. YAW OVERRIDE / ROBUST TRAJECTORY ALIGNMENT (LAMBDA_TRAJ)
    # =========================================================================
    
    # --- Gestione Contesto Persistente ---
    if ctx is not None:
        if not hasattr(ctx, 'last_stable_yaw'): 
            ctx.last_stable_yaw = ref_yaw
    else:
        class MockCtx: pass
        ctx = MockCtx()
        ctx.last_stable_yaw = ref_yaw

    # --- Recupero Flag e Parametri ---
    lambda_traj = getattr(ctx, 'lambda_traj_effective', 0.0) if ctx else 0.0
    yaw_tracking_enabled = getattr(p, 'yaw_tracking_enabled', True)
    fixed_yaw_target = getattr(p, 'traj_target_yaw', 0.0)

    # --- Calcolo Direzione Forza XY (Yaw Fisico) ---
    F_req_xy_raw = F_req[:2]
    
    # 1. FILTRAGGIO VETTORIALE
    if not hasattr(ctx, 'F_xy_filt'):
        ctx.F_xy_filt = np.zeros(2)
        
    alpha_force = getattr(p, 'alpha_force_filter', 0.1) # Abbassato per più fluidità
    ctx.F_xy_filt = (1.0 - alpha_force) * ctx.F_xy_filt + alpha_force * F_req_xy_raw
    F_vec_smooth = ctx.F_xy_filt
    
    # 2. SELEZIONE VETTORE PER ALLINEAMENTO
    vector_for_yaw = F_vec_smooth # Fallback (F_req punta CONTROVENTO)
    using_wind_vector = False

    if F_ext_total is not None:
        f_ext_xy = F_ext_total[:2]
        # Se il vento è significativo, usiamo quello come riferimento primario
        if np.linalg.norm(f_ext_xy) > 0.5: 
            vector_for_yaw = f_ext_xy
            using_wind_vector = True

    vector_mag = np.linalg.norm(vector_for_yaw)

    # 3. CALCOLO ANGOLO TARGET (CORRETTO CON LOGICA AERODINAMICA)
    if vector_mag > 0.5:
        # Calcolo angolo base del vettore
        yaw_vec_angle = np.arctan2(vector_for_yaw[1], vector_for_yaw[0])
        
        # LOGICA DI DIREZIONE:
        # - Se usiamo F_ext (Vento): Esso punta SOTTOVENTO. Dobbiamo aggiungere PI per puntare il naso CONTROVENTO.
        # - Se usiamo F_req (Cavi): Esso punta (circa) CONTROVENTO. Non serve aggiungere PI.
        if using_wind_vector:
            yaw_absolute_target = yaw_vec_angle + np.pi
        else:
            yaw_absolute_target = yaw_vec_angle
            
        # --- LOGICA DI OFFSET: AERODINAMICA VS STRUTTURA ---
        if getattr(p, 'payload_shape', '') in ['box', 'rect', 'square']:
            # Confrontiamo le larghezze delle facce per minimizzare il drag
            if p.pay_w <= p.pay_l:
                # La faccia X (larga pay_w) è la più stretta. Nessun offset.
                offset_opt = 0.0
            else:
                # La faccia Y (larga pay_l) è la più stretta. Ruotiamo di 90°.
                offset_opt = np.pi / 2.0
                
            # Per un rettangolo con lati diversi, la simmetria fisica è di 180° (pi)
            if p.pay_w != p.pay_l:
                step_sym = np.pi
            else:
                step_sym = np.pi / 2.0 # Per un quadrato perfetto
        else:
            # Per cilindro/sfera l'aerodinamica non cambia ruotando lo yaw.
            # Offset strutturale (es. 45° per 4 droni) per minimizzare lo sbilanciamento dei cavi.
            offset_opt = np.pi / p.N
            step_sym = 2 * np.pi / p.N
        
        # --- COMPENSAZIONE DINAMICA DEL RITARDO DI FASE ---
        if not hasattr(ctx, 'last_yaw_abs_tgt'):
            ctx.last_yaw_abs_tgt = yaw_absolute_target
            
        dt = getattr(p, 'optimization_dt', 0.05)
        
        # Calcolo della velocità angolare attuale del vento (limitata per sicurezza)
        omega_wind = ((yaw_absolute_target - ctx.last_yaw_abs_tgt + np.pi) % (2 * np.pi) - np.pi) / dt
        omega_wind = np.clip(omega_wind, -0.5, 0.5) 
        ctx.last_yaw_abs_tgt = yaw_absolute_target
        
        # Calcolo dinamico dei 3 ritardi del sistema
        tau_force = dt * (1.0 - getattr(p, 'alpha_force_filter', 0.2)) / getattr(p, 'alpha_force_filter', 0.2)
        alpha_targets = getattr(p, 'formation_responsiveness', 0.06)
        tau_targets = dt * (1.0 - alpha_targets) / alpha_targets
        tau_pid = getattr(p, 'kv_pos', 20.0) / getattr(p, 'kp_pos', 25.0)
        
        # Compensazione totale (in radianti) iniettata come feedforward
        dynamic_lag_offset = omega_wind * (tau_force + tau_targets + tau_pid)
        
        # Applicazione al target finale
        desired_orientation = yaw_absolute_target + offset_opt + dynamic_lag_offset #+ np.pi / 180.0
        
        # Troviamo l'angolo di simmetria più vicino all'attuale per evitare "spin" inutili
        curr_yaw = ctx.last_stable_yaw
        
        # Calcoliamo la differenza rispetto all'orientamento attuale
        diff = desired_orientation - curr_yaw
        
        # Arrotondiamo al "passo" di simmetria più vicino (k * step_sym)
        k_steps = round(diff / step_sym)
        
        # Il target finale "aggancia" la simmetria più vicina
        yaw_target_raw = desired_orientation - (k_steps * step_sym)
        
    else:
        yaw_target_raw = ctx.last_stable_yaw

    # =====================================================================
    # [NUOVO] CALCOLO VELOCITÀ DEL TARGET (FEEDFORWARD)
    # =====================================================================
    if not hasattr(ctx, 'last_yaw_target_raw'):
        ctx.last_yaw_target_raw = yaw_target_raw
        
    # Calcoliamo di quanto si è spostato il vento in questo esatto istante
    target_vel = (yaw_target_raw - ctx.last_yaw_target_raw + np.pi) % (2 * np.pi) - np.pi
    ctx.last_yaw_target_raw = yaw_target_raw

    # 4. APPLICAZIONE CON SMORZAMENTO ("LAZY WEATHER VANE")
    if lambda_traj > 0.5:
        curr_yaw = ctx.last_stable_yaw
        yaw_err = yaw_target_raw - curr_yaw
        
        # Normalizzazione errore in [-pi, pi]
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi 
        
        deadband_deg = 0.0 
        deadband_rad = np.radians(deadband_deg)
        
        if abs(yaw_err) < deadband_rad:
            yaw_err_effective = 0.0
        else:
            yaw_err_effective = yaw_err - np.sign(yaw_err) * deadband_rad

        force_range = max(p.yaw_force_max - p.yaw_force_min, 1e-3)

        authority = np.clip((vector_mag - p.yaw_force_min) / force_range, 0.0, 1.0)
        yaw_gain = 0.15 * authority 
        
        # >>> AZIONE DI CONTROLLO COMPLETA: Proporzionale + Feedforward <<<
        # Il Feedforward fa ruotare il drone alla stessa velocità del vento
        # L'azione proporzionale (yaw_err_effective * yaw_gain) serve ora SOLO a 
        # chiudere il gap accumulato durante la Fase 0.
        step_yaw = (yaw_err_effective * yaw_gain) + target_vel
        
        max_step = np.radians(1.5) # Limite velocità rotazione
        step_yaw = np.clip(step_yaw, -max_step, max_step)
        
        new_stable_yaw = curr_yaw + step_yaw
        new_stable_yaw = (new_stable_yaw + np.pi) % (2 * np.pi) - np.pi
        
        ctx.last_stable_yaw = new_stable_yaw
        effective_yaw = new_stable_yaw

    else:
        # Modo Traiettoria Classico
        if yaw_tracking_enabled:
            target = ref_yaw
        else:
            target = fixed_yaw_target
        curr = ctx.last_stable_yaw
        err = (target - curr + np.pi) % (2*np.pi) - np.pi
        ctx.last_stable_yaw = curr + 1.0 * err ##################################### 0.1
        effective_yaw = ctx.last_stable_yaw
    
    # =========================================================================
    # AERODYNAMIC ATTITUDE CALCULATION (ROBUST VECTOR METHOD)
    # =========================================================================
    
    # 1. Portiamo la forza richiesta (F_req) nel frame ruotato solo di Yaw (Ref Yaw)
    #    Questo ci dice come il payload deve inclinarsi rispetto al suo attuale "avanti".
    c_psi, s_psi = np.cos(effective_yaw), np.sin(effective_yaw)
    R_z_inv = np.array([
        [c_psi, s_psi, 0],
        [-s_psi, c_psi, 0],
        [0, 0, 1]
    ])
    
    # F_req_body_yawed è la forza vista "dal sedile del pilota" (senza roll/pitch)
    F_req_body_yawed = R_z_inv @ F_req
    f_bx, f_by, f_bz = F_req_body_yawed

    # 2. Calcolo Esatto degli Angoli di Eulero
    #    Calcoliamo gli angoli necessari per allineare l'asse Z del payload con la forza risultante.
    #    NON forziamo mai phi=0 manualmente. Se il yaw è allineato al vento, 
    #    f_by sarà ~0 e quindi phi verrà calcolato automaticamente come ~0.
    #    Se invece il yaw è disallineato, phi compenserà la forza laterale stabilizzando il sistema.
    
    # Pitch (Theta): gestisce la componente X (Avanti/Indietro)
    theta_aero = np.arctan2(f_bx, f_bz)
    
    # Roll (Phi): gestisce la componente Y (Destra/Sinistra)
    # Nota: usiamo sqrt(fx^2 + fz^2) al denominatore per la corretta definizione di Eulero
    phi_aero = np.arctan2(-f_by, np.sqrt(f_bx**2 + f_bz**2))

    # --- Blending Finale ---
    phi_blend = (1.0 - lambda_aero) * phi_des_p + lambda_aero * phi_aero
    theta_blend = (1.0 - lambda_aero) * theta_des_p + lambda_aero * theta_aero
    
    # =========================================================================
    # CoM-BASED ATTITUDE CORRECTION (phi_CoM, theta_CoM)
    # =========================================================================
    # Balanced roll: phi_CoM = arctan(e_filt_despun_y / L_des)
    phi_CoM = np.arctan(e_filt_despun[1] / L_des)
    
    # Balanced pitch: theta_CoM = -arctan(e_filt_despun_x / L_des)
    theta_CoM = -np.arctan(e_filt_despun[0] / L_des)
    
    # Final roll: phi_final = k_tilt * phi_blend + lambda_CoM * phi_CoM
    phi_final = k_tilt * phi_blend + lambda_CoM * phi_CoM
    
    # Final pitch: theta_final = k_tilt * theta_blend + lambda_CoM * theta_CoM
    theta_final = k_tilt * theta_blend + lambda_CoM * theta_CoM

    R_des_p = get_rotation_matrix(phi_blend, theta_blend, effective_yaw)
    
    # Blended formation attitude: R_des_f = R_z(psi) * R_y(theta_final) * R_x(phi_final)
    R_des_f = get_rotation_matrix(phi_final, theta_final, effective_yaw)
    
    
    # Pure tilt rotation: R_tilt = R_des_f * R_z(psi)^T
    R_z_mat = np.array([
        [np.cos(effective_yaw), -np.sin(effective_yaw), 0],
        [np.sin(effective_yaw), np.cos(effective_yaw), 0],
        [0, 0, 1]
    ])
    R_tilt = R_des_f @ R_z_mat.T
    
    
    # =========================================================================
    # REQUIRED MOMENT CALCULATION (vec_M_req)
    # =========================================================================
    omega_des = state.get('pay_omega', np.zeros(3))
    omega_curr = state.get('pay_omega', np.zeros(3))
    
    # 1. INERTIAL MOMENTUM (about Fixed GC)
    # Usiamo p.J direttamente (inerzia geometrica)
    term_J_alpha = I_CoM @ acc_ang_cmd_pay
    term_gyro = np.cross(omega_curr, I_CoM @ omega_curr)
    
    M_inertial_body = term_J_alpha + term_gyro
    M_inertial_world = R_des_p @ M_inertial_body
    
    # 2. SLOSHING MOMENT COMPENSATION
    # Lo sloshing crea una coppia di disturbo: tau_slosh = r_slosh x F_slosh
    # Il controllore deve opporsi a questa coppia, quindi aggiungiamo -tau_slosh a M_req
    # O più semplicemente: M_req deve includere la coppia necessaria a vincere lo sloshing.
    
    if e_slosh_body is None:
        e_slosh_body = np.zeros(3)

    if p.m_liquid > 1e-6:
        # Braccio del momento: vettore dal CoM sistema al centroide del liquido
        # e_CoM = e_static + (m_liquid/m_tot)*e_slosh  (Section 2.2.4)
        e_static = getattr(p, 'CoM_offset', np.zeros(3))
        e_com_body = e_static + (p.m_liquid / m_tot) * e_slosh_body
        # Braccio rispetto al CoM (non al GC)
        r_slosh_from_com_body = e_slosh_body - e_com_body  # = (m_payload/m_tot)*(e_slosh - e_static)

        # Forza del liquido sul payload (reazione): F_slosh = -m_liquid * ddot_e_slosh
        # com_acc_body = (m_liquid/m_tot)*ddot_e_slosh  =>  ddot_e_slosh = com_acc_body * m_tot/m_liquid
        ddot_e_slosh = com_acc_body * (m_tot / p.m_liquid)
        F_slosh_reaction_world = R_des_p @ (-p.m_liquid * ddot_e_slosh)
    else:
        r_slosh_from_com_body = np.zeros(3)
        F_slosh_reaction_world = np.zeros(3)

    r_slosh_world = R_des_p @ r_slosh_from_com_body

    # Coppia disturbo dello sloshing rispetto al CoM
    M_slosh_disturbance = np.cross(r_slosh_world, F_slosh_reaction_world)
    
    # 3. AERODYNAMIC MOMENTUM
    # M_aero = (R_des * CoP_geo) x F_wind
    # CoP è definito rispetto al GC, quindi nessuna correzione necessaria
    r_cop_geo = getattr(p, 'CoP', np.zeros(3))
    r_cop_eff_body = r_cop_geo - com_offset_body
    CoP_world = R_des_p @ r_cop_eff_body
    F_moment_source = F_aero_for_moment if F_aero_for_moment is not None else F_wind
    M_aero = np.cross(CoP_world, F_moment_source)
    
    # TOTAL REQUIRED MOMENT
    # M_req = M_inertial - M_slosh_disturbance - M_aero
    # Nota: I segni dipendono dalla convenzione. 
    # Vogliamo M_cavi tale che: M_cavi + M_aero + M_slosh = M_inertial
    # Quindi M_cavi = M_inertial - M_aero - M_slosh
    
    M_req = M_inertial_world - M_slosh_disturbance - M_aero
    
    if M_ext_total is not None:
        M_req -= M_ext_total

    
    if np.linalg.norm(omega_curr) > 1e-3 and p.m_liquid > 1e-6:
         v_com_world = R_des_p @ com_vel_body
         F_coriolis = 2 * p.m_liquid * np.cross(omega_curr, v_com_world)
         M_coriolis = np.cross(r_slosh_world, F_coriolis)
         M_req -= M_coriolis

    W_req = np.concatenate([F_req, M_req])
    
    
    # =========================================================================
    # CoM STATIC SHIFT (vec_Delta_static)
    # =========================================================================
    # Delta_static = -vec_e_filt_despun * lambda_static
    lambda_static = getattr(p, 'lambda_static', 1.0)
    Delta_static = e_filt_despun * lambda_static
    
    # Apply safety limit
    max_shift = getattr(p, 'max_com_shift', 0.5)
    shift_norm = np.linalg.norm(Delta_static)
    if shift_norm > max_shift:
        Delta_static = (Delta_static / shift_norm) * max_shift
    
    # Convert to 3D (stays in XY plane)
    Delta_static_3d = np.array([Delta_static[0], Delta_static[1], 0.0])
    
    
    # =========================================================================
    # MOMENTUM SHIFT CALCULATION (vec_Delta_mom)
    # =========================================================================
    Delta_moment_applied = np.zeros(3)
    if hasattr(p, 'enable_momentum_shift') and p.enable_momentum_shift:
        F_req_norm_sq = np.linalg.norm(F_req)**2
        if F_req_norm_sq > 1e-2:
            # Delta_mom = (F_req x M_req) / ||F_req||^2
            raw_delta = np.cross(F_req, M_req) / F_req_norm_sq
            
            # Apply filtering if context available
            if ctx is not None:
                alpha_filt = p.alpha_moment_filter
                if not hasattr(ctx, 'last_delta_moment'):
                    ctx.last_delta_moment = np.zeros(3)
                ctx.last_delta_moment = alpha_filt * raw_delta + (1 - alpha_filt) * ctx.last_delta_moment
                Delta_moment_applied = ctx.last_delta_moment
            else:
                Delta_moment_applied = raw_delta
    
    
    # =========================================================================
    # DYNAMIC ANGLE & SAFETY BOUNDS
    # =========================================================================
    # alpha_dyn = arctan(||F_req_xy|| / |F_req_z|)
    F_req_z_abs = abs(F_req[2]) if abs(F_req[2]) > 0.1 else 0.1
    F_req_xy_norm = np.linalg.norm(F_req[:2])
    alpha_dyn = np.arctan(F_req_xy_norm / F_req_z_abs)
    
    # Safety bounds
    # alpha_safe = 1.1 * alpha_dyn
    min_safe_angle = np.radians(getattr(p, 'min_safe_angle', 10))
    alpha_safe = max(alpha_dyn*1.05, min_safe_angle)
    
    # alpha_limit = 0.9 * pi/2
    alpha_limit = 0.9 * (np.pi / 2)
    
    # Slew rate limiting
    # Slew rate limiting
    if ctx is not None and hasattr(ctx, 'last_theta_opt') and ctx.last_theta_opt is not None:
        max_delta = np.radians(getattr(p, 'max_angle_variation', 0.05))
        search_min = max(alpha_safe, ctx.last_theta_opt - max_delta)
        search_max = min(alpha_limit, ctx.last_theta_opt + max_delta)
    else:
        search_min = alpha_safe
        search_max = alpha_limit

    # 1. Forziamo search_min a non superare MAI il limite fisico assoluto
    search_min = min(search_min, alpha_limit - 1e-4)

    # 2. Se i limiti si incrociano, il limite inferiore vince ma il massimo si adegua stando SEMPRE sopra
    if search_max <= search_min:
        search_max = search_min + 1e-4

    if k_tilt < 0.1:
        theta_ref_rad = np.radians(getattr(p, 'theta_ref', 30.0))
        min_allowed_angle = theta_ref_rad - np.radians(15.0)
        search_min = max(search_min, min_allowed_angle)
        # Riapplichiamo il cap di sicurezza nel caso min_allowed_angle fosse assurdo
        search_min = min(search_min, alpha_limit - 1e-4)
        if search_max <= search_min:
             search_max = search_min + 1e-4
    
    
    # =========================================================================
    # SHAPE TRANSFORMATION (Ellipse Deformation)
    # =========================================================================
    # Wind shift direction: hat_v_F_req = F_req_xy / ||F_req_xy||
    if F_req_xy_norm > 1e-3:
        wind_shift_norm = F_req[:2] / F_req_xy_norm  # Vettore unitario direzione forza orizzontale
        wind_shift_magnitude = F_req_xy_norm / F_req_z_abs  # = tan(alpha_dyn)
    else:
        wind_shift_norm = np.zeros(2)
        wind_shift_magnitude = 0.0
    
    # Required force direction basis
    wind_xy = F_req[:2]
    wind_mag = np.linalg.norm(wind_xy)
    if wind_mag > 1e-3:
        hat_v_F_req_xy = wind_xy / wind_mag  # Parallel direction
        hat_v_F_req_perp = np.array([-wind_xy[1], wind_xy[0]]) / wind_mag  # Perpendicular
    else:
        hat_v_F_req_xy = np.array([1.0, 0.0])
        hat_v_F_req_perp = np.array([0.0, 1.0])
    
    # R_req = [hat_v_F_req, hat_v_F_req_perp]
    R_req = np.column_stack((hat_v_F_req_xy, hat_v_F_req_perp))
    
    # Shape transformation with blending
    if lambda_shape > 1e-3:
        # S_fact = min(||F_req_xy|| / F_ref, 2.0)
        F_ref = getattr(p, 'F_ref', 15.0)
        S_fact = (wind_mag / F_ref)#**2
        S_fact = min(S_fact, 2.0)
        
        # k_par = max(1.0 - (1.0 - lambda_par) * min(S_fact, 1.0), 0.6)
        lambda_par = getattr(p, 'lambda_par', 0.8)
        k_par = max(1.0 - (1.0 - lambda_par) * min(S_fact, 1.0), 0.6)
        
        # k_perp = 1.0 + (lambda_perp - 1.0) * S_fact
        lambda_perp = getattr(p, 'lambda_perp', 1.8)
        k_perp = 1.0 + (lambda_perp - 1.0) * S_fact
        
        # S_def = diag(k_par, k_perp)
        S_def = np.diag([k_par, k_perp])
        
        # S_mixed = I_2 * (1 - lambda_shape) + S_def * lambda_shape
        S_mixed = np.eye(2) * (1.0 - lambda_shape) + S_def * lambda_shape
        
        # M_shape = R_req * S_mixed * R_req^T
        M_shape = R_req @ S_mixed @ R_req.T
    else:
        M_shape = np.eye(2)
    
    
    # =========================================================================
    # RADIAL VECTORS (Flat unit vectors - vec_u_rad_i)
    # =========================================================================
    # u_rad_i = [cos(2*pi*(i-1)/N), sin(2*pi*(i-1)/N)]
    p_a_hat = np.zeros_like(p.attach_vecs)
    for i in range(p.N):
        norm_xy = np.linalg.norm(p.attach_vecs[:2, i])
        if norm_xy > 1e-6:
            p_a_hat[:2, i] = p.attach_vecs[:2, i] / norm_xy
        else:
            ang = i * (2 * np.pi / p.N)
            p_a_hat[:2, i] = [np.cos(ang), np.sin(ang)]
    
    
    # =========================================================================
    # ATTACHMENT POINT PREPARATION
    # =========================================================================
    # vec_att_eff_i = vec_att_i - vec_CoM
    # These are relative to CoM in body frame
    attach_eff = np.zeros_like(p.attach_vecs)
    for i in range(p.N):
        attach_eff[:, i] = p.attach_vecs[:, i] - com_offset_body
    
    
    # =========================================================================
    # SAFETY TENSION BOUNDS
    # =========================================================================
    k_safe = getattr(p, 'k_safe', 0.1)
    k_safe2 = getattr(p, 'k_safe2', 0.1)
    base_safe = k_safe * (m_tot * p.g / p.N)
    F_ext_mag = np.linalg.norm(F_ext_total) if F_ext_total is not None else 0.0
    dynamic_safe = k_safe2 * (F_ext_mag / p.N)
    T_safe = base_safe + dynamic_safe
    
    k_limit = getattr(p, 'k_limit', 0.7)
    T_limit = (p.F_max_thrust - p.m_drone * p.g) * k_limit
    
    # [FIX] Saturazione logica: la tensione minima di sicurezza non può MAI 
    # richiedere più dell'85% della forza fisica disponibile, altrimenti i droni cadono.
    if T_safe >= (T_limit * 0.85):
        T_safe = T_limit * 0.85
    
    
    # =========================================================================
    # COST FUNCTION WEIGHTS
    # =========================================================================
    w_T = getattr(p, 'w_T', 1.0)
    w_ref = getattr(p, 'w_ref', 30.0)
    w_smooth = getattr(p, 'w_smooth', 100.0)
    w_barrier = getattr(p, 'w_barrier', 50.0)
    w_cond = getattr(p, 'w_cond', 500.0)
    w_resid_f = getattr(p, 'w_resid_f', 1000.0)
    w_resid_m = getattr(p, 'w_resid_m', 1000.0)
    
    # Reference angle
    M_req_norm = np.linalg.norm(M_req)
    M_nom_rot = 2.0
    # Guadagno: per ogni unità di M_nom, allarga di X gradi
    k_theta_boost = np.radians(10.0) # Esempio: +20 gradi se siamo al limite di coppia
    theta_boost = k_theta_boost * np.clip(M_req_norm / M_nom_rot, 0.0, 1.0)
    
    # Riferimento base (dal parametro)
    theta_ref = np.radians(getattr(p, 'theta_ref', 30.0))
    
    # Nuovo target per la funzione di costo: Base + Boost
    # Se serve tanta coppia, l'ottimizzatore cercherà un angolo più largo
    theta_ref_dyn = theta_ref + theta_boost

    alpha_prev = ctx.last_theta_opt if (ctx and hasattr(ctx, 'last_theta_opt') and ctx.last_theta_opt is not None) else theta_ref_dyn
    
    
    # =========================================================================
    # OPTIMIZATION COST FUNCTION
    # =========================================================================
    def cost_function(alpha_k):
        """
        J(alpha) = w_T*J_T + w_barrier*J_barrier + w_ref*J_ref + 
                   w_smooth*J_smooth + w_cond*J_cond + w_resid*resid
        
        Following the exact formulation from the document.
        """
        
        # --- Step 1: h(alpha_k) = L_des * cos(alpha_k) ---
        h = L_des * np.cos(alpha_k)
        
        # --- Step 2: r(alpha_k) = h * tan(alpha_k) ---
        r = h * np.tan(alpha_k)
        
        # --- Step 3: Delta_shift(alpha_k) = h * (||F_req_xy|| / |F_req_z|) ---
        rotated_Z = R_tilt @ np.array([0, 0, h])

        # Shift totale richiesto dalla forza orizzontale: h * tan(alpha_dyn) * hat_F_xy
        required_xy = h * wind_shift_magnitude * wind_shift_norm

        # Shift già fornito dall'inclinazione R_tilt (componente XY del vettore Z ruotato)
        provided_xy = rotated_Z[:2]

        # Delta residuo: zero se R_tilt compensa completamente
        Delta_shift_3d = np.zeros(3)
        Delta_shift_3d[:2] = required_xy - provided_xy

        # --- Twist ---
        M_req_world = W_req[3:]
        M_req_yaw_z = (R_z_mat.T @ M_req_world)[2]
        
        f_z_safe = max(abs(W_req[2]), 1.0)
        r_safe = max(r, 0.1)  # r_safe = max(r_opt, 0.1) nel blocco finale
        k_twist = getattr(p, 'lambda_twist', 1.0)
        raw_gamma = k_twist * M_req_yaw_z / (f_z_safe * r_safe)
        gamma_twist = np.clip(raw_gamma, -np.radians(45), np.radians(45))
        c_g, s_g = np.cos(gamma_twist), np.sin(gamma_twist)
        R_twist = np.array([
            [c_g, -s_g],
            [s_g,  c_g]
        ])
        
        # --- Step 4-5: Drone positions D_i(alpha_k) ---
        A_matrix = np.zeros((6, p.N))
        
        for i in range(p.N):
            # vec_att_eff_R_i = R_des_f * vec_att_eff_i
            att_eff_pay_i = R_des_p @ attach_eff[:, i]
            att_eff_form_i = R_des_f @ attach_eff[:, i]
            
            # Apply yaw rotation to radial vector BEFORE deformation
            u_rad_yawed = R_z_mat @ np.array([p_a_hat[0, i], p_a_hat[1, i], 0.0])
            
            # Apply shape deformation: vec_u_def_i = M_shape * vec_u_rad_i
            xy_deformed = M_shape @ u_rad_yawed[:2]
            xy_twisted = R_twist @ xy_deformed
            
            # Geometric expansion
            vec_flat = np.array([
                xy_twisted[0] * r,
                xy_twisted[1] * r,
                h
            ])
            geo_expansion = R_tilt @ vec_flat
            
            # D_i = CoM + att_eff_R + geo_expansion + Delta_shift + Delta_mom + Delta_static
            D_i = att_eff_form_i + geo_expansion + Delta_shift_3d + Delta_moment_applied + Delta_static_3d
            
            # Enforce minimum z: D_i_z = max(D_i_z, z_min)
            if D_i[2] < z_min:
                D_i[2] = z_min
            
            # --- Step 6: Cable vector L_i = D_i - att_eff_R_i ---
            L_vec_i = D_i - att_eff_pay_i
            
            # --- Step 7: Cable direction hat_v_L_i = L_i / ||L_i|| ---
            norm_L = np.linalg.norm(L_vec_i)
            if norm_L < 1e-3:
                norm_L = 1e-3
            hat_v_L_i = L_vec_i / norm_L
            
            # --- Step 9: Configuration matrix A_i = [hat_v_L_i; att_eff_R_i x hat_v_L_i] ---
            A_matrix[:3, i] = hat_v_L_i
            A_matrix[3:, i] = np.cross(att_eff_pay_i, hat_v_L_i)
        
        # --- Step 10: Condition number kappa(A) = min(||A|| * ||A^+||, 1000) ---
        
        try:
            cond_num = min(np.linalg.cond(A_matrix), 1000)
        except:
            cond_num = 1000
        
        # --- Step 11: Solve for T*(alpha_k) ---
        T_opt, resid_val = solve_tensions(A_matrix, W_req, T_safe, T_limit)
        
        # --- Step 12: T_max and T_min ---
        T_max = np.max(T_opt)
        T_min = np.min(T_opt)
        
        # --- Step 13: J_cond = kappa(A) ---
        J_cond = np.log10(cond_num) / cond_scale
        
        # --- Step 14: J_barrier = sum(B_low_i + B_high_i) ---
        J_barrier = 0.0
        scale_barrier = 100.0  
        
        # Margini di attivazione
        T_alert_low = min(T_safe * 1.5, T_limit * 0.4) 
        T_alert_high = max(T_limit * 0.85, T_alert_low + 0.1)
        
        # Softplus: log(1 + exp(k * x)) / k
        # Agisce come una curva che "curva dolcemente" verso lo zero
        k_sharp = 0.7 # Più è alto, più assomiglia a un angolo acuto
        
        for t_val in T_opt:
            # Barriera Inferiore (si attiva quando T_alert_low - t_val > 0)
            x_low = T_alert_low - t_val
            range_low = T_alert_low - T_safe
            # Calcolo Softplus per il limite inferiore
            soft_low = np.log(1.0 + np.exp(k_sharp * x_low)) / k_sharp
            J_barrier += scale_barrier * (soft_low / range_low)**2
            
            # Barriera Superiore (si attiva quando t_val - T_alert_high > 0)
            x_high = t_val - T_alert_high
            range_high = T_limit - T_alert_high
            # Calcolo Softplus per il limite superiore
            soft_high = np.log(1.0 + np.exp(k_sharp * x_high)) / k_sharp
            J_barrier += scale_barrier * (soft_high / range_high)**2
        
        # --- Step 15: J_T = T_max / T_limit ---
        J_T = T_max / T_limit
        
        # --- Step 16: J_ref = (alpha_k - theta_ref)^2 ---
        J_ref = ((alpha_k - theta_ref_dyn)**2) / alpha_scale_sq
        
        # --- Step 17: J_smooth = (alpha_k - alpha_prev)^2 ---
        smooth_scale_sq = np.radians(3.5)**2
        J_smooth = ((alpha_k - alpha_prev)**2) / smooth_scale_sq
        
        # --- Step 18: resid(alpha_k) = (1/2) * ||A*T* - W_req||^2 ---
        W_achieved = A_matrix @ T_opt
        diff_W = W_achieved - W_req
        J_resid_f = np.sum(diff_W[:3]**2) / (F_nom**2)
        J_resid_m = np.sum(diff_W[3:]**2) / (M_nom**2)
        
        # --- Step 19: Total cost J(alpha_k) ---
        J_total = (w_T * J_T + 
                   w_barrier * J_barrier + 
                   w_ref * J_ref + 
                   w_smooth * J_smooth + 
                   + w_cond * J_cond  
                   + w_resid_f * J_resid_f  
                   + w_resid_m * J_resid_m
                   )
        
        return J_total
    
    
    # =========================================================================
    # STEP 20: OPTIMIZATION - Find optimal alpha
    # =========================================================================
    # alpha_opt = argmin J(alpha_k) subject to alpha_safe < alpha < alpha_limit
    
    if search_min >= search_max:
        search_min = search_max - 1e-4

    
    
    res = minimize_scalar(cost_function, bounds=(search_min, search_max), method='bounded')
    alpha_opt = res.x
    
    # Update context
    if ctx is not None:
        ctx.last_theta_opt = alpha_opt
    
    
    # =========================================================================
    # FINAL FORMATION SHAPE CALCULATION
    # =========================================================================
    # --- Step 1: h(alpha_opt) ---
    h_opt = L_des * np.cos(alpha_opt)
    
    # --- Step 2: r(alpha_opt) ---
    r_opt = h_opt * np.tan(alpha_opt)
    
    # --- Step 3: Delta_shift(alpha_opt) ---
    rotated_Z_opt = R_tilt @ np.array([0, 0, h_opt])

    required_xy_opt = h_opt * wind_shift_magnitude * wind_shift_norm
    provided_xy_opt = rotated_Z_opt[:2]

    Delta_shift_opt = np.zeros(3)
    Delta_shift_opt[:2] = required_xy_opt - provided_xy_opt

    # --- Twist ---
    f_z_safe = max(abs(W_req[2]), 1.0) 
    r_safe = max(r_opt, 0.1)
    k_twist = getattr(p, 'lambda_twist', 1.0)
    raw_gamma = k_twist * W_req[5] / (f_z_safe * r_safe)
    gamma_twist = np.clip(raw_gamma, -np.radians(45), np.radians(45))
    c_g, s_g = np.cos(gamma_twist), np.sin(gamma_twist)
    R_twist = np.array([
        [c_g, -s_g],
        [s_g,  c_g]
    ])
    
    # --- Step 4-5: Final drone positions ---
    target_pos = np.zeros((3, p.N))
    L_vecs_final = []
    L_rigid_norms = []
    A_matrix_final = np.zeros((6, p.N))
    
    for i in range(p.N):
        # Rotated attachment point
        att_eff_pay_i = R_des_p @ attach_eff[:, i]
        att_eff_form_i = R_des_f @ attach_eff[:, i]
        
        # Apply yaw rotation to radial vector BEFORE deformation
        u_rad_yawed = R_z_mat @ np.array([p_a_hat[0, i], p_a_hat[1, i], 0.0])
        
        # Apply shape deformation
        xy_deformed = M_shape @ u_rad_yawed[:2]
        xy_twisted = R_twist @ xy_deformed
        
        # Geometric expansion
        vec_flat = np.array([
            xy_twisted[0] * r_opt,
            xy_twisted[1] * r_opt,
            h_opt
        ])
        geo_expansion = R_tilt @ vec_flat
        
        # Final drone position: D_i = att_eff_R + geo_exp + Delta_shift + Delta_mom + Delta_static
        D_i = att_eff_form_i + geo_expansion + Delta_shift_opt + Delta_moment_applied + Delta_static_3d
        
        # Enforce minimum z
        if D_i[2] < z_min:
            D_i[2] = z_min
        
        target_pos[:, i] = D_i #+ (R_des_f @ com_offset_body)
        
        # --- Step 6: Cable vector L_i ---
        L_vec_i = D_i - att_eff_pay_i
        L_rigid = np.linalg.norm(L_vec_i)
        
        if L_rigid < 1e-3:
            L_rigid = 1e-3
        
        # --- Step 7: Cable direction hat_v_L_i ---
        hat_v_L_i = L_vec_i / L_rigid
        L_vecs_final.append(hat_v_L_i)
        L_rigid_norms.append(L_rigid)
        
        # --- Step 9: Configuration matrix ---
        A_matrix_final[:3, i] = hat_v_L_i
        A_matrix_final[3:, i] = np.cross(att_eff_pay_i, hat_v_L_i)
    
    
    # =========================================================================
    # STEP 11: FINAL TENSION CALCULATION
    # =========================================================================
    T_final, _ = solve_tensions(A_matrix_final, W_req, T_safe, T_limit)
    
    
    # =========================================================================
    # ACTIVE WINCH DAMPING & CABLE LENGTH COMMAND
    # =========================================================================
    v_pay = state['pay_vel']
    omega_pay = state['pay_omega']
    v_uavs = state['uav_vel']
    
    # --- Step 12: Dynamic stiffness k_stiff_i = k_cable * (L_ref / ||L_i||) ---
    L_ref = getattr(p, 'L_ref_stiffness', 3.0)
    k_cable = getattr(p, 'k_cable', 150.0)
    
    # --- Step 14: Damping coefficient k_damp_i = 2 * zeta_des * sqrt(k_stiff_i * m_p/N) ---
    zeta_des = getattr(p, 'winch_damping_ratio', 0.7)
    
    # --- Step 17: Feed-forward jerk compensation ---
    tau_relax = getattr(p, 'winch_tau_relax', 0.1)
    beta_jerk = 0.1  # Filter coefficient for tension derivative
    
    # Initialize damping correction if needed
    if ctx is not None:
        if not hasattr(ctx, 'last_damp_correction'):
            ctx.last_damp_correction = np.zeros(p.N)
        if not hasattr(ctx, 'last_T_final'):
            ctx.last_T_final = T_final.copy()
        if not hasattr(ctx, 'dot_T_filt'):
            ctx.dot_T_filt = np.zeros(p.N)
    
    L_winch_cmd = np.zeros(p.N)
    ff_forces = np.zeros((3, p.N))
    
    for i in range(p.N):
        # --- Step 12: k_stiff_i ---
        k_stiff_i = k_cable * (L_ref / L_rigid_norms[i])
        
        # --- Step 14: k_damp_i ---
        k_damp_i = 2 * zeta_des * np.sqrt(k_stiff_i * (m_tot/ p.N))
        
        # --- Step 8: Attachment point velocity ---
        att_eff_pay_i = R_des_p @ attach_eff[:, i]
        v_att_point = v_pay + np.cross(omega_pay, att_eff_pay_i)
        
        # Relative velocity: vec_v_rel_i = vec_v_d_i - vec_v_eff_R_i
        v_rel = v_uavs[:, i] - v_att_point
        
        # Cable extension velocity: v_cable_i = v_rel · hat_v_L_i
        v_cable_i = np.dot(v_rel, L_vecs_final[i])
        
        # Suppress noise
        if abs(v_cable_i) < 0.02:
            v_cable_i = 0.0
        
        # --- Step 15: Static elastic compensation Delta_L_static_i = T_i / k_stiff_i ---
        Delta_L_static = T_final[i] / k_stiff_i
        
        # --- Step 16: Velocity damping Delta_L_vel_i = (k_damp_i * v_cable_i) / k_stiff_i ---
        Delta_L_vel = (k_damp_i * v_cable_i) / k_stiff_i
        
        # --- Step 17: Dynamic jerk compensation ---
        # dot_T_i ≈ (T_i(k) - T_i(k-1)) / dt
        if ctx is not None and hasattr(ctx, 'last_T_final'):
            dt = getattr(p, 'optimization_dt', 0.04)
            dot_T_raw = (T_final[i] - ctx.last_T_final[i]) / dt
            
            # Filter: dot_T_filt_k = (1 - beta) * dot_T_filt_k-1 + beta * dot_T_raw
            ctx.dot_T_filt[i] = (1.0 - beta_jerk) * ctx.dot_T_filt[i] + beta_jerk * dot_T_raw
            dot_T_i = ctx.dot_T_filt[i]
        else:
            dot_T_i = 0.0
        
        # Delta_L_dynamic_i = tau_relax * (dot_T_filt / k_stiff_i)
        Delta_L_dynamic = tau_relax * (dot_T_i / k_stiff_i)
        
        # --- Step 18: Correction with clamping Delta_L_corr = clamp(Delta_L_vel + Delta_L_dynamic, -L_lim, +L_lim) ---
        L_lim = getattr(p, 'winch_limit_correction', 0.5)
        Delta_L_corr = np.clip(Delta_L_vel + Delta_L_dynamic, -L_lim, L_lim)
        
        # Apply smoothing if context available
        if ctx is not None:
            prev_corr = ctx.last_damp_correction[i]
            # Smooth changes to avoid jumps
            clamped_change = np.clip(Delta_L_corr - prev_corr, -0.002, 0.002)
            final_correction = prev_corr + (clamped_change * 0.2)
            ctx.last_damp_correction[i] = final_correction
        else:
            final_correction = Delta_L_corr
        
        # --- Step 19: Final cable length command L_cmd_i = ||L_i|| - Delta_L_static - Delta_L_corr ---
        L_cmd_raw = L_rigid_norms[i] - Delta_L_static - final_correction
        
        # --- Step 20: Integral clamping (optional altitude correction) ---
        # This would be added if there's payload altitude error
        # For now, we apply the minimum length constraint
        L_min = 0.1
        L_winch_cmd[i] = max(L_cmd_raw, L_min)
        
        # Feedforward forces
        ff_forces[:, i] = T_final[i] * L_vecs_final[i]
    
    # Store tension for next iteration
    if ctx is not None:
        ctx.last_T_final = T_final.copy()
    
    
    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    return target_pos, L_winch_cmd, alpha_opt, ff_forces