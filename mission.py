import numpy as np
import formation
import physics
import controls
import phases.phase0 as phase0
import phases.phase1 as phase1
import phases.phase2 as phase2
import phases.phase3 as phase3
import phases.phase4 as phase4
import phases.phase5 as phase5
import phases.phase6 as phase6

# ==============================================================================
# UTILS & HELPERS
# ==============================================================================
def check_stabilization(t, ctx, p, state):
    v_lim = getattr(p, 'stab_vel_limit', 0.1)
    w_lim = getattr(p, 'stab_omega_limit', 0.1)
    wait_time = getattr(p, 'stabilization_wait_time', 1.0) 
    
    v_curr = np.linalg.norm(state['pay_vel'])
    w_curr = np.linalg.norm(state['pay_omega'])
    
    if v_curr < v_lim and w_curr < w_lim:
        if ctx.stabilization_start_time < 0:
            ctx.stabilization_start_time = t
            
        if (t - ctx.stabilization_start_time) >= wait_time:
            return True
    else:
        ctx.stabilization_start_time = -1.0
        
    return False

def get_mission_reference(t, ctx, p, state):
    base_pos = ctx.ref_snap_pos if ctx.ref_snap_pos is not None else p.home_pos
    base_yaw = ctx.ref_snap_yaw
    
    ref = {
        'pos': base_pos.copy(), 'vel': np.zeros(3), 'acc': np.zeros(3),
        'yaw': base_yaw, 'z_rel': p.safe_altitude_offset, 'scale': 1.0
    }
    return ref


def update_dynamic_wind(t, p):
    """
    Calcola il vento. 
    LOGICA RICHIESTA:
    - Se enable_gusts == True:  Vento Base = 0, conta solo la gust_schedule.
    - Se enable_gusts == False: Vento = Base ruotato (gust_schedule ignorata).
    """
    
    current_wind = np.zeros(3)

    base_vec = getattr(p, 'initial_wind_vec', np.zeros(3)).copy()
    rot_vel = getattr(p, 'Rot_wind', np.zeros(3))
    rot_mag = np.linalg.norm(rot_vel)
    
    if rot_mag > 1e-6:
        # Applica rotazione se definita
        u = rot_vel / rot_mag
        theta = rot_mag * t
        cross_prod = np.cross(u, base_vec)
        dot_prod = np.dot(u, base_vec)
        current_base_wind = (base_vec * np.cos(theta) + 
                             cross_prod * np.sin(theta) + 
                             u * dot_prod * (1 - np.cos(theta)))
    else:
        current_base_wind = base_vec

    # 2. Calcola la componente RAFFICHE (Gusts)
    gust_wind = np.zeros(3)
    if getattr(p, 'enable_gusts', False):
        for gust in p.gust_schedule:
            t_s = gust['t_start']
            t_duration = gust['duration']
            t_e = t_s + t_duration
            ramp = gust.get('ramp', 0.5) 
            
            if t_s <= t <= t_e:
                target_vec = gust['vec']
                scale = 1.0
                if t < (t_s + ramp):
                    scale = (t - t_s) / ramp
                elif t > (t_e - ramp):
                    scale = (t_e - t) / ramp
                scale = np.clip(scale, 0.0, 1.0)
                
                gust_wind += (target_vec * scale)

    # 3. SOMMA TOTALE (Base + Raffiche)
    p.wind_vel = current_base_wind + gust_wind

# ==============================================================================
# MISSION CONTEXT CLASS
# ==============================================================================
class MissionContext:
    def __init__(self, p):
        self.phase = 0
        self.start_phase_time = 0.0        
        self.stabilization_start_time = -1.0       
        self.traj_duration = 0.0           
        self.last_valid_yaw = 0.0          
        
        self.nav_start_pos = None          
        self.ref_snap_pos = None
        self.ref_snap_yaw = 0.0
        self.ref_snap_att = (0.0, 0.0)
        self.final_ref_pos = None
        self.final_yaw = None
        
        self.int_err_uav = np.zeros((3, p.N))
        self.pay_err_integral = np.zeros(3)
        self.last_u_acc = np.zeros((3, p.N))
        
        # Optimizer Cache
        self.last_opt_time = -1.0
        self.cache_targets = None
        self.cache_L_winch = None
        self.cache_ff = None
        self.opt_filter_init = False
        self.filt_targets = None
        self.filt_L_winch = None
        self.filt_ff = None
        
        self.hover_config_targets = None
        self.hover_config_L = None
        self.hover_config_ff = None
        self.hover_theta_opt = 0.0 
        self.theta_log = []
        self.current_aero_blend = 0.0
        
        self.custom_uav_targets = None
        self.custom_feedforward = None
        self.current_L_rest = np.ones(p.N) * p.L
        
        self.transition_att = (0.0, 0.0) 
        self.pos_correction = np.zeros(3)
        self.last_delta_moment = np.zeros(3)
        self.filtered_acc_opt = np.zeros(3)
        self.last_theta_opt = np.radians(getattr(p, 'theta_ref', 30.0)) 
        
        self.uav_formation_offset = np.zeros((3, p.N))
        self.calc_offset_flag = False
        
        self.current_ref = None
        self.final_ref_pos = None

    def enter_phase(self, new_phase, t, current_ref_pos, current_ref_yaw, actual_pay_pos=None):
        print(f"\n[t={t:.2f}] === TRANSITION: Phase {self.phase} -> {new_phase} ===")
        self.phase = new_phase
        self.start_phase_time = t 
        self.stabilization_start_time = -1.0 

        if new_phase == 1:
            if self.custom_uav_targets is not None:
                self.filt_targets = self.custom_uav_targets.copy()
                self.filt_L_winch = self.current_L_rest.copy()
                self.filt_ff = np.zeros((3, self.filt_targets.shape[1]))
                self.opt_filter_init = True
                
                if self.hover_theta_opt != 0.0:
                    self.last_theta_opt = self.hover_theta_opt

                self.calc_offset_flag = True
                
        
        if new_phase == 4:
            self.filtered_acc_opt = np.zeros(3)

        self.ref_snap_pos = current_ref_pos.copy()
        self.ref_snap_yaw = current_ref_yaw

        if hasattr(self, 'transition_att'):
            self.ref_snap_att = self.transition_att
        else:
            self.ref_snap_att = (0.0, 0.0)
            self.transition_att = (0.0, 0.0)
        
        if new_phase == 3:
            self.nav_start_pos = current_ref_pos.copy()

def rotate_attitude_to_yaw(roll_ref, pitch_ref, yaw_ref, yaw_current):
    """
    Calcola Roll/Pitch locali necessari affinché il vettore Z del drone (Thrust)
    punti nella stessa direzione definita da (roll_ref, pitch_ref) nel frame yaw_ref.
    
    Logica Vettoriale:
    1. Calcola il vettore Z target nel mondo.
    2. Ruota questo vettore nel frame locale (senza roll/pitch, solo yaw).
    3. Trova gli angoli che allineano la Z del drone a questo vettore.
    """
    # 1. Calcola il vettore Z desiderato nel frame "World/Ref"
    # Nota: R * [0,0,1] è semplicemente la terza colonna della matrice di rotazione
    R_ref = physics.get_rotation_matrix(roll_ref, pitch_ref, yaw_ref)
    z_ref_world = R_ref[:, 2] # Terza colonna
    
    # 2. Portiamo questo vettore nel frame "Senza Yaw" del drone
    # Ruotiamo indietro di yaw_current
    # R_z(-yaw) @ z_ref_world
    cy = np.cos(-yaw_current)
    sy = np.sin(-yaw_current)
    
    # Rotazione Z manuale per efficienza
    z_local_x = cy * z_ref_world[0] - sy * z_ref_world[1]
    z_local_y = sy * z_ref_world[0] + cy * z_ref_world[1]
    z_local_z = z_ref_world[2]
    
    # 3. Ora dobbiamo trovare roll_cmd, pitch_cmd tali che:
    # R_y(p) * R_x(r) * [0,0,1] = [z_local_x, z_local_y, z_local_z]
    # Sviluppando R_y(p)R_x(r) applicato a Z:
    # Vettore risultante = [ sin(p)*cos(r), -sin(r), cos(p)*cos(r) ]
    
    # Risolviamo per Roll (r):
    # -sin(r) = z_local_y  =>  sin(r) = -z_local_y
    # Clampiamo per sicurezza numerica
    sin_r = max(-1.0, min(1.0, -z_local_y))
    roll_cmd = np.arcsin(sin_r)
    
    # Risolviamo per Pitch (p):
    # tan(p) = x / z  (visto che x = sin(p)cos(r) e z = cos(p)cos(r))
    # Usiamo arctan2 per il quadrante corretto
    pitch_cmd = np.arctan2(z_local_x, z_local_z)
    
    return roll_cmd, pitch_cmd

# ==============================================================================
# 1. GUIDANCE FUNCTION
# ==============================================================================
def update_guidance(t, x, p, ctx):
    # Assicuriamo che il vento sia aggiornato per questo step di guida
    update_dynamic_wind(t, p)

    state = physics.unpack_state(x, p)
    
    if ctx.hover_config_targets is None:
        _init_hover_config(p, ctx, state)

    ref = get_mission_reference(t, ctx, p, state)
    
    if ctx.phase == 0: 
        ramp_finished = phase0.run(t, ctx, p, state, ref)
        if ramp_finished:
             if check_stabilization(t, ctx, p, state):
                 if p.stop_after_phase > 0: 
                    ctx.enter_phase(1, t, ref['pos'], ref['yaw'], actual_pay_pos=state['pay_pos'])
            
    elif ctx.phase == 1: 
        if phase1.run_liftoff(t, ctx, p, state, ref):
            if p.stop_after_phase > 1: 
                ctx.enter_phase(2, t, ref['pos'], ref['yaw'], actual_pay_pos=state['pay_pos'])

    elif ctx.phase == 2:
        if phase2.run_alignment(t, ctx, p, state, ref):
            if p.stop_after_phase > 2:
                ctx.enter_phase(3, t, ref['pos'], ref['yaw'], actual_pay_pos=state['pay_pos'])

    elif ctx.phase == 3:
        phase_finished = phase3.run_attitude_settling(t, ctx, p, state, ref)
        if phase_finished:
            if check_stabilization(t, ctx, p, state):
                if p.stop_after_phase > 3:
                    ctx.enter_phase(4, t, ref['pos'], ref['yaw'], actual_pay_pos=state['pay_pos'])
            
    elif ctx.phase == 4: 
        phase_finished = phase4.run(t, ctx, p, state, ref)
        if phase_finished:
            if check_stabilization(t, ctx, p, state):
                if p.stop_after_phase > 4:
                    ctx.enter_phase(5, t, ref['pos'], ref['yaw'], actual_pay_pos=state['pay_pos'])
            
    elif ctx.phase == 5:
        phase5.run_winch(t, ctx, p, state, ref)
        
    elif ctx.phase >= 6:
        phase6.run_hold(t, ctx, p, state, ref)
    
    ctx.current_ref = ref

    if ctx.phase >= 0:
        _run_optimization(t, ctx, p, state, ref)
    else:
        ctx.custom_uav_targets = None
        ctx.custom_feedforward = None

# ==============================================================================
# 2. PHYSICS FUNCTION
# ==============================================================================
def equations_of_motion(t, x, p, ctx):
    update_dynamic_wind(t, p)
    state = physics.unpack_state(x, p)
    
    ref = ctx.current_ref if ctx.current_ref is not None else {
        'pos': p.home_pos, 'vel': np.zeros(3), 'acc': np.zeros(3), 'yaw': 0.0
    }
    
    motors_idle = False
    if ctx.phase == 0:
        t_idle_limit = getattr(p, 't_idle', 1.0)
        if (t - ctx.start_phase_time) < t_idle_limit:
            motors_idle = True

    ctx.motors_idle = motors_idle
    
    u_acc, debug_forces = controls.compute_controls(state, ref, p, ctx, t)
    dx = physics.compute_derivatives(t, state, u_acc, p, ctx, debug_forces)
    
    return dx

def _init_hover_config(p, ctx, state):
    hover_state = {
        'uav_pos': state['uav_pos'], 'uav_vel': np.zeros((3, p.N)),
        'pay_pos': p.home_pos, 'pay_vel': np.zeros(3),
        'pay_att': np.zeros(3), 'pay_omega': np.zeros(3)
    }
    
    wind_vec_world = p.wind_vel
    wind_mag = np.linalg.norm(wind_vec_world)
    
    # Evitiamo divisioni per zero
    if wind_mag < 1e-3:
        F_wind_s = np.zeros(3)
    else:
        wind_dir_s = wind_vec_world / wind_mag
        
        R_s = physics.get_rotation_matrix(0.0, 0.0, 0.0)
        w_body_s = R_s.T @ wind_dir_s
        
        # Calcolo Area Proiettata
        if p.payload_shape in ['box','rect','square']:
                A_x_s = p.pay_w * p.pay_h; A_y_s = p.pay_l * p.pay_h; A_z_s = p.pay_l * p.pay_w
                proj_area_s = A_x_s*abs(w_body_s[0]) + A_y_s*abs(w_body_s[1]) + A_z_s*abs(w_body_s[2])
        else:
                A_side = 2.0 * p.R_disk * p.pay_h; A_top = np.pi * p.R_disk**2
                sin_tilt = np.sqrt(w_body_s[0]**2 + w_body_s[1]**2)
                proj_area_s = A_side * sin_tilt + A_top * abs(w_body_s[2])
        
        F_wind_s = 0.5 * p.rho * p.Cd_pay * proj_area_s * (wind_mag**2) * wind_dir_s
    
    ht, hl, h_theta, hff = formation.compute_optimal_formation(
        p, hover_state, np.zeros(3), np.zeros(3), 0.0, 
        force_attitude=(0.0, 0.0), F_ext_total=F_wind_s,
        F_aero_for_moment=F_wind_s 
    )
    ctx.hover_config_targets = ht
    ctx.hover_config_L = hl
    ctx.hover_config_ff = hff
    ctx.hover_theta_opt = h_theta

def _solve_aerodynamic_equilibrium_iterative(p, wind_vec_world):
    """
    Risolve iterativamente l'equilibrio aerodinamico per trovare il vero target attitude.
    Sostituisce l'approssimazione rettangolare fallace.
    """
    wind_mag = np.linalg.norm(wind_vec_world)
    if wind_mag < 1e-3:
        # CORREZIONE: Restituiamo 3 valori per coerenza col resto del codice
        return np.zeros(3), 0.0, 0.0 
        
    wind_dir = wind_vec_world / wind_mag
    phi, theta, psi = 0.0, 0.0, 0.0
    
    # 5 iterazioni sono sufficienti per una buona stima a 25Hz
    f_wind_final = np.zeros(3)
    
    for _ in range(5):
        R_curr = physics.get_rotation_matrix(phi, theta, psi)
        wind_in_body = R_curr.T @ wind_dir
        
        # Calcolo Area Precisa (come in graphs.py / physics.py)
        if p.payload_shape in ['box', 'rect', 'square']:
            A_x, A_y, A_z = p.pay_w * p.pay_h, p.pay_l * p.pay_h, p.pay_l * p.pay_w
            proj_area = A_x * abs(wind_in_body[0]) + A_y * abs(wind_in_body[1]) + A_z * abs(wind_in_body[2])
        else:
            # Cylinder logic
            A_side = 2.0 * p.R_disk * p.pay_h
            A_top = np.pi * p.R_disk**2
            sin_tilt = np.sqrt(wind_in_body[0]**2 + wind_in_body[1]**2)
            proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])
            
        f_wind_final = 0.5 * p.rho * p.Cd_pay * proj_area * (wind_mag**2) * wind_dir
        
        # Aggiorna angoli di equilibrio
        m_tot_eq = p.m_payload + p.m_liquid
        F_tot = np.array([0.0, 0.0, -m_tot_eq * p.g]) + f_wind_final
        F_req = -F_tot
        theta = np.arctan2(F_req[0], abs(F_req[2]))
        phi = np.arctan2(-F_req[1], np.sqrt(F_req[0]**2 + F_req[2]**2))
        
    return f_wind_final, phi, theta   

def _run_optimization(t, ctx, p, state, ref):
    # -------------------------------------------------------------------------
    # 1. FILTRAGGIO ACCELERAZIONE (Invariato)
    # -------------------------------------------------------------------------
    raw_acc_cmd = ref['acc']
    alpha_opt_acc = getattr(p, 'alpha_acc_filter', 0.1)
    ctx.filtered_acc_opt = (1 - alpha_opt_acc) * ctx.filtered_acc_opt + alpha_opt_acc * raw_acc_cmd
    acc_for_optimizer = ctx.filtered_acc_opt

    target_yaw = ref['yaw']

    # -------------------------------------------------------------------------
    # 2. GESTIONE RAMPE PER LAMBDA AERO
    # -------------------------------------------------------------------------
    # Valore target globale definito nei parametri (es. 1.0)
    lambda_aero_effective = 0.0
    lambda_traj_effective = 0.0
    
    
    if ctx.phase == 0:
        # PRETENSION: Forza entrambi a 0
        lambda_aero_effective = 0.0
        lambda_traj_effective = 0.0
        
    elif ctx.phase == 1:
        # LIFTOFF: Forza entrambi a 0
        lambda_aero_effective = 0.0
        lambda_traj_effective = p.lambda_traj
        
    elif ctx.phase == 2:
        # ALIGNMENT: lambda_aero = 0, lambda_traj attivo
        lambda_aero_effective = p.lambda_aero
        lambda_traj_effective = p.lambda_traj
        
    elif ctx.phase == 3:
        # ATTITUDE SETTLING: Entrambi attivi secondo parametri
        lambda_aero_effective = p.lambda_aero
        lambda_traj_effective = p.lambda_traj
        
    elif ctx.phase == 4:
        # NAVIGATION: Entrambi attivi secondo parametri
        lambda_aero_effective = p.lambda_aero
        lambda_traj_effective = p.lambda_traj
        
    elif ctx.phase == 5:
        # WINCH DOWN: Forza entrambi a 0
        lambda_aero_effective = 0.0
        lambda_traj_effective = 0.0
        
    else:
        # LANDING (phase 6+): Forza entrambi a 0
        lambda_aero_effective = 0.0
        lambda_traj_effective = 0.0
    
    # Salva nel contesto per uso nelle fasi
    ctx.lambda_aero_effective = lambda_aero_effective
    ctx.lambda_traj_effective = lambda_traj_effective

    #if current_lambda > 0.5:
    #    v_apparent = p.wind_vel - state['pay_vel']
    #    if np.linalg.norm(v_apparent[:2]) > 0.5:
    #        wind_heading = np.arctan2(v_apparent[1], v_apparent[0])
    #        target_yaw = wind_heading

    # -------------------------------------------------------------------------
    # 3. STIMA ANGOLI AERODINAMICI E BLENDING DEI TARGET PID
    # -------------------------------------------------------------------------
    wind_vec_world = p.wind_vel
    
    # Calcolo iterativo corretto: ora restituisce 3 valori sempre
    f_wind_est, roll_eq, pitch_eq = _solve_aerodynamic_equilibrium_iterative(p, wind_vec_world)
    
    # Assegnazione agli angoli target aerodinamici
    pitch_aero = pitch_eq
    roll_aero = roll_eq

    effective_yaw_est = getattr(ctx, 'last_stable_yaw', target_yaw)

    base_roll, base_pitch = (ctx.transition_att if hasattr(ctx, 'transition_att') else (0.0, 0.0))
    
    # BLENDING: Mix tra target geometrico e target aerodinamico usando la lambda calcolata
    tgt_roll_raw = (1.0 - lambda_aero_effective) * base_roll + lambda_aero_effective * roll_aero
    tgt_pitch_raw = (1.0 - lambda_aero_effective) * base_pitch + lambda_aero_effective * pitch_aero
    target_yaw_raw = target_yaw

    # --- NUOVO: FILTRO ESPONENZIALE PER AMMORBIDIRE IL TARGET ---
    if not hasattr(ctx, 'filt_target_att'):
        ctx.filt_target_att = np.array([tgt_roll_raw, tgt_pitch_raw, target_yaw_raw])
        
    # Usiamo lo stesso parametro di fluidità usato per la forma della formazione
    alpha_att = getattr(p, 'formation_responsiveness', 0.06) 
    
    # Filtro lineare per Roll e Pitch
    ctx.filt_target_att[0] = (1.0 - alpha_att) * ctx.filt_target_att[0] + alpha_att * tgt_roll_raw
    ctx.filt_target_att[1] = (1.0 - alpha_att) * ctx.filt_target_att[1] + alpha_att * tgt_pitch_raw
    
    # Filtro circolare per lo Yaw (per gestire correttamente il salto tra -pi e +pi)
    yaw_err = (target_yaw_raw - ctx.filt_target_att[2] + np.pi) % (2 * np.pi) - np.pi
    ctx.filt_target_att[2] += alpha_att * yaw_err

    # Assegniamo i valori ammorbiditi per il PID e per i grafici
    tgt_roll = ctx.filt_target_att[0]
    tgt_pitch = ctx.filt_target_att[1]
    target_yaw = ctx.filt_target_att[2]

    ctx.current_target_att = np.array([tgt_roll, tgt_pitch, target_yaw])

    # -------------------------------------------------------------------------
    # 4. CALCOLO PID ASSETTO (HIGH PRECISION / ZERO ERROR MODE)
    # -------------------------------------------------------------------------
    curr_att = state['pay_att']
    curr_omega = state['pay_omega']
    
    # --- A. DEFINIZIONE GAINS AGGRESSIVI ---
    # Questi valori forzano l'errore a zero.
    # Nota: Assicurati di aver aggiornato parameters.py con questi valori o inseriscili qui hardcoded per test.
    kp_roll = getattr(p, 'kp_roll', 60.0)    # Molto alto per rigidità
    kp_pitch = getattr(p, 'kp_pitch', 60.0)
    kp_yaw = getattr(p, 'kp_yaw', 40.0)
    
    kv_rot = getattr(p, 'kv_rot', 25.0)      # Damping alto per frenare gli scatti
    ki_rot = getattr(p, 'ki_rot', 1.0)      # Integrale alto per eliminare l'errore statico
    
    # --- B. CALCOLO ERRORE VETTORIALE ---
    # Calcoliamo la differenza diretta tra target e attuale
    err_att = np.array([tgt_roll, tgt_pitch, target_yaw]) - curr_att
    
    # Normalizzazione Yaw in [-pi, +pi] (Fondamentale)
    err_att[2] = (err_att[2] + np.pi) % (2 * np.pi) - np.pi

    # --- C. GESTIONE INTEGRALE (La memoria dell'errore) ---
    if not hasattr(ctx, 'int_err_att'):
        ctx.int_err_att = np.zeros(3)
    
    # Accumuliamo SEMPRE l'integrale se siamo in volo (Phase >= 1)
    # Questo permette al drone di "imparare" a compensare sbilanciamenti fin dal decollo.
    if ctx.phase >= 1: 
        ctx.int_err_att += err_att * p.optimization_dt
        
        # Anti-Windup: Limita l'integrale per evitare che cresca all'infinito
        # Deve essere abbastanza alto (es. 50.0) per permettere correzioni forti se necessario
        limit = getattr(p, 'rot_int_lim', 50.0) 
        ctx.int_err_att = np.clip(ctx.int_err_att, -limit, limit)
    else:
        # A terra resettiamo l'integrale
        ctx.int_err_att = np.zeros(3)
    
    # --- D. CALCOLO COMANDO ACCELERAZIONE ANGOLARE (PID) ---
    # Formula PID: Kp*errore + Ki*integrale - Kv*velocità_angolare
    # Usiamo err_att direttamente per pulizia e coerenza
    acc_ang_cmd = np.array([
        kp_roll  * err_att[0] + ki_rot * ctx.int_err_att[0] - kv_rot * curr_omega[0],
        kp_pitch * err_att[1] + ki_rot * ctx.int_err_att[1] - kv_rot * curr_omega[1],
        kp_yaw   * err_att[2] + ki_rot * ctx.int_err_att[2] - kv_rot * curr_omega[2]
    ])

    if lambda_traj_effective > 0.5:
        # 1. Spegniamo la richiesta di coppia attiva sullo Yaw
        acc_ang_cmd[2] = 0.0
        
        # 2. Uccidiamo l'integrale per evitare "wind-up" o memorie passate
        ctx.int_err_att[2] = 0.0
        
        # 3. (Opzionale ma pulito) Aggiorniamo il target visualizzato/debug
        # affinché corrisponda alla realtà fisica, evitando salti se lambda torna a 0
        if hasattr(ctx, 'last_stable_yaw'):
            target_yaw = ctx.last_stable_yaw
    
    # -------------------------------------------------------------------------
    # 5. PREPARAZIONE STATO PER OTTIMIZZATORE
    # -------------------------------------------------------------------------
    calc_state = state.copy()
    calc_state['pay_pos'] = ref['pos'] 
    calc_state['pay_vel'] = ref['vel']
    # Diciamo all'ottimizzatore che siamo già all'assetto desiderato (feedforward shape)
    calc_state['pay_att'] = np.array([tgt_roll, tgt_pitch, target_yaw]) 
    calc_state['pay_omega'] = np.zeros(3) 

    # Correzione posizione Payload (Loop esterno)
    pay_acc_correction = np.zeros(3)
    
    if ctx.phase >= 3: 
        # Errore posizione Payload
        err_pos = ref['pos'] - state['pay_pos']
        
        # Accumulo Integrale Payload
        ctx.pay_err_integral += err_pos * p.optimization_dt
        ctx.pay_err_integral = np.clip(ctx.pay_err_integral, -p.pay_int_lim, p.pay_int_lim)
        
        # PID Posizione Payload -> Output: Accelerazione richiesta
        # NOTA: Qui stiamo calcolando un'accelerazione, non una posizione!
        pay_acc_correction = (p.kp_pay_corr * err_pos) + \
                             (p.ki_pay_corr * ctx.pay_err_integral) - \
                             (p.kd_pay_corr * state['pay_vel']) # Smorzamento sul payload
        
        # Rampa di attivazione per evitare scatti all'inizio della fase 3
        k_corr_ramp = 1.0
        if ctx.phase == 3:
            t_in_phase = t - ctx.start_phase_time
            k_corr_ramp = np.clip(t_in_phase / 2.0, 0.0, 1.0) 
        
        pay_acc_correction *= k_corr_ramp

    else:
        ctx.pay_err_integral = np.zeros(3)
        pay_acc_correction = np.zeros(3)

    # L'accelerazione totale richiesta all'ottimizzatore è:
    # Traiettoria (Feedforward) + Correzione PID (Feedback)
    acc_for_optimizer = ctx.filtered_acc_opt + pay_acc_correction

    # IMPORTANTE: Resettiamo pos_correction a zero perché ora agiamo sulle forze, 
    # non sulla geometria "sporca".
    ctx.pos_correction = np.zeros(3)

    # -------------------------------------------------------------------------
    # 6. CALCOLO FORZE AERODINAMICHE REALI (v_rel = v_pay - v_wind)
    # -------------------------------------------------------------------------
    # Usiamo la velocità REALE del payload (state, non ref) e il vento corrente.
    # compute_payload_aero calcola correttamente:
    #   F_aero = -0.5 * rho * Cd * A_proj(v_rel_dir_body) * ||v_rel||^2 * v_rel_dir
    # con v_rel = v_pay_real - v_wind, includendo il cross-term quadratico.
    
    R_opt = physics.get_rotation_matrix(tgt_roll, tgt_pitch, target_yaw)
    
    # Forza aerodinamica unificata con velocità reale del payload
    f_aero_total = physics.compute_payload_aero(
        state['pay_vel'],   # velocità reale, non ref['vel']
        wind_vec_world,     # vento corrente
        R_opt,              # assetto stimato (blend aero/geometrico)
        p
    )
    
    # Per il calcolo del momento (weather-vane), usiamo la stessa forza
    # ma con solo la componente del vento (v_pay=0) per separare
    # il contributo stazionario da quello dinamico nel feedforward di coppia.
    # Se il payload è fermo, f_aero_total == f_wind_for_moment.
    v_pay_norm = np.linalg.norm(state['pay_vel'])
    if v_pay_norm < 0.5:
        # Payload lento: usa forza totale anche per il momento (più preciso)
        f_aero_moment = f_aero_total
    else:
        # Payload in moto: separa contributo vento statico per il momento
        # per evitare che la velocità di navigazione distorca il feedforward di coppia
        f_aero_moment = physics.compute_payload_aero(
            np.zeros(3),    # v_pay = 0: solo contributo vento
            wind_vec_world,
            R_opt,
            p
        )

    # -------------------------------------------------------------------------
    # 7. GENERAZIONE DATI CoM (Perfect Knowledge Simulation)
    # -------------------------------------------------------------------------
    current_com_offset = getattr(p, 'CoM_offset', np.zeros(3)).copy()
    com_vel_body = np.zeros(3)
    com_acc_body = np.zeros(3)
    e_slosh_body = np.zeros(3)
    
    if getattr(p, 'enable_sloshing', False):
        amp = getattr(p, 'slosh_amp', 0.3)
        freq = getattr(p, 'slosh_freq', 0.5)
        omega = 2.0 * np.pi * freq
        
        m_tot = p.m_payload + p.m_liquid
        if m_tot > 1e-6:
            mass_ratio = p.m_liquid / m_tot  # VERO spostamento fisico del baricentro!
        else:
            mass_ratio = 0.0

        liq_pos_x = amp * np.sin(omega * t)
        liq_pos_y = (amp / 2.0) * np.cos(omega * t)
        liq_vel_x = amp * omega * np.cos(omega * t)
        liq_vel_y = -(amp / 2.0) * omega * np.sin(omega * t)
        liq_acc_x = -amp * (omega**2) * np.sin(omega * t)
        liq_acc_y = -(amp / 2.0) * (omega**2) * np.cos(omega * t)

        e_slosh_body = np.array([liq_pos_x, liq_pos_y, 0.0])
        current_com_offset += mass_ratio * np.array([liq_pos_x, liq_pos_y, 0.0])
        com_vel_body = mass_ratio * np.array([liq_vel_x, liq_vel_y, 0.0])
        com_acc_body = mass_ratio * np.array([liq_acc_x, liq_acc_y, 0.0])

    # Bilanciamento pesi ottimizzatore
    curr_w_T = p.w_T
    F_total_force_balance = f_aero_total
    F_aero_moment_only = f_aero_moment

    if ctx.phase == 0:
        acc_for_optimizer = np.zeros(3)
        # In fase 0 il payload è a terra: v_pay ≈ 0, quindi f_aero_total
        # è già solo vento. Nessuna distinzione necessaria.
        F_total_force_balance = f_aero_total
        F_aero_moment_only = f_aero_total       
    
    if ctx.phase >= 3:
        tilt_sum = abs(tgt_roll) + abs(tgt_pitch)
        factor = 1.0 + (tilt_sum * p.weight_tilt_scaling)          
        curr_w_T = p.w_T / factor    
    
    eccentricity = np.linalg.norm(current_com_offset[:2])
    if eccentricity > 0.1:
        curr_w_T = curr_w_T / (1.0 + 5.0 * eccentricity)

    # -------------------------------------------------------------------------
    # 8. CHIAMATA OTTIMIZZATORE
    # -------------------------------------------------------------------------
    raw_targets, raw_L_winch, theta_opt, raw_ff = formation.compute_optimal_formation(
        p, calc_state, acc_for_optimizer, acc_ang_cmd, target_yaw, 
        force_attitude=(tgt_roll, tgt_pitch),
        F_ext_total=F_total_force_balance, 
        w_T_override=curr_w_T, 
        ctx=ctx,
        aero_blend_override=lambda_aero_effective,  # USIAMO LA LAMBDA CON RAMPE
        com_offset_body=current_com_offset,
        com_vel_body=com_vel_body,
        com_acc_body=com_acc_body,
        e_slosh_body=e_slosh_body, 
        F_aero_for_moment=F_aero_moment_only 
    )
    
    # -------------------------------------------------------------------------
    # 9. POST-PROCESSING (Offset, Filtering, etc.)
    # -------------------------------------------------------------------------
    if ctx.calc_offset_flag and ctx.filt_targets is not None and raw_targets is not None:
        ctx.uav_formation_offset = ctx.filt_targets - raw_targets
        ctx.calc_offset_flag = False
    
    if ctx.phase == 1:
        if raw_targets is not None:
            raw_targets = raw_targets + ctx.uav_formation_offset
    else:
        ctx.uav_formation_offset *= 0.0

    alpha_resp = getattr(p, 'formation_responsiveness', 0.07)
    if not hasattr(ctx, 'filt_theta_opt'):
        ctx.filt_theta_opt = theta_opt
    
    # Filtro esponenziale per rendere la curva visiva perfettamente fluida
    ctx.filt_theta_opt = (1.0 - alpha_resp) * ctx.filt_theta_opt + alpha_resp * theta_opt
    
    ctx.theta_log.append((t, ctx.filt_theta_opt))
    ctx.cache_targets = raw_targets
    ctx.cache_L_winch = raw_L_winch
    ctx.cache_ff = raw_ff
    ctx.last_opt_time = t

    if raw_targets is None:
            raw_targets = p.uav_offsets
            raw_L_winch = np.ones(p.N)*p.L
            raw_ff = np.zeros((3, p.N))

    if not ctx.opt_filter_init:
        ctx.filt_targets = raw_targets
        ctx.filt_L_winch = raw_L_winch
        ctx.filt_ff = raw_ff
        ctx.opt_filter_init = True

    
    alpha = getattr(p, 'formation_responsiveness', 0.4)


    ctx.filt_targets = (1 - alpha) * ctx.filt_targets + alpha * raw_targets
    ctx.filt_L_winch = (1 - alpha) * ctx.filt_L_winch + alpha * raw_L_winch
    ctx.filt_ff = (1 - alpha) * ctx.filt_ff + alpha * raw_ff
    
    if ctx.phase == 0:
        ctx.filt_targets = ctx.custom_uav_targets
        ctx.filt_L_winch = ctx.current_L_rest
        ctx.filt_ff = ctx.custom_feedforward
    else:
        if ctx.phase >= 5 and hasattr(ctx, 'winch_geom_targets') and ctx.winch_geom_targets is not None:
                ctx.filt_targets = ctx.winch_geom_targets
        
        ctx.custom_uav_targets = ctx.filt_targets + ctx.pos_correction[:, np.newaxis]
        ctx.current_L_rest = ctx.filt_L_winch 
        ctx.custom_feedforward = ctx.filt_ff