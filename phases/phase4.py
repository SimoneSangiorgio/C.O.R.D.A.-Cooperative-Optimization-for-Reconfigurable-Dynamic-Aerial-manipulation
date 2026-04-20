### NAVIGATION ###

import numpy as np

def get_bezier_point(s, p0, p1, p2, p3):
    return (1-s)**3 * p0 + 3 * (1-s)**2 * s * p1 + 3 * (1-s) * s**2 * p2 + s**3 * p3

def get_bezier_tangent(s, p0, p1, p2, p3):
    return 3 * (1-s)**2 * (p1 - p0) + 6 * (1-s) * s * (p2 - p1) + 3 * s**2 * (p3 - p2)

def get_bezier_acceleration(s, p0, p1, p2, p3):
    return 6 * (1-s) * (p2 - 2*p1 + p0) + 6 * s * (p3 - 2*p2 + p1)

def run(t, ctx, p, state, ref):
    elapsed = t - ctx.start_phase_time
    
    # 1. Inizializzazione Traiettoria (Bezier)
    if not hasattr(ctx, 'bezier_initialized') or not ctx.bezier_initialized:
        dist_linear = np.linalg.norm(p.payload_goal_pos - ctx.nav_start_pos)
        path_len_est = dist_linear * 1.2 
        ctx.traj_duration = max(5.0, path_len_est / p.nav_avg_vel)
        
        P0, P3 = ctx.nav_start_pos, p.payload_goal_pos
        vec_direct = P3 - P0
        dist = np.linalg.norm(vec_direct)
        dir_direct = vec_direct / (dist + 1e-6)
        
        dir_start = np.array([np.cos(ctx.ref_snap_yaw), np.sin(ctx.ref_snap_yaw), 0.0])
        P1 = P0 + dir_start * (dist * 0.4)
        perp_vec = np.cross(dir_direct, np.array([0,0,1])) 
        P2 = (P0 + dir_direct * (dist * 0.7)) + perp_vec * (dist * 1.2)
        
        ctx.bezier_P0, ctx.bezier_P1, ctx.bezier_P2, ctx.bezier_P3 = P0, P1, P2, P3
        ctx.bezier_initialized = True

    # 2. Input Shaping
    eff_L = p.L 
    T_pend = p.pendulum_correction * (2 * np.pi * np.sqrt(eff_L / p.g))
    dt_1, dt_2 = (0.5 * T_pend, 1.0 * T_pend) if p.input_shaping_enabled else (0.0, 0.0)
        
    def sample_trajectory(time_val):
        tv = np.clip(time_val, 0, ctx.traj_duration)
        tau = tv / ctx.traj_duration
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / ctx.traj_duration
        s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (ctx.traj_duration**2)
        
        pos = get_bezier_point(s, ctx.bezier_P0, ctx.bezier_P1, ctx.bezier_P2, ctx.bezier_P3)
        tan = get_bezier_tangent(s, ctx.bezier_P0, ctx.bezier_P1, ctx.bezier_P2, ctx.bezier_P3)
        curv = get_bezier_acceleration(s, ctx.bezier_P0, ctx.bezier_P1, ctx.bezier_P2, ctx.bezier_P3)
        return pos, tan * s_dot, (curv * s_dot**2 + tan * s_ddot)

    p1, v1, a1 = sample_trajectory(elapsed)
    p2, v2, a2 = sample_trajectory(elapsed - dt_1)
    p3, v3, a3 = sample_trajectory(elapsed - dt_2)
    
    # Calcolo riferimenti base
    ref['pos'] = 0.25*p1 + 0.50*p2 + 0.25*p3
    ref['vel'] = 0.25*v1 + 0.50*v2 + 0.25*v3
    
    # Rampa di accelerazione per evitare scatti nel vettore forza richiesto
    acc_ramp = np.clip(elapsed / 2.0, 0.0, 1.0)
    ref['acc'] = (0.25*a1 + 0.50*a2 + 0.25*a3) * acc_ramp

    # 3. GESTIONE YAW (Gerarchia e Stabilità)
    lambda_traj_eff = getattr(ctx, 'lambda_traj_effective', 0.0)

    if lambda_traj_eff > 0.5:
        # MODO PHYSICS: Segue fedelmente l'ottimizzatore (Weather Vane)
        target_yaw = getattr(ctx, 'last_stable_yaw', ctx.last_valid_yaw)
    elif getattr(p, 'yaw_tracking_enabled', False):
        # MODO TRAJECTORY: Segue la tangente della curva Bezier
        v_act = ref['vel']
        if np.hypot(v_act[0], v_act[1]) > 0.2:
            target_yaw = np.arctan2(v_act[1], v_act[0])
        else:
            target_yaw = ctx.last_valid_yaw
    else:
        # MODO FIXED: Segue traj_target_yaw configurato
        target_yaw = getattr(p, 'traj_target_yaw', 0.0)

    # Filtro Yaw: istantaneo in Physics mode per inseguire il vento, smorzato negli altri modi
    diff = np.arctan2(np.sin(target_yaw - ctx.last_valid_yaw), np.cos(target_yaw - ctx.last_valid_yaw))
    ref['yaw'] = ctx.last_valid_yaw + (1.0 if lambda_traj_eff > 0.5 else 0.1) * diff
    ctx.last_valid_yaw = ref['yaw']

    # 4. GESTIONE ASSETTO (Blending e Disaccoppiamento)
    if getattr(ctx, 'lambda_aero_effective', 0.0) > 0.5:
        # MODO Aero: l'assetto geometrico va a zero
        ctx.current_aero_blend = 1.0
        ctx.transition_att = (0.0, 0.0)
    else:
        # MODO Classico: ruotiamo i riferimenti (Roll/Pitch) in base allo Yaw attuale
        from mission import rotate_attitude_to_yaw
        r_curr, p_curr = rotate_attitude_to_yaw(p.traj_target_roll, p.traj_target_pitch, 
                                               p.traj_target_yaw, ref['yaw'])
        ctx.current_aero_blend = 0.0
        ctx.transition_att = (r_curr, p_curr)

    return elapsed >= (ctx.traj_duration + dt_2 + 1.0)