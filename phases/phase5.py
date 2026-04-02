import numpy as np

# ==============================================================================
# HELPER FUNCTIONS PER ANGOLI
# ==============================================================================
def norm_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def shortest_angular_diff(start, end):
    diff = (end - start + np.pi) % (2 * np.pi) - np.pi
    return diff

# ==============================================================================
# MAIN PHASE FUNCTION
# ==============================================================================
def run_winch(t, ctx, p, state, ref):
    from mission import rotate_attitude_to_yaw
    
    # Init Riferimenti
    ref['pos'] = p.payload_goal_pos
    ref['vel'] = np.zeros(3)
    ref['acc'] = np.zeros(3)
    
    # --- INIZIALIZZAZIONE FASE ---
    if not hasattr(ctx, 'winch_init_done') or not ctx.winch_init_done:
        yaw_source = None
        if hasattr(ctx, 'last_stable_yaw') and ctx.last_stable_yaw is not None:
            yaw_source = ctx.last_stable_yaw
        
        if yaw_source is None:
            if hasattr(ctx, 'last_valid_yaw') and ctx.last_valid_yaw is not None:
                yaw_source = ctx.last_valid_yaw
            else:
                yaw_source = ref['yaw']

        start_yaw_norm = norm_angle(yaw_source)
        final_yaw_norm = norm_angle(p.final_target_yaw)
        
        ctx.start_yaw = start_yaw_norm
        ctx.total_yaw_diff = shortest_angular_diff(start_yaw_norm, final_yaw_norm)
        
        ctx.start_world_roll = p.traj_target_roll
        ctx.start_world_pitch = p.traj_target_pitch
        ctx.start_blend_val = getattr(ctx, 'current_aero_blend', 0.0)
            
        ctx.winch_init_done = True

    # --- ESECUZIONE TRAIETTORIA ---
    elapsed = t - ctx.start_phase_time
    duration = p.t_attitude_2
    
    # Calcolo curve (Posizione, Velocità, Accelerazione)
    if elapsed <= duration:
        tau = elapsed / duration
        # Quintic (Posizione)
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        # Derivata Prima (Velocità normalizzata)
        s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / duration
        # Derivata Seconda (Accelerazione normalizzata)
        s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (duration**2)
    else:
        s = 1.0; s_dot = 0.0; s_ddot = 0.0
    
    # 1. Yaw Posizione
    curr_yaw_target = ctx.start_yaw + s * ctx.total_yaw_diff
    curr_yaw_target = norm_angle(curr_yaw_target)
    ref['yaw'] = curr_yaw_target
    
    # --- NOVITÀ: FEEDFORWARD VELOCITÀ E ACCELERAZIONE YAW ---
    # Questo dice al PID di "accompagnare" il movimento invece di frenarlo
    ref['yaw_vel'] = s_dot * ctx.total_yaw_diff
    ref['yaw_acc'] = s_ddot * ctx.total_yaw_diff
    
    # Sincronizzazione Ottimizzatore
    ctx.last_valid_yaw = curr_yaw_target
    ctx.last_stable_yaw = curr_yaw_target
    
    # 2. Assetto (Roll/Pitch)
    target_world_roll = (1-s)*ctx.start_world_roll + s*p.final_target_roll
    target_world_pitch = (1-s)*ctx.start_world_pitch + s*p.final_target_pitch
    
    local_roll, local_pitch = rotate_attitude_to_yaw(
        target_world_roll, 
        target_world_pitch, 
        p.final_target_yaw, 
        curr_yaw_target
    )
    
    ctx.transition_att = (local_roll, local_pitch)
    
    # 3. Blend e Chiusura
    ctx.current_aero_blend = (1 - s) * ctx.start_blend_val + s * 0.0
    ctx.winch_geom_targets = None 
    
    if elapsed > duration + 1.0:
        final_r, final_p = rotate_attitude_to_yaw(
            p.final_target_roll, p.final_target_pitch, 
            p.final_target_yaw, p.final_target_yaw 
        )
        ctx.transition_att = (final_r, final_p)
        ctx.current_aero_blend = 0.0 
        ctx.phase = 6
        ctx.final_yaw = p.final_target_yaw 
        ctx.final_L_rest = ctx.current_L_rest.copy()
        
        # Reset feedforward alla fine
        ref['yaw_vel'] = 0.0
        ref['yaw_acc'] = 0.0