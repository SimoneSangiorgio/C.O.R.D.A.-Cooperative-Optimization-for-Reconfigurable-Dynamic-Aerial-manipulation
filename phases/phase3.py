import numpy as np

def run_attitude_settling(t, ctx, p, state, ref):
    elapsed = t - ctx.start_phase_time
    SETTLING_DURATION = p.t_attitude_1 
    
    ref['pos'] = ctx.ref_snap_pos
    ref['vel'] = np.zeros(3)
    ref['acc'] = np.zeros(3)
    ref['yaw'] = ctx.last_valid_yaw
    
    # Parametro configurato dall'utente
    lambda_aero_eff = getattr(ctx, 'lambda_aero_effective', 0.0)
    config_blend = lambda_aero_eff
    
    # Inizializzazione assetto di partenza
    if not hasattr(ctx, 'att_settle_start') or ctx.att_settle_start is None:
        if hasattr(ctx, 'transition_att') and ctx.transition_att is not None:
            ctx.att_settle_start = ctx.transition_att
        else:
            ctx.att_settle_start = (0.0, 0.0)
    
    start_r, start_p = ctx.att_settle_start
    
    # Calcolo progress (Quintic curve)
    progress = elapsed / SETTLING_DURATION
    tau = np.clip(progress, 0, 1)
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    
    # --- LOGICA BIFORCATA ---
    if config_blend > 0.5:
        # MODO DINAMICO: Rampa Aero Blend 0 -> 1
        ctx.current_aero_blend = 0.0 + (1.0 - 0.0) * s
        
        # Assetto geometrico va a 0 (Aero domina)
        curr_r = start_r * (1 - s)
        curr_p = start_p * (1 - s)
        
        
    else:
        # MODO CLASSICO: Rampa Assetto, Aero Blend fisso a 0
        ctx.current_aero_blend = 0.0
        
        # >>> NUOVA LOGICA: Trasforma angoli dal frame "Traj" al frame corrente <<<
        from mission import rotate_attitude_to_yaw
        
        # 1. DEFINIAMO IL RIFERIMENTO ASSOLUTO
        # Questi angoli sono definiti rispetto a p.traj_target_yaw (es. Nord)
        # e NON rispetto al naso del drone.
        target_roll_ref = p.traj_target_roll
        target_pitch_ref = p.traj_target_pitch
        yaw_ref_frame = p.traj_target_yaw  # <--- MODIFICA QUI (Era p.final_target_yaw)
        
        # 2. PRENDIAMO LO YAW ATTUALE DELLA FORMAZIONE
        # Questo è dove i droni stanno effettivamente guardando (dovuto a vento o tracking)
        yaw_current = ref['yaw']
        
        # 3. TRASFORMAZIONE COORDINATE
        # Calcola quale Roll/Pitch deve avere il drone ORA (con il suo yaw_current)
        # per ottenere l'inclinazione assoluta desiderata rispetto a yaw_ref_frame.
        target_roll, target_pitch = rotate_attitude_to_yaw(
            target_roll_ref, target_pitch_ref, yaw_ref_frame, yaw_current
        )
        
        # Interpola verso i valori trasformati
        curr_r = start_r + (target_roll - start_r) * s
        curr_p = start_p + (target_pitch - start_p) * s

    ctx.transition_att = (curr_r, curr_p)
    
    if elapsed >= SETTLING_DURATION + 1.0:
        # Fine fase: Assicuriamo i valori finali corretti
        if config_blend > 0.5:
            ctx.current_aero_blend = 1.0
            ctx.transition_att = (0.0, 0.0)
            
            # --- AGGIUNGI QUESTA RIGA QUI SOTTO ---
            ctx.ref_snap_yaw = ctx.last_valid_yaw  
            # --------------------------------------

        else:
            ctx.current_aero_blend = 0.0
            from mission import rotate_attitude_to_yaw
            final_r, final_p = rotate_attitude_to_yaw(
                p.traj_target_roll, p.traj_target_pitch, 
                p.traj_target_yaw, ctx.last_valid_yaw
            )
            ctx.transition_att = (final_r, final_p)
            
        return True