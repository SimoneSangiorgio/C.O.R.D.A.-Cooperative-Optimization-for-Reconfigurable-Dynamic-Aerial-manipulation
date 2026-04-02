import numpy as np

def compute_controls(s, ref, p, ctx, t):
    u_acc = np.zeros((3, p.N))
    
    d_int_uav = np.zeros((3, p.N))
    d_int_pay = np.zeros(3)
    
    acc_traj = ref['acc']
    is_idle = getattr(ctx, 'motors_idle', False)
    
    # Parametri
    v_pay = s['pay_vel']
    v_error_pay = ref['vel'] - v_pay
    k_active_damp = getattr(p, 'k_active_damp', 2.0)
    alpha_motor = getattr(p, 'alpha_motor_lag', 0.15) 
    t_idle_end = getattr(p, 't_idle', 1.0)
    
    # Vettore gravità per calcoli
    g_vec = np.array([0, 0, p.g])

    for i in range(p.N):
        # --- 1. IDLE STATE (Motori Spenti) ---
        if is_idle:
            # Output -g significa Thrust = m(-g + g) = 0
            u_acc[:, i] = -g_vec 
            # Reset integrale per evitare accumulo
            d_int_uav[:, i] = -5.0 * s['int_uav'][:, i] 
            ctx.last_u_acc[:, i] = u_acc[:, i]
            
            # Inizializza filtro se non esiste
            if not hasattr(ctx, 'filt_u_acc'):
                ctx.filt_u_acc = np.zeros((3, p.N))
            ctx.filt_u_acc[:, i] = u_acc[:, i]
            continue 

        # --- 2. ACTIVE STATE (Calcolo PID) ---
        
        # Target Posizione
        if hasattr(ctx, 'custom_uav_targets') and ctx.custom_uav_targets is not None:
             pos_des = ref['pos'] + ctx.custom_uav_targets[:, i]
        else:
             pos_des = ref['pos'] + p.uav_offsets[:, i]
        
        vel_des = ref['vel'] 
        
        # Errori
        e_p = pos_des - s['uav_pos'][:, i]
        e_v = vel_des - s['uav_vel'][:, i]
        curr_int = s['int_uav'][:, i]
        
        # Gestione Integrale (Anti-Windup base)
        d_int_uav[:, i] = e_p 
        k_int_ramp = 1.0
        
        # --- MODIFICA PER TRANSIZIONE DOLCE ---
        if ctx.phase == 0:
             # Se i motori sono in IDLE (spenti), resetta l'integrale per evitare accumuli spuri
             if getattr(ctx, 'motors_idle', True):
                 d_int_uav[:, i] = -1.0 * curr_int
                 k_int_ramp = 0.0
             else:
                 # Se siamo in PRETENSION (motori attivi), ABILITA l'integrale!
                 # Questo permette al PID di accumulare la forza necessaria a 
                 # contrastare il vento PRIMA che il carico si stacchi da terra.
                 k_int_ramp = 1.0
        
        # Feedforward
        FF_custom = np.zeros(3)
        if hasattr(ctx, 'custom_feedforward') and ctx.custom_feedforward is not None:
            FF_val = ctx.custom_feedforward[:, i] / p.m_drone
            FF_custom = FF_val

        if ctx.phase >= 2:
             FF_term = FF_custom # L'ottimizzatore sa tutto
        else:
             FF_term = acc_traj + FF_custom

        # Calcolo PID
        P_term = p.kp_pos * e_p
        D_term = p.kv_pos * e_v
        I_term = p.ki_pos * curr_int * k_int_ramp 

        
        k_damp_z = getattr(p, 'k_active_damp_z', 0.5)  # Guadagno ridotto: il Z è più sensibile
        D_active = k_active_damp * np.array([v_error_pay[0], v_error_pay[1], 0.0])
        D_active[2] = k_damp_z * v_error_pay[2]
        
        cmd = P_term + D_term + I_term + FF_term + D_active

        # --- 3. MOTOR SOFT START LOGIC (SOLUZIONE SCATTO) ---
        # Applica una rampa sigmoidale all'uscita del controller subito dopo l'Idle.
        # Questo sovrascrive qualsiasi comando brusco del PID.
        
        ramp_duration = 2.0 # Secondi di rampa per accensione morbida
        
        if t >= t_idle_end:
            dt_ramp = t - t_idle_end
            if dt_ramp < ramp_duration:
                # Progresso da 0.0 a 1.0
                prog = dt_ramp / ramp_duration
                # Curva sigmoidale smooth (3x^2 - 2x^3)
                scale_factor = 3*prog**2 - 2*prog**3
                
                # Formula Magica:
                # Vogliamo scalare la SPINTA (Thrust), non l'accelerazione matematica.
                # Thrust_desiderata = m * (cmd + g)
                # Thrust_scalata = Thrust_desiderata * scale_factor
                # m * (cmd_finale + g) = m * (cmd + g) * scale_factor
                # cmd_finale = (cmd + g) * scale_factor - g
                
                cmd = (cmd + g_vec) * scale_factor - g_vec
    
        # Saturazione Fisica (Thrust Max)
        max_acc_phys = (p.F_max_thrust / p.m_drone) 
        
        # Consideriamo la spinta totale richiesta (comando + gravità)
        total_acc_req = cmd + g_vec
        norm_total = np.linalg.norm(total_acc_req)
        
        if norm_total > max_acc_phys:
            # Scaliamo mantenendo la direzione
            total_acc_clamped = (total_acc_req / norm_total) * max_acc_phys
            # Torniamo in coordinate di accelerazione comando
            cmd = total_acc_clamped - g_vec

        if not hasattr(ctx, 'filt_u_acc'):
            ctx.filt_u_acc = np.zeros((3, p.N))
        ctx.filt_u_acc[:, i] = (1.0 - alpha_motor) * ctx.filt_u_acc[:, i] + alpha_motor * cmd
        u_acc[:, i] = ctx.filt_u_acc[:, i]
            
        ctx.last_u_acc[:, i] = u_acc[:, i]

    return u_acc, {'d_int_uav': d_int_uav, 'd_int_pay': d_int_pay}