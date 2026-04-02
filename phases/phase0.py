### PRETENSION - VIRTUAL FLIGHT STATE ###

import numpy as np
import formation as formation

def run(t, ctx, p, state, ref):
    # Riferimento Payload fermo a terra
    # NOTA: Usiamo p.pay_h/2 per assicurarci che il riferimento sia al centro geometrico
    ref['pos'] = np.array([p.home_pos[0], p.home_pos[1], p.home_pos[2] + p.pay_h/2]) 
    ref['vel'] = np.zeros(3)
    ref['acc'] = np.zeros(3)

    # Calcolo YAW ottimale per vento
    ref['yaw'] = 0.0
    ctx.last_valid_yaw = ref['yaw']  
        
    # Parametri temporali
    T_IDLE = getattr(p, 't_idle', 1.0)
    T_PRETENSION = getattr(p, 't_pretension', 5.0)
    T_SETTLE = 1.0 # Tempo extra per stabilizzare la tensione asimmetrica
    
    total_phase_duration = T_IDLE + T_PRETENSION + T_SETTLE
    
    # --- SETUP INIZIALE ---
    # Calcoliamo gli offset iniziali reali basati sulla posizione corrente dei droni a terra
    if not hasattr(ctx, 'pretension_start_offsets') or ctx.pretension_start_offsets is None:
        real_start = np.zeros((3, p.N))
        for k in range(p.N):
            real_start[:, k] = state['uav_pos'][:, k] - ref['pos']
        ctx.pretension_start_offsets = real_start

    start_targets = ctx.pretension_start_offsets
    
    # Default fallback values
    end_targets = p.uav_offsets
    end_L = np.ones(p.N) * p.L

    # --- GESTIONE MOTORI ---
    if t < T_IDLE:
        ctx.motors_idle = True
        blend = 0.0
    else:
        ctx.motors_idle = False
        run_time = t - T_IDLE
        
        # Calcolo Blend (Quintic Ramp) per movimento fluido
        if run_time < T_PRETENSION:
            tau = run_time / T_PRETENSION
            blend = 10*tau**3 - 15*tau**4 + 6*tau**5
        else:
            blend = 1.0 # Mantiene la posizione finale durante il T_SETTLE

        # =========================================================================
        # TRUCCO "VIRTUAL FLIGHT STATE" (Robust CoM Handling)
        # =========================================================================
        # Creiamo uno stato fittizio per l'ottimizzatore come se il carico stesse volando.
        # Questo permette di calcolare la geometria esatta per compensare il CoM.
        
        flight_state = state.copy()
        
        # 1. Simula Quota di Volo di sicurezza
        # Solleviamo virtualmente il payload per calcolare la geometria dei cavi in tensione
        target_z = p.home_pos[2] + getattr(p, 'safe_altitude_offset', 3.0)
        flight_state['pay_pos'] = np.array([ref['pos'][0], ref['pos'][1], target_z])
        
        # 2. Azzera velocità e rotazioni (Hovering statico ideale)
        flight_state['pay_vel'] = np.zeros(3)
        flight_state['pay_omega'] = np.zeros(3)
        # IMPORTANTE: Forziamo l'assetto piatto nello stato virtuale
        flight_state['pay_att'] = np.zeros(3) 
        
        # 3. Gestione Dinamica del CoM (Sloshing & Offset)
        base_offset = getattr(p, 'CoM_offset', np.zeros(3)).copy()
        
        if getattr(p, 'enable_sloshing', False):
            amp = getattr(p, 'slosh_amp', 0.1)
            freq = getattr(p, 'slosh_freq', 0.1)
            omega = 2.0 * np.pi * freq
            
            m_tot_p0 = p.m_payload + p.m_liquid
            # Rapporto di massa per sloshing
            if p.m_payload > 1e-6:
                mass_ratio = p.m_liquid / m_tot_p0
            else:
                mass_ratio = 0.0
            
            # Calcolo posizione liquido al tempo t
            liq_dx = amp * np.sin(omega * t)
            liq_dy = amp * np.cos(omega * t)
            
            # Trasferimento al CoM del sistema
            com_offset_body = base_offset + mass_ratio * np.array([liq_dx, liq_dy, 0.0])
            
            # Velocità e accelerazione del CoM (Feedforward)
            liq_vx = amp * omega * np.cos(omega * t)
            liq_vy = -amp * omega * np.sin(omega * t)
            com_vel_body = mass_ratio * np.array([liq_vx, liq_vy, 0.0])
            
            liq_ax = -amp * (omega**2) * np.sin(omega * t)
            liq_ay = -amp * (omega**2) * np.cos(omega * t)
            com_acc_body = mass_ratio * np.array([liq_ax, liq_ay, 0.0])
            
        else:
            com_offset_body = base_offset
            com_vel_body = np.zeros(3)
            com_acc_body = np.zeros(3)

        # 4. Calcolo Vento Virtuale
        wind_vec = p.wind_vel 
        wind_mag = np.linalg.norm(wind_vec)
        
        if wind_mag < 1e-3:
            F_wind_virtual = np.zeros(3)
        else:
            wind_dir = wind_vec / wind_mag
            if hasattr(p, 'payload_shape') and p.payload_shape in ['box', 'rect', 'square']:
                 ref_area = p.pay_l * p.pay_h 
            else:
                 ref_area = 2.0 * p.R_disk * p.pay_h
            # Forza del vento statica stimata
            F_wind_virtual = 0.5 * p.rho * p.Cd_pay * ref_area * (wind_mag**2) * wind_dir

        # 5. CHIAMATA OTTIMIZZATORE
        try:
            # Forziamo aero_blend=0.0 per ignorare l'inclinazione aerodinamica a terra
            # Vogliamo solo compensare il CoM traslando i droni.
            target_aero = 0.0 

            e_slosh_body = np.array([liq_dx, liq_dy, 0.0]) if getattr(p, 'enable_sloshing', False) else np.zeros(3)
            
            opt_targets, opt_L, opt_theta, opt_ff = formation.compute_optimal_formation(
                p, flight_state,
                acc_cmd_pay=np.zeros(3),
                acc_ang_cmd_pay=np.zeros(3),
                ref_yaw=0.0,
                force_attitude=(0.0, 0.0), 
                F_ext_total=F_wind_virtual,
                ctx=ctx,
                aero_blend_override=target_aero,
                com_offset_body=com_offset_body,
                com_vel_body=com_vel_body,
                com_acc_body=com_acc_body,
                e_slosh_body=e_slosh_body,
                F_aero_for_moment=F_wind_virtual
            )
            
            if opt_targets is not None:
                end_targets = opt_targets
                end_L = opt_L
                
                # Salviamo nel contesto per il debug o usi futuri
                ctx.hover_config_targets = opt_targets
                ctx.hover_config_L = opt_L
                ctx.hover_config_ff = opt_ff
                ctx.hover_theta_opt = opt_theta
                
        except Exception as e:
            # Fallback in caso di errore numerico
            print(f"Phase0 Opt Error: {e}")
            end_targets = p.uav_offsets
            end_L = np.ones(p.N) * p.L

    # =========================================================================
    # APPLICAZIONE E INTERPOLAZIONE
    # =========================================================================
    
    current_targets = np.zeros((3, p.N))
    current_L = np.zeros(p.N)
    ctx.custom_feedforward = np.zeros((3, p.N))
    
    for k in range(p.N):
        # Interpolazione della posizione (Sposta i droni sopra il CoM reale)
        current_targets[:, k] = start_targets[:, k] + blend * (end_targets[:, k] - start_targets[:, k])
        
        # Interpolazione della lunghezza (Accorcia/Allunga i cavi per compensare il peso asimmetrico)
        # Se start_L non è noto, assumiamo p.L. 
        # end_L contiene la compensazione elastica (es. cavo più corto dove c'è più peso)
        current_L[k] = p.L + blend * (end_L[k] - p.L)
    
    ctx.custom_uav_targets = current_targets
    ctx.current_L_rest = current_L
    
    # --- CHECK FINE FASE ---
    is_finished = False
    
    # Attendiamo la fine di T_PRETENSION + T_SETTLE
    if not ctx.motors_idle and (t - T_IDLE) >= (T_PRETENSION + T_SETTLE):
        is_finished = True
        
        # Salvataggio offset finale per garantire continuità perfetta in Phase 1
        ctx.initial_phase1_offsets = end_targets.copy()
        
        # Inizializza i filtri della Phase 1 con i valori ottimizzati attuali
        # Questo evita che il filtro "resetti" la posizione causando uno scatto
        ctx.filt_targets = end_targets.copy()
        ctx.filt_L_winch = end_L.copy()
        ctx.filt_ff = np.zeros((3, p.N)) 
        ctx.opt_filter_init = True
        
    return is_finished