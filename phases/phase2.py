import numpy as np

def run_alignment(t, ctx, p, state, ref):
    elapsed = t - ctx.start_phase_time
    T_ALIGNMENT = p.t_alignment
    
    # 1. Mantenimento Posizione (Hover sul posto)
    ref['pos'] = ctx.ref_snap_pos 
    ref['vel'] = np.zeros(3)
    ref['acc'] = np.zeros(3)
    
    lambda_traj_eff = getattr(ctx, 'lambda_traj_effective', 0.0)

    # 2. Determinazione Target Yaw secondo gerarchia
    if lambda_traj_eff > 0.5:
        # MODO PHYSICS: Segue l'ottimizzatore (Vento/Forze) - Priorità Assoluta
        target_yaw = getattr(ctx, 'last_stable_yaw', ctx.ref_snap_yaw)
    else:
        if getattr(p, 'yaw_tracking_enabled', False):
            # MODO TRACKING: Allinea al goal finale (Coordinate Turn)
            diff = p.payload_goal_pos[:2] - p.home_pos[:2]
            dist_xy = np.linalg.norm(diff)
            
            if dist_xy > 1.0:
                psi_traj = np.arctan2(diff[1], diff[0])
                
                # Trova orientamento con minima rotazione per simmetria droni
                min_yaw = np.inf
                for k in range(p.N):
                    candidate = psi_traj - k*(2*np.pi/p.N)
                    candidate = np.arctan2(np.sin(candidate), np.cos(candidate))
                    diff_angle = np.arctan2(np.sin(candidate - ctx.ref_snap_yaw), np.cos(candidate - ctx.ref_snap_yaw))
                    
                    if abs(diff_angle) < abs(min_yaw): 
                        min_yaw = diff_angle
                        target_yaw = candidate
            else:
                target_yaw = ctx.ref_snap_yaw
        else:
            # MODO FIXED: Usa il parametro statico traj_target_yaw
            target_yaw = getattr(p, 'traj_target_yaw', 0.0)

    # 3. Applicazione Riferimento (Rimosso ritardo in Modo Physics)
    if lambda_traj_eff > 0.5:
        # Nessuna interpolazione: seguiamo istantaneamente l'ottimizzatore per evitare il lag di 10°
        ref['yaw'] = target_yaw
    else:
        # Interpolazione Smoothstep solo per i modi geometrici/fissi
        tau = min(1.0, max(0.0, elapsed / T_ALIGNMENT))
        s_rot = 3*tau**2 - 2*tau**3 
        
        delta = np.arctan2(np.sin(target_yaw - ctx.ref_snap_yaw), np.cos(target_yaw - ctx.ref_snap_yaw))
        ref['yaw'] = ctx.ref_snap_yaw + s_rot * delta
    
    ctx.last_valid_yaw = ref['yaw']
    
    if elapsed > T_ALIGNMENT + 1.0:
        return True
        
    return False