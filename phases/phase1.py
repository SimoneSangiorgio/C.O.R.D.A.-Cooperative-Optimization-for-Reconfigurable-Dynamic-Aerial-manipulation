### LIFTOFF ###

import numpy as np

def get_quintic_traj_1d(t, T, p0, p1):
    if t < 0: return p0, 0, 0
    if t >= T: return p1, 0, 0
    tau = t / T
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    sd = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    sdd = (60*tau - 180*tau**2 + 120*tau**3) / T**2
    return p0 + (p1 - p0)*s, (p1 - p0)*sd, (p1 - p0)*sdd

# --- FUNZIONE 1: LIFTOFF (Solo Salita) ---
def run_liftoff(t, ctx, p, state, ref):
    LIFT_HEIGHT = 1 + max(p.pay_l, p.pay_h, p.pay_w, p.R_disk)
    T_LIFT = p.t_lift
    
    # 0. Wait time iniziale per far assestare i feedforward (opzionale)
    WAIT_FOR_FF = 0.5 
    
    elapsed = t - ctx.start_phase_time
    total_lift_time = WAIT_FOR_FF + T_LIFT
    
    # 1. Setup posizioni start/end (Usa la posizione REALE di scatto)
    if ctx.ref_snap_pos is not None:
        start_pos_z = ctx.ref_snap_pos[2]
        base_x = ctx.ref_snap_pos[0] 
        base_y = ctx.ref_snap_pos[1]
    else:
        start_pos_z = p.safe_altitude_offset 
        base_x = p.home_pos[0]
        base_y = p.home_pos[1]
        
    target_pos_z = p.home_pos[2] + p.safe_altitude_offset + LIFT_HEIGHT
    
    # 2. Calcolo Traiettoria
    if elapsed < WAIT_FOR_FF:
        # HOLD iniziale dolce (mantiene lo stato fine fase 0)
        z_curr = start_pos_z
        z_vel = 0
        z_acc = 0
    else:
        traj_time = elapsed - WAIT_FOR_FF
        z_curr, z_vel, z_acc = get_quintic_traj_1d(traj_time, T_LIFT, start_pos_z, target_pos_z)
    
    # 3. Aggiornamento Reference
    ref['pos'][0] = base_x
    ref['pos'][1] = base_y
    ref['pos'][2] = z_curr
    
    # Iniettiamo una piccola componente di smorzamento attivo nel riferimento se necessario
    # Ma il PID se ne occupa già con k_active_damp.
    
    ref['vel'] = np.array([0, 0, z_vel])
    ref['acc'] = np.array([0, 0, z_acc])
    
    lambda_traj = getattr(ctx, 'lambda_traj_effective', 0.0)

    if not hasattr(ctx, 'phase1_yaw_start') or ctx.phase1_yaw_start is None:
        ctx.phase1_yaw_start = ctx.last_valid_yaw

    # Calcola target yaw (gerarchia esistente)
    if lambda_traj > 0.5:
        target_yaw = getattr(ctx, 'last_stable_yaw', ctx.ref_snap_yaw)
    elif getattr(p, 'yaw_tracking_enabled', False):
        diff = p.payload_goal_pos[:2] - p.home_pos[:2]
        target_yaw = np.arctan2(diff[1], diff[0])
    else:
        target_yaw = getattr(p, 'traj_target_yaw', 0.0)

    # Interpolazione quintica su tutta la durata del liftoff
    tau = np.clip(elapsed / total_lift_time, 0.0, 1.0)
    s = 10*tau**3 - 15*tau**4 + 6*tau**5

    delta = np.arctan2(np.sin(target_yaw - ctx.phase1_yaw_start),
                       np.cos(target_yaw - ctx.phase1_yaw_start))
    ref['yaw'] = ctx.phase1_yaw_start + s * delta
    ctx.last_valid_yaw = ref['yaw']
    
    if elapsed >= total_lift_time:
        ctx.phase1_yaw_start = None
        ref['vel'] = np.zeros(3)
        ref['acc'] = np.zeros(3)
        return True
        
    return False