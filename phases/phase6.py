### HOLD ###

import numpy as np

def run_hold(t, ctx, p, state, ref):
    if ctx.final_ref_pos is not None: ref['pos'] = ctx.final_ref_pos
    else: ref['pos'] = p.payload_goal_pos
    
    if ctx.final_yaw is not None:
        ref['yaw'] = ctx.final_yaw
    else:
        ref['yaw'] = ctx.last_valid_yaw
        
    # Forza l'assetto finale costantemente
    ctx.transition_att = (p.final_target_roll, p.final_target_pitch)
        
    ref['vel'] = np.zeros(3)
    ref['acc'] = np.zeros(3)