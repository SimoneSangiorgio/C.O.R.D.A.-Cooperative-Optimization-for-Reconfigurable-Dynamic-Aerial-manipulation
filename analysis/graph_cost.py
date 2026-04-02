import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import formation as formation
from parameters import SysParams

# =============================================================================
# 1. HOOK TO CAPTURE DATA FROM THE OPTIMIZER
# =============================================================================
original_solve_tensions = formation.solve_tensions
hook_data = {}

def hooked_solve_tensions(A, W_req, T_min_safe, T_max_limit):
    res = original_solve_tensions(A, W_req, T_min_safe, T_max_limit)
    hook_data['A'] = A
    hook_data['W_req'] = W_req
    hook_data['T_min_safe'] = T_min_safe
    hook_data['T_max_limit'] = T_max_limit
    hook_data['T_opt'] = res[0]
    return res

formation.solve_tensions = hooked_solve_tensions

# =============================================================================
# 2. FUNCTION TO RECONSTRUCT COSTS AT THE OPTIMUM
# =============================================================================
def compute_cost_components(p, hook_data, alpha_opt, alpha_prev):
    A = hook_data['A']
    W_req = hook_data['W_req']
    T_opt = hook_data['T_opt']
    T_safe = hook_data['T_min_safe']
    T_limit = hook_data['T_max_limit']

    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)
    F_nom = max(m_tot * p.g, 1.0)
    arm_nom = getattr(p, 'pay_l', 1.0) * 0.5
    M_nom = max(F_nom * arm_nom, 0.1)

    # 1. Calculate RAW components
    try:
        cond_num = min(np.linalg.cond(A), 1000)
    except:
        cond_num = 1000
    J_cond = np.log10(cond_num) / 2.0

    # 1. Calculate RAW components
    try:
        cond_num = min(np.linalg.cond(A), 1000)
    except:
        cond_num = 1000
    J_cond = np.log10(cond_num) / 2.0

    J_barrier = 0.0
    scale_barrier = 100.0
    
    # [FIX] Barrier margin protection identical to formation_coupled.py
    T_alert_low = min(T_safe * 1.5, T_limit * 0.4) 
    T_alert_high = max(T_limit * 0.85, T_alert_low + 0.1)
    
    k_sharp = 0.7
    for t_val in T_opt:
        x_low = T_alert_low - t_val
        range_low = T_alert_low - T_safe
        if range_low > 0:
            soft_low = np.log(1.0 + np.exp(k_sharp * x_low)) / k_sharp
            J_barrier += scale_barrier * (soft_low / range_low)**2

        x_high = t_val - T_alert_high
        range_high = T_limit - T_alert_high
        if range_high > 0:
            soft_high = np.log(1.0 + np.exp(k_sharp * x_high)) / k_sharp
            J_barrier += scale_barrier * (soft_high / range_high)**2

    T_max = np.max(T_opt)
    J_T = T_max / T_limit

    M_req_norm = np.linalg.norm(W_req[3:])
    M_nom_rot = 2.0
    theta_boost = np.radians(10.0) * np.clip(M_req_norm / M_nom_rot, 0.0, 1.0)
    theta_ref_dyn = np.radians(p.theta_ref) + theta_boost
    J_ref = ((alpha_opt - theta_ref_dyn)**2) / (np.radians(20.0)**2)

    J_smooth = ((alpha_opt - alpha_prev)**2) / (np.radians(3.5)**2)

    W_achieved = A @ T_opt
    diff_W = W_achieved - W_req
    J_resid_f = np.sum(diff_W[:3]**2) / (F_nom**2)
    J_resid_m = np.sum(diff_W[3:]**2) / (M_nom**2)

    # 2. Application of WEIGHTS (w_i * J_i)
    wJ_T = p.w_T * J_T
    wJ_barrier = p.w_barrier * J_barrier
    wJ_ref = p.w_ref * J_ref
    wJ_smooth = p.w_smooth * J_smooth
    wJ_cond = p.w_cond * J_cond
    wJ_resid_f = p.w_resid_f * J_resid_f
    wJ_resid_m = p.w_resid_m * J_resid_m

    # 3. Calculation of the TOTAL COST
    J_tot = wJ_T + wJ_barrier + wJ_ref + wJ_smooth + wJ_cond + wJ_resid_f + wJ_resid_m

    return {
        'J_T (Tensions)': wJ_T,
        'J_barrier (Safety)': wJ_barrier,
        'J_ref (Reference)': wJ_ref,
        'J_smooth (Smoothness)': wJ_smooth,
        'J_cond (Geometry)': wJ_cond,
        'J_resid_f (Force)': wJ_resid_f,
        'J_resid_m (Moment)': wJ_resid_m,
        'J_TOTAL (Total Cost)': J_tot
    }

# =============================================================================
# 3. CONFIGURATION AND SIMULATION
# =============================================================================
p = SysParams()

# =============================================================================
# ---> MODIFY WEIGHTS HERE <---
# Try setting w_cond or w_barrier to 0.0: you will see their line flatten 
# perfectly on zero, and J_TOTAL will not take it into account.
# =============================================================================
p.theta_ref       = 30.0
p.N = 4

#p.w_T       = 2.0     
#p.w_ref     = 80.0    # LOWERED: the formation opens more easily to the wind
#p.w_smooth  = 30.0    # We keep it at 0 for static analysis
#p.w_barrier = 5.5     # RAISED: mathematical "cushion" preventing slack cables
#p.w_cond    = 1.0     # A minimum of conditioning is always good
#p.w_resid_f = 7000.0  
#p.w_resid_m = 3000.0


p.w_T       = 2.0     
p.w_ref     = 100.0    # LOWERED: the formation opens more easily to the wind
p.w_smooth  = 10.0    # We keep it at 0 for static analysis
p.w_barrier = 0.3    # RAISED: mathematical "cushion" preventing slack cables
p.w_cond    = 1.0     # A minimum of conditioning is always good
p.w_resid_f = 10000.0  
p.w_resid_m = 1000.0

p.min_drone_z_rel = -10.0
p.lambda_aero = 0
p.lambda_shape = 1
p.lambda_par = 2.0
p.lambda_perp = 0.5
p.max_angle_variation = 360.0
# =============================================================================

p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)

forces_wind_N = np.linspace(0, 35, 100)

alpha_opts_deg = []
alpha_refs_deg = []
costs_history = {
    'J_T (Tensions)': [], 'J_barrier (Safety)': [], 'J_ref (Reference)': [], 
    'J_smooth (Smoothness)': [], 'J_cond (Geometry)': [], 'J_resid_f (Force)': [], 
    'J_resid_m (Moment)': [], 'J_TOTAL (Total Cost)': []
}

class Context: pass
ctx = Context()
ctx.last_theta_opt = np.radians(p.theta_ref)
ctx.last_stable_yaw = 0.0

m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)

for F_N in forces_wind_N:
    F_ext = np.array([F_N, 0.0, 0.0])
    theta_eq = np.arctan2(F_N, m_tot * p.g)
    
    state = {
        'pay_pos': np.zeros(3),
        'pay_att': np.array([0.0, theta_eq, 0.0]),
        'pay_vel': np.zeros(3),
        'pay_omega': np.zeros(3),
        'uav_vel': np.zeros((3, p.N))
    }

    alpha_prev = ctx.last_theta_opt

    pos, L, a_opt, ff = formation.compute_optimal_formation(
        p, state, np.zeros(3), np.zeros(3), 0.0,
        force_attitude=(0.0, theta_eq),
        F_ext_total=F_ext,
        com_offset_body=p.CoM_offset,
        ctx=ctx# <-- NO TEMPORAL MEMORY
    )

    alpha_opts_deg.append(np.degrees(a_opt))
    
    M_req = hook_data['W_req'][3:]
    M_req_norm = np.linalg.norm(M_req)
    theta_boost = np.radians(10.0) * np.clip(M_req_norm / 2.0, 0.0, 1.0)
    a_ref_dyn = np.radians(p.theta_ref) + theta_boost
    alpha_refs_deg.append(np.degrees(a_ref_dyn))

    costs = compute_cost_components(p, hook_data, a_opt, alpha_prev)
    for k in costs_history:
        # Save the exact value. No clamp to 1e-6 thanks to 'symlog'
        costs_history[k].append(costs[k])

formation.solve_tensions = original_solve_tensions

# =============================================================================
# 4. PLOT CREATION
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Angle Evolution ---
ax1.plot(forces_wind_N, alpha_opts_deg, label=r'$\alpha_{opt}$', color='navy', linewidth=2.5)
#ax1.plot(forces_wind_N, alpha_refs_deg, label=r'$\alpha_{ref}$', color='crimson', linestyle='--', linewidth=2)
ax1.axhline(p.theta_ref, label=r'$\alpha_{ref}$', color='darkgreen', linestyle=':', linewidth=2)

ax1.set_xlabel('Horizontal Force [N]', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cone Angle [Degrees]', fontsize=12, fontweight='bold')
ax1.set_title(r'Variation of $\alpha_{opt}$', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)

# --- Plot 2: Weighted Cost Contributions ---
colors = plt.cm.tab10(np.linspace(0, 1, len(costs_history) - 1)) # -1 because TOTAL has a fixed color

for idx, (nome_costo, valori) in enumerate(costs_history.items()):
    if nome_costo == 'J_TOTAL (Total Cost)':
        # J_TOTAL in black and thick dashed to stand out
        ax2.plot(forces_wind_N, valori, label=nome_costo, color='black', linewidth=3, linestyle='--')
    else:
        # Turn off visualization of constants that remain fixed at 0 (to not clutter the legend)
        if max(valori) > 0.0:
            ax2.plot(forces_wind_N, valori, label=nome_costo, color=colors[idx], linewidth=2.5)

ax2.set_xlabel('Horizontal Force [N]', fontsize=12, fontweight='bold')
ax2.set_ylabel('Weighted Cost Value ($w_i \cdot J_i$)', fontsize=12, fontweight='bold')
ax2.set_title("Contributions and Total Cost $J_{tot}$", fontsize=14, fontweight='bold')

# We use 'symlog' (symmetrical log) to perfectly support the mathematical zero
ax2.set_yscale('symlog', linthresh=1e-2)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()