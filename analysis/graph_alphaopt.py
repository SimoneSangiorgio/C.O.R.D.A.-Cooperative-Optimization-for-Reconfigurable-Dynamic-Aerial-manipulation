import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Add parent directory to path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import formation as formation
from parameters import SysParams

# =============================================================================
# 1. BASE SETTINGS AND WEIGHT VARIATIONS
# =============================================================================

# Your current base values (as from graph_cost.py)
base_weights = {
    'w_T': 2.0,
    'w_ref': 80.0,
    'w_barrier': 5.5,
    'w_cond': 1.0,
    'w_resid_f': 7000.0,
    'w_resid_m': 3000.0
}

# Define the 3 levels (Low, Medium/Base, High) to test for each parameter
# The second value always corresponds to your default value.
variations = {
    'w_T':       [0.002,   2.0,   500.0],
    'w_ref':     [0.08,   80.0,    8000.0],
    'w_barrier': [0.5,    5.5,     500.0],
    'w_cond':    [0.001,    1.0,     10000.0],
    'w_resid_f': [7.0, 7000.0,  700000.0],
    'w_resid_m': [3.0,  3000.0,  10000.0]
}

labels = ['Low', 'Medium (Base)', 'High']
colors = ['#2ca02c', '#1f77b4', '#d62728']  # Green (Low), Blue (Medium), Red (High)

forces_wind_N = np.linspace(0, 35, 100)

# =============================================================================
# 2. PLOT GRID CREATION
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# =============================================================================
# 3. SIMULATION AND PLOTTING LOOP
# =============================================================================
for idx, weight_name in enumerate(base_weights.keys()):
    ax = axes[idx]
    
    # For each value (low, medium, high) of this weight
    for v_idx, test_value in enumerate(variations[weight_name]):
        
        # Initialize parameters
        p = SysParams()
        
        # Base configurations for this static analysis
        p.theta_ref = 30.0
        p.N = 4
        p.w_smooth = 0.0  # Disabled for static analysis
        p.min_drone_z_rel = -10.0
        p.lambda_aero = 0
        p.lambda_shape = 1
        p.lambda_par = 2.0
        p.lambda_perp = 0.5
        p.max_angle_variation = 360.0
        
        # Apply ALL base weights
        for bw_name, bw_val in base_weights.items():
            setattr(p, bw_name, bw_val)
            
        # OVERWRITE the current weight under test with the test_value
        setattr(p, weight_name, test_value)
        
        # Recompute base geometry
        p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
        
        # Initialize context cleanly for each curve
        class Context: pass
        ctx = Context()
        ctx.last_theta_opt = np.radians(p.theta_ref)
        ctx.last_stable_yaw = 0.0
        
        m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)
        alpha_opts_deg = []
        
        # Run static wind sweep
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

            # Call the optimizer
            pos, L, a_opt, ff = formation.compute_optimal_formation(
                p, state, np.zeros(3), np.zeros(3), 0.0,
                force_attitude=(0.0, theta_eq),
                F_ext_total=F_ext,
                com_offset_body=p.CoM_offset,
                ctx=ctx
            )
            alpha_opts_deg.append(np.degrees(a_opt))
            
        # Plot the curve
        ax.plot(forces_wind_N, alpha_opts_deg, color=colors[v_idx], linewidth=2.5, 
                label=f"{labels[v_idx]} ({test_value})")
        
    # Format individual subplot
    ax.axhline(p.theta_ref, color='black', linestyle=':', label='Reference (30°)', alpha=0.7)
    ax.set_title(f"Impact of weight: '{weight_name}'", fontsize=14, fontweight='bold', color='black')
    ax.set_xlabel('Horizontal Force [N]', fontsize=11)
    ax.set_ylabel(r'Cone Angle $\alpha_{opt}$ [Degrees]', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.show()