"""
graph_windo_dir.py
------------------
Static analysis: formation response to varying wind direction.

The payload is kept fixed at a given position while the wind direction
rotates from -180° to +180° (yaw sweep) at constant wind speed.
At each angle the C.O.R.D.A. optimizer is called and the following
metrics are recorded:

  1. Power index  Σ T_i^{1.5}  (Actuator-Disk Theory proxy for total power)
  2. Load imbalance: standard deviation of cable tensions
  3. Maximum and minimum cable tension (saturation / slack margins)

Physical note on the power index
---------------------------------
Each UAV must generate a thrust that:
  • counters its own weight:           +m_drone * g * ẑ
  • pulls the cable with tension T_i:  +T_i * û_{L,i}   (ff_forces[:,i])
  • overcomes aerodynamic drag:        -F_w_drone

So the net thrust vector for UAV i is:
    F_thrust_i = ff_forces[:,i] + m_drone*g*ẑ - F_w_drone

The power proxy follows Actuator-Disk Theory: P ∝ T^{3/2}.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from parameters import SysParams
import formation as formation
import physics
from mission import MissionContext


# ==============================================================================
# PHYSICAL HELPERS
# ==============================================================================

def get_wind_force_at_attitude(p, v_wind_world: np.ndarray,
                               attitude: tuple) -> np.ndarray:
    """
    Compute the aerodynamic drag force on the payload for a given wind vector
    and payload attitude (phi, theta, psi).

    The projected area is computed in the payload body frame, then the drag
    force is expressed back in the world frame.
    """
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    if wind_mag < 1e-3:
        return np.zeros(3)

    wind_dir      = v_wind_world / wind_mag
    R_pay         = physics.get_rotation_matrix(phi, theta, psi)
    wind_in_body  = R_pay.T @ wind_dir

    if getattr(p, 'payload_shape', '') in ['box', 'rect', 'square']:
        A_x = p.pay_w * p.pay_h
        A_y = p.pay_l * p.pay_h
        A_z = p.pay_l * p.pay_w
        proj_area = (A_x * abs(wind_in_body[0])
                   + A_y * abs(wind_in_body[1])
                   + A_z * abs(wind_in_body[2]))
    elif getattr(p, 'payload_shape', '') == 'sphere':
        proj_area = np.pi * p.R_disk ** 2          # constant for a sphere
    else:                                           # cylinder
        A_side    = 2.0 * p.R_disk * p.pay_h
        A_top     = np.pi * p.R_disk ** 2
        sin_tilt  = np.sqrt(wind_in_body[0] ** 2 + wind_in_body[1] ** 2)
        proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])

    f_wind = (0.5 * p.rho * p.Cd_pay * proj_area * wind_mag ** 2) * wind_dir
    return f_wind


def compute_aerodynamic_equilibrium(p, v_wind_world: np.ndarray,
                                    base_yaw: float = 0.0):
    """
    Iteratively find the payload roll/pitch equilibrium under aerodynamic drag.

    The iteration solves for (phi, theta) such that the cable resultant
    force is collinear with gravity + wind drag, expressed in the yaw-aligned
    body frame.  Convergence is typically reached in < 15 steps.

    Returns
    -------
    phi   : equilibrium roll  [rad]
    theta : equilibrium pitch [rad]
    f_wind: wind force at equilibrium [N, world frame]
    """
    phi, theta = 0.0, 0.0
    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)

    c_psi, s_psi = np.cos(base_yaw), np.sin(base_yaw)
    R_z_inv = np.array([[c_psi,  s_psi, 0],
                         [-s_psi, c_psi, 0],
                         [0,      0,     1]])

    for _ in range(15):
        f_wind = get_wind_force_at_attitude(p, v_wind_world, (phi, theta, base_yaw))
        F_tot  = np.array([0.0, 0.0, -m_tot * p.g]) + f_wind
        F_req  = -F_tot

        # Decompose F_req in the yaw-aligned frame to obtain pure roll/pitch
        F_req_body = R_z_inv @ F_req
        theta = np.arctan2(F_req_body[0], abs(F_req_body[2]))
        phi   = np.arctan2(-F_req_body[1],
                           np.sqrt(F_req_body[0] ** 2 + F_req_body[2] ** 2))

    return phi, theta, f_wind


def compute_power_index(ff_forces: np.ndarray, p, v_wind: np.ndarray) -> float:
    """
    Compute a proxy for the total electrical power consumed by all UAVs,
    based on Actuator-Disk Theory:  P_i ∝ ||F_thrust_i||^{3/2}.

    The thrust vector for UAV i is:
        F_thrust_i = ff_forces[:,i]          (cable tension reaction)
                   + m_drone * g * ẑ        (UAV weight compensation)
                   - F_aero_drone            (aerodynamic drag on the UAV body)

    Parameters
    ----------
    ff_forces : (3, N) array – feedforward force vectors from the optimizer
                (i.e. T_i * û_{L,i}, directed upward toward each UAV)
    p         : system parameters
    v_wind    : wind velocity vector [m/s, world frame]

    Returns
    -------
    Scalar power index  Σ_i ||F_thrust_i||^{3/2}
    """
    v_mag = np.linalg.norm(v_wind)
    if v_mag > 1e-3:
        f_aero_drone = (0.5 * p.rho * p.Cd_uav * p.A_uav
                        * v_mag ** 2 * (v_wind / v_mag))
    else:
        f_aero_drone = np.zeros(3)

    g_vec     = np.array([0.0, 0.0, p.m_drone * p.g])   # weight compensation
    p_total   = 0.0

    for i in range(p.N):
        # Net thrust vector: tension contribution + UAV weight - aerodynamic drag
        thrust_vec = ff_forces[:, i] + g_vec - f_aero_drone
        thrust_mag = np.linalg.norm(thrust_vec)
        p_total   += thrust_mag ** 1.5

    return p_total


# ==============================================================================
# CONTEXT MOCK
# ==============================================================================
class MockCtx:
    """Minimal context providing optimizer state continuity across sweep steps."""
    def __init__(self, p):
        self.last_theta_opt       = np.radians(getattr(p, 'theta_ref', 30.0))
        self.last_stable_yaw      = 0.0
        self.last_damp_correction = np.zeros(p.N)
        self.last_T_final         = np.zeros(p.N)   # must be array, not None
        self.dot_T_filt           = np.zeros(p.N)
        self.F_xy_filt            = np.zeros(2)
        self.e_filt_despun        = np.zeros(3)


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
def run_wind_direction_sweep():
    # --------------------------------------------------------------------------
    # Parameter setup
    # --------------------------------------------------------------------------
    p = SysParams()

    # Payload geometry
    p.payload_shape = 'box'
    p.pay_l   = 1.0
    p.pay_w   = 0.4
    p.pay_h   = 0.1
    p.R_disk  = 0.5
    p.m_payload = 1.5


    p.enable_sloshing = False

    # Wind speed (constant magnitude, direction sweeps)
    wind_speed = 10.0   # [m/s]

    # Optimizer weights
    p.w_T        = 1.0
    p.w_ref      = 50.0
    p.w_barrier  = 0
    p.w_cond     = 0
    p.w_resid_f  = 300.0
    p.w_resid_m  = 100.0

    # Static sweep: disable smoothness penalty so each angle is solved independently
    p.w_smooth            = 0.0
    p.max_angle_variation = 20.0   # degrees – relaxed for static sweep
    p.k_limit = 0.9

    # Adaptation (pure geometric, no aerodynamic tilt)
    p.lambda_shape = 0.0
    p.lambda_aero  = 0.0
    p.lambda_static = 0.0
    p.lambda_twist  = 0.0

    # Rebuild geometry with updated payload dimensions
    p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)

    # Sweep angles
    angles_deg = np.linspace(-180, 180, 73)   # 5° step

    # Result arrays
    res_power       = []
    res_imbalance   = []   # std dev of tensions
    res_max_tension = []
    res_min_tension = []

    # Fixed wind vector in world frame; we rotate the formation (yaw) instead
    wind_vec_world = np.array([wind_speed, 0.0, 0.0])
    p.wind_vel     = wind_vec_world

    ctx = MockCtx(p)

    print("Running wind-direction sweep …")
    print(f"  Payload : {p.payload_shape.upper()}  "
          f"({p.pay_l} m × {p.pay_w} m × {p.pay_h} m)")
    print(f"  Wind    : {wind_speed} m/s")
    print(f"  Steps   : {len(angles_deg)}")

    for deg in angles_deg:
        yaw_rad = np.radians(deg)
        ctx.last_stable_yaw = yaw_rad

        # Aerodynamic equilibrium at this yaw (lambda_aero=0 → phi=theta=0)
        phi_eq, theta_eq, _ = compute_aerodynamic_equilibrium(
            p, wind_vec_world, base_yaw=yaw_rad)
        phi_target   = phi_eq   * p.lambda_aero
        theta_target = theta_eq * p.lambda_aero

        # Wind force at the resulting attitude
        f_wind = get_wind_force_at_attitude(
            p, wind_vec_world, (phi_target, theta_target, yaw_rad))

        state = {
            'uav_pos'  : np.zeros((3, p.N)),
            'uav_vel'  : np.zeros((3, p.N)),
            'pay_pos'  : np.zeros(3),
            'pay_vel'  : np.zeros(3),
            'pay_att'  : np.array([phi_target, theta_target, yaw_rad]),
            'pay_omega': np.zeros(3),
            'int_uav'  : np.zeros((3, p.N)),
        }

        _, _, _, ff_forces = formation.compute_optimal_formation(
            p, state,
            acc_cmd_pay     = np.zeros(3),
            acc_ang_cmd_pay = np.zeros(3),
            ref_yaw         = yaw_rad,
            force_attitude  = (phi_target, theta_target),
            F_ext_total     = f_wind,
            F_aero_for_moment = f_wind,
            ctx             = ctx,
            com_offset_body = p.CoM_offset,
        )

        tensions = np.linalg.norm(ff_forces, axis=0)

        res_power.append(compute_power_index(ff_forces, p, wind_vec_world))
        res_imbalance.append(float(np.std(tensions)))
        res_max_tension.append(float(np.max(tensions)))
        res_min_tension.append(float(np.min(tensions)))

    print("Sweep complete.")

    # --------------------------------------------------------------------------
    # Safety limits (for reference lines)
    # --------------------------------------------------------------------------
    m_tot        = p.m_payload + p.m_liquid
    T_safe_base  = p.k_safe * (m_tot * p.g / p.N)
    T_max_limit  = (p.F_max_thrust - p.m_drone * p.g) * p.k_limit

    best_idx = int(np.argmin(res_power))
    best_yaw = angles_deg[best_idx]

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    #fig.suptitle(
    #    f"Formation Response to Wind Direction — {p.payload_shape.upper()} Payload\n"
    #    f"Wind speed: {wind_speed} m/s  |  "
    #    r"$\lambda_{\mathrm{shape}}=0$, $\lambda_{\mathrm{aero}}=0$",
    #    fontsize=12, fontweight='bold'
    #)

    # --- 1. Power index ---
    axs[0].plot(angles_deg, res_power, 'royalblue', linewidth=2)
    axs[0].axvline(best_yaw, color='g', linestyle='--', linewidth=1.5,
                   label=f'Min-power yaw: {best_yaw:.0f}°')
    axs[0].set_title('Energetic Efficiency vs. Wind Direction',
                     fontweight='bold', fontsize=11)
    axs[0].set_ylabel(r'Power Index  $\sum_i \|\mathbf{F}_{\mathrm{thrust},i}\|^{1.5}$')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(fontsize=9)

    # --- 2. Load imbalance ---
    axs[1].plot(angles_deg, res_imbalance, color='mediumpurple', linewidth=2)
    axs[1].set_title('Formation Load Imbalance  (lower → more symmetric)',
                     fontweight='bold', fontsize=11)
    axs[1].set_ylabel('Tension Std. Dev. [N]')
    axs[1].grid(True, linestyle=':', alpha=0.6)

    # --- 3. Safety margins ---
    axs[2].plot(angles_deg, res_max_tension, 'tomato',     linewidth=2,
                label='Max tension  (saturation risk)')
    axs[2].plot(angles_deg, res_min_tension, 'darkorange', linewidth=2,
                label='Min tension  (slack risk)')
    axs[2].axhline(T_max_limit, color='red',    linestyle=':', linewidth=1.5,
                   label=f'Saturation limit  ({T_max_limit:.1f} N)')
    axs[2].axhline(T_safe_base, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Min safe tension  ({T_safe_base:.2f} N)')
    axs[2].axhline(0.0, color='black', linewidth=0.8)

    axs[2].fill_between(angles_deg, T_max_limit, T_max_limit + 20,
                        color='red', alpha=0.08)
    axs[2].fill_between(angles_deg, 0, T_safe_base,
                        color='orange', alpha=0.12)

    axs[2].set_title('Operational Safety Margins',
                     fontweight='bold', fontsize=11)
    axs[2].set_ylabel('Cable Tension [N]')
    axs[2].set_xlabel('Wind Direction relative to Payload Yaw [deg]')
    axs[2].set_ylim(-1.0, T_max_limit * 1.25)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper right', ncol=2, fontsize=9)

    mean_power = np.mean(res_power)
    axs[0].set_ylim(mean_power - 1.0, mean_power + 1.0)
    
    # E per il secondo pannello:
    mean_imb = np.mean(res_imbalance)
    axs[1].set_ylim(mean_imb - 0.5, mean_imb + 0.5)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.35)
    plt.show()


if __name__ == "__main__":
    run_wind_direction_sweep()