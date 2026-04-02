"""
graphs_hovering_wind.py
-----------------------
Static analysis: comparison of three formation control strategies under
increasing wind load during hovering.

Strategies compared
-------------------
  • Rigid          (λ_shape=0, λ_aero=0) – fixed circular formation, no tilt
  • Shape-Adaptive (λ_shape=1, λ_aero=0) – elliptical deformation enabled
  • Aero-Tilt      (λ_shape=0, λ_aero=1) – payload tilts into the wind

For each strategy and each wind force level the following metrics are
recorded from the optimizer output:

  1. Power index  Σ_i ||F_thrust_i||^{3/2}  (Actuator-Disk Theory)
  2. Minimum cable tension  (slack / stability margin)
  3. Maximum cable tension  (saturation margin)

Physical note on the power index
---------------------------------
The thrust vector for UAV i is:

    F_thrust_i = ff_forces[:,i]          (cable tension, directed upward)
               + m_drone * g * ẑ        (UAV self-weight compensation)
               - F_aero_drone            (aerodynamic drag on the UAV body)

where ff_forces[:,i] = T_i * û_{L,i} is directly returned by the optimizer.

Force–speed conversion
-----------------------
To give the x-axis a meaningful physical interpretation the nominal wind
force (computed at zero tilt, using the payload frontal area) is converted
to an equivalent wind speed, which is then used to evaluate the actual
aerodynamic force on the tilted payload for the Aero-Tilt strategy.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import numpy as np
import matplotlib.pyplot as plt

import formation as formation
from parameters import SysParams
import physics


# ==============================================================================
# PHYSICAL HELPERS
# ==============================================================================

def get_wind_force_at_attitude(p, v_wind_world: np.ndarray,
                               attitude: tuple) -> np.ndarray:
    """
    Aerodynamic drag force on the payload for a given wind vector and attitude.

    Parameters
    ----------
    p            : system parameters
    v_wind_world : wind velocity [m/s, world frame]
    attitude     : (phi, theta, psi) Euler angles [rad]

    Returns
    -------
    f_wind : drag force vector [N, world frame]
    """
    phi, theta, psi = attitude
    wind_mag = np.linalg.norm(v_wind_world)
    if wind_mag < 1e-3:
        return np.zeros(3)

    wind_dir     = v_wind_world / wind_mag
    R_pay        = physics.get_rotation_matrix(phi, theta, psi)
    wind_in_body = R_pay.T @ wind_dir

    shape = getattr(p, 'payload_shape', 'cylinder')

    if shape in ['box', 'rect', 'square']:
        A_x = p.pay_w * p.pay_h
        A_y = p.pay_l * p.pay_h
        A_z = p.pay_l * p.pay_w
        proj_area = (A_x * abs(wind_in_body[0])
                   + A_y * abs(wind_in_body[1])
                   + A_z * abs(wind_in_body[2]))
    elif shape == 'sphere':
        proj_area = np.pi * p.R_disk ** 2          # orientation-independent
    else:                                           # cylinder
        A_side    = 2.0 * p.R_disk * p.pay_h
        A_top     = np.pi * p.R_disk ** 2
        sin_tilt  = np.sqrt(wind_in_body[0] ** 2 + wind_in_body[1] ** 2)
        proj_area = A_side * sin_tilt + A_top * abs(wind_in_body[2])

    f_wind = (0.5 * p.rho * p.Cd_pay * proj_area * wind_mag ** 2) * wind_dir
    return f_wind


def get_nominal_area(p) -> float:
    """
    Frontal area of the payload at zero tilt, used for the
    force–speed conversion on the sweep x-axis.
    """
    shape = getattr(p, 'payload_shape', 'cylinder')
    if shape in ['box', 'rect', 'square']:
        return p.pay_w * p.pay_h
    elif shape == 'sphere':
        return np.pi * p.R_disk ** 2
    else:
        return 2.0 * p.R_disk * p.pay_h


def compute_aerodynamic_equilibrium(p, v_wind_world: np.ndarray):
    """
    Iterative solver for the hovering aerodynamic equilibrium attitude.

    Finds (phi, theta) such that the combined gravity + drag force is
    balanced by a purely vertical cable resultant (yaw = 0).

    Returns
    -------
    phi   : equilibrium roll  [rad]
    theta : equilibrium pitch [rad]
    f_wind: wind force at equilibrium [N, world frame]
    """
    phi, theta = 0.0, 0.0
    m_tot = p.m_payload + getattr(p, 'm_liquid', 0.0)

    for _ in range(15):
        f_wind = get_wind_force_at_attitude(p, v_wind_world, (phi, theta, 0.0))
        F_tot  = np.array([0.0, 0.0, -m_tot * p.g]) + f_wind
        F_req  = -F_tot
        theta  = np.arctan2(F_req[0], abs(F_req[2]))
        phi    = np.arctan2(-F_req[1],
                            np.sqrt(F_req[0] ** 2 + F_req[2] ** 2))

    return phi, theta, f_wind


def compute_power_index(ff_forces: np.ndarray, p, v_wind: np.ndarray) -> float:
    """
    Total power proxy using Actuator-Disk Theory: P ∝ ||F_thrust||^{3/2}.

    The thrust vector required by UAV i is:

        F_thrust_i = ff_forces[:,i]          (cable tension reaction)
                   + m_drone * g * ẑ        (self-weight compensation)
                   - F_aero_drone            (aerodynamic drag, same for all UAVs)

    Parameters
    ----------
    ff_forces : (3, N) feedforward forces from the optimizer
    p         : system parameters
    v_wind    : wind velocity [m/s, world frame]

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

    g_vec   = np.array([0.0, 0.0, p.m_drone * p.g])
    p_total = 0.0

    for i in range(p.N):
        thrust_vec = ff_forces[:, i] + g_vec - f_aero_drone
        thrust_mag = np.linalg.norm(thrust_vec)
        p_total   += thrust_mag ** 1.5

    return p_total


# ==============================================================================
# CONTEXT MOCK
# ==============================================================================
class MockCtx:
    """Minimal context providing optimizer state continuity across sweep steps."""
    def __init__(self, theta_ref_deg: float, N: int):
        self.last_theta_opt       = np.radians(theta_ref_deg)
        self.last_stable_yaw      = 0.0
        self.last_damp_correction = np.zeros(N)
        self.last_T_final         = np.zeros(N)   # must be array, not None
        self.dot_T_filt           = np.zeros(N)
        self.F_xy_filt            = np.zeros(2)
        self.e_filt_despun        = np.zeros(3)


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
def run_hovering_wind_sweep():
    # --------------------------------------------------------------------------
    # Base parameter setup (shared across all three strategies)
    # --------------------------------------------------------------------------
    p_base = SysParams()

    # Nominal cone angle and optimizer weights
    p_base.theta_ref  = 30.0
    p_base.w_ref      = 150.0
    p_base.w_smooth   = 100.0
    p_base.w_barrier  = 10.0
    p_base.w_T        = 1.0
    p_base.w_cond     = 10.0
    p_base.w_resid_f  = 300.0
    p_base.w_resid_m  = 100.0

    # Payload
    p_base.payload_shape = 'sphere'
    p_base.enable_sloshing = False

    # Safety margins
    p_base.F_ref   = 30.0
    p_base.k_limit = 0.95
    p_base.k_safe  = 0.1
    p_base.k_safe2 = 0.5
    p_base.min_drone_z_rel = -10

    # Deformation parameters
    p_base.lambda_perp = 1.3
    p_base.lambda_par  = 0.9

    # Allow up to 10° change per sweep step for smooth progression
    p_base.max_angle_variation = 10.0

    p_base.uav_offsets, p_base.attach_vecs, p_base.geo_radius = (
        formation.compute_geometry(p_base))

    # --------------------------------------------------------------------------
    # Nominal frontal area (used for force → wind-speed conversion)
    # --------------------------------------------------------------------------
    A_nom = get_nominal_area(p_base)

    # Force sweep: 0 – 40 N nominal wind force
    wind_forces = np.linspace(0.0, 40.0, 30)

    # Result arrays
    pow_rigid,  min_T_rigid,  max_T_rigid  = [], [], []
    pow_adapt,  min_T_adapt,  max_T_adapt  = [], [], []
    pow_aero,   min_T_aero,   max_T_aero   = [], [], []

    # Independent context objects so each strategy has its own alpha history
    ctx_rigid = MockCtx(p_base.theta_ref, p_base.N)
    ctx_adapt = MockCtx(p_base.theta_ref, p_base.N)
    ctx_aero  = MockCtx(p_base.theta_ref, p_base.N)

    # Base hover state (static, zero velocity)
    state_base = {
        'pay_pos'  : np.zeros(3),
        'pay_att'  : np.zeros(3),
        'pay_vel'  : np.zeros(3),
        'pay_omega': np.zeros(3),
        'uav_vel'  : np.zeros((3, p_base.N)),
        'uav_pos'  : p_base.uav_offsets.copy(),
        'int_uav'  : np.zeros((3, p_base.N)),
    }

    print("Running hovering wind sweep …")
    print(f"  Payload : {p_base.payload_shape.upper()}")
    print(f"  N UAVs  : {p_base.N}")
    print(f"  Steps   : {len(wind_forces)}")

    for f_nom in wind_forces:
        # ----------------------------------------------------------------
        # Convert nominal force to equivalent wind speed, then obtain the
        # actual drag force on the upright payload (attitude = 0).
        # ----------------------------------------------------------------
        if f_nom > 1e-3:
            v_sq   = f_nom / (0.5 * p_base.rho * p_base.Cd_pay * A_nom)
            v_wind_mag = np.sqrt(v_sq)
        else:
            v_wind_mag = 0.0

        v_wind         = np.array([v_wind_mag, 0.0, 0.0])
        F_wind_upright = get_wind_force_at_attitude(p_base, v_wind, (0, 0, 0))

        state = copy.deepcopy(state_base)

        # ================================================================
        # STRATEGY 1 — RIGID (λ_shape=0, λ_aero=0)
        # ================================================================
        p_rigid = copy.deepcopy(p_base)
        p_rigid.lambda_shape = 0.0
        p_rigid.lambda_aero  = 0.0

        _, _, _, ff_rigid = formation.compute_optimal_formation(
            p_rigid, copy.deepcopy(state),
            acc_cmd_pay     = np.zeros(3),
            acc_ang_cmd_pay = np.zeros(3),
            ref_yaw         = 0.0,
            force_attitude  = (0.0, 0.0),
            F_ext_total     = F_wind_upright,
            com_offset_body = p_rigid.CoM_offset,
            ctx             = ctx_rigid,
        )
        T_rigid = np.linalg.norm(ff_rigid, axis=0)
        pow_rigid.append(compute_power_index(ff_rigid, p_rigid, v_wind))
        min_T_rigid.append(float(np.min(T_rigid)))
        max_T_rigid.append(float(np.max(T_rigid)))

        # ================================================================
        # STRATEGY 2 — SHAPE-ADAPTIVE (λ_shape=1, λ_aero=0)
        # ================================================================
        p_adapt = copy.deepcopy(p_base)
        p_adapt.lambda_shape = 1.0
        p_adapt.lambda_aero  = 0.0

        _, _, _, ff_adapt = formation.compute_optimal_formation(
            p_adapt, copy.deepcopy(state),
            acc_cmd_pay     = np.zeros(3),
            acc_ang_cmd_pay = np.zeros(3),
            ref_yaw         = 0.0,
            force_attitude  = (0.0, 0.0),
            F_ext_total     = F_wind_upright,
            com_offset_body = p_adapt.CoM_offset,
            ctx             = ctx_adapt,
        )
        T_adapt = np.linalg.norm(ff_adapt, axis=0)
        pow_adapt.append(compute_power_index(ff_adapt, p_adapt, v_wind))
        min_T_adapt.append(float(np.min(T_adapt)))
        max_T_adapt.append(float(np.max(T_adapt)))

        # ================================================================
        # STRATEGY 3 — AERO-TILT (λ_shape=0, λ_aero=1)
        #
        # The payload tilts to aerodynamic equilibrium; wind force is
        # recomputed at the tilted attitude.  For a sphere the projected
        # area is constant, so the drag force equals F_wind_upright; the
        # meaningful difference lies in the moment balance and formation tilt.
        # ================================================================
        p_aero = copy.deepcopy(p_base)
        p_aero.lambda_shape = 0.0
        p_aero.lambda_aero  = 1.0

        phi_eq, theta_eq, _ = compute_aerodynamic_equilibrium(p_aero, v_wind)
        F_wind_tilted = get_wind_force_at_attitude(
            p_aero, v_wind, (phi_eq, theta_eq, 0.0))

        state_aero = copy.deepcopy(state)
        state_aero['pay_att'] = np.array([phi_eq, theta_eq, 0.0])

        _, _, _, ff_aero = formation.compute_optimal_formation(
            p_aero, state_aero,
            acc_cmd_pay     = np.zeros(3),
            acc_ang_cmd_pay = np.zeros(3),
            ref_yaw         = 0.0,
            force_attitude  = (phi_eq, theta_eq),
            F_ext_total     = F_wind_tilted,
            com_offset_body = p_aero.CoM_offset,
            ctx             = ctx_aero,
        )
        T_aero = np.linalg.norm(ff_aero, axis=0)
        pow_aero.append(compute_power_index(ff_aero, p_aero, v_wind))
        min_T_aero.append(float(np.min(T_aero)))
        max_T_aero.append(float(np.max(T_aero)))

    print("Sweep complete.")

    # --------------------------------------------------------------------------
    # Safety limits
    # --------------------------------------------------------------------------
    m_tot       = p_base.m_payload + p_base.m_liquid
    T_safe_base = p_base.k_safe * (m_tot * p_base.g / p_base.N)
    T_max_limit = (p_base.F_max_thrust - p_base.m_drone * p_base.g) * p_base.k_limit

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
    fig.suptitle(
        "Hovering Performance vs. Nominal Wind Force — Strategy Comparison\n"
        f"Payload: {p_base.payload_shape.upper()}  |  N = {p_base.N} UAVs",
        fontsize=12, fontweight='bold'
    )

    LABEL_RIGID  = r'Rigid  ($\lambda_s=0,\ \lambda_a=0$)'
    LABEL_ADAPT  = r'Shape-Adaptive  ($\lambda_s=1,\ \lambda_a=0$)'
    LABEL_AERO   = r'Aero-Tilt  ($\lambda_s=0,\ \lambda_a=1$)'

    COLOR_RIGID  = 'dimgray'
    COLOR_ADAPT  = 'mediumseagreen'
    COLOR_AERO   = 'royalblue'

    # --- 1. Power index ---
    ax1 = axes[0]
    ax1.plot(wind_forces, pow_rigid, '--', color=COLOR_RIGID,  lw=2, label=LABEL_RIGID)
    ax1.plot(wind_forces, pow_adapt, '-',  color=COLOR_ADAPT,  lw=2, label=LABEL_ADAPT)
    ax1.plot(wind_forces, pow_aero,  '-',  color=COLOR_AERO,   lw=2, label=LABEL_AERO)
    ax1.set_ylabel(
        r'Power Index  $\sum_i \|\mathbf{F}_{\mathrm{thrust},i}\|^{1.5}$',
        fontsize=10)
    ax1.set_title('Energetic Efficiency vs. Nominal Wind Force',
                  fontweight='bold', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9)

    # --- 2. Minimum cable tension ---
    ax2 = axes[1]
    ax2.plot(wind_forces, min_T_rigid, '--', color=COLOR_RIGID,  lw=2, label=LABEL_RIGID)
    ax2.plot(wind_forces, min_T_adapt, '-',  color=COLOR_ADAPT,  lw=2, label=LABEL_ADAPT)
    ax2.plot(wind_forces, min_T_aero,  '-',  color=COLOR_AERO,   lw=2, label=LABEL_AERO)
    ax2.axhline(T_safe_base, color='darkorange', linestyle=':',
                label=rf'$T_{{\mathrm{{safe}}}}$ = {T_safe_base:.2f} N')
    ax2.axhline(0.0, color='crimson', linestyle='-', alpha=0.5,
                label='Cable slack  (T = 0)')
    ax2.fill_between(wind_forces, 0.0, T_safe_base,
                     color='red', alpha=0.07)
    ax2.set_ylabel('Minimum Cable Tension [N]', fontsize=10)
    ax2.set_title('Slack Margin  (higher → safer against cable slack)',
                  fontweight='bold', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=9)

    # --- 3. Maximum cable tension ---
    ax3 = axes[2]
    ax3.plot(wind_forces, max_T_rigid, '--', color=COLOR_RIGID,  lw=2, label=LABEL_RIGID)
    ax3.plot(wind_forces, max_T_adapt, '-',  color=COLOR_ADAPT,  lw=2, label=LABEL_ADAPT)
    ax3.plot(wind_forces, max_T_aero,  '-',  color=COLOR_AERO,   lw=2, label=LABEL_AERO)
    ax3.axhline(T_max_limit, color='crimson', linestyle=':',
                label=rf'Saturation limit = {T_max_limit:.1f} N')
    ax3.fill_between(wind_forces, T_max_limit, T_max_limit * 1.2,
                     color='red', alpha=0.08)
    ax3.set_xlabel('Nominal Wind Force [N]', fontsize=10)
    ax3.set_ylabel('Maximum Cable Tension [N]', fontsize=10)
    ax3.set_title('Saturation Margin  (lower → safer against motor saturation)',
                  fontweight='bold', fontsize=11)
    ax3.set_ylim(bottom=0.0, top=T_max_limit * 1.25)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_hovering_wind_sweep()