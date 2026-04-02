"""
graph_wrench_residual.py
========================
UN UNICO grafico con 3 subplot:
  [0] WFW Feasibility Map   (F_x vs M_z)
  [1] Force Residual        (3 strategie vs vento)
  [2] Moment Residual       (3 strategie vs vento)

Nessun salvataggio. Nessun grafico aggiuntivo.
"""

import sys, os, copy, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _p in [ROOT, os.path.dirname(__file__)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter

import formation as formation
from parameters import SysParams

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'figure.dpi': 120,
    'axes.grid': True, 'grid.alpha': 0.35, 'grid.linestyle': ':',
})


# ==============================================================================
class MockCtx:
    def __init__(self, theta_ref_deg, N):
        self.last_theta_opt        = np.radians(theta_ref_deg)
        self.last_stable_yaw       = 0.0
        self.last_damp_correction  = np.zeros(N)
        self.last_T_final          = np.zeros(N)
        self.dot_T_filt            = np.zeros(N)
        self.F_xy_filt             = np.zeros(2)
        self.e_filt_despun         = np.zeros(3)
        self.last_delta_moment     = np.zeros(3)
        self.lambda_traj_effective = 0.0


def make_params(lambda_shape=1.0, lambda_aero=0.0):
    p = SysParams()
    p.payload_shape       = 'sphere'
    p.m_liquid            = 1e-9
    p.enable_sloshing     = False
    p.lambda_shape        = lambda_shape
    p.lambda_aero         = lambda_aero
    p.lambda_CoM          = 0.0
    p.lambda_twist        = 1.0
    p.w_T                 = 1.0
    p.w_ref               = 50.0
    p.w_smooth            = 50.0
    p.w_barrier           = 100.0
    p.w_cond              = 10.0
    p.w_resid_f           = 3000.0
    p.w_resid_m           = 1000.0
    p.max_angle_variation = 20.0
    p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
    return p


def base_state(p):
    return {
        'pay_pos': np.zeros(3), 'pay_att': np.zeros(3),
        'pay_vel': np.zeros(3), 'pay_omega': np.zeros(3),
        'uav_vel': np.zeros((3, p.N)), 'uav_pos': p.uav_offsets.copy(),
        'int_uav': np.zeros((3, p.N)),
    }


def tension_limits(p, F_ext_mag=0.0):
    m_tot  = p.m_payload + p.m_liquid
    T_safe = p.k_safe * (m_tot * p.g / p.N) + p.k_safe2 * (F_ext_mag / p.N)
    T_lim  = (p.F_max_thrust - p.m_drone * p.g) * p.k_limit
    return T_safe, T_lim


def reconstruct_wrench(ff, p, R_pay=None):
    if R_pay is None:
        R_pay = np.eye(3)
    F_ach = np.sum(ff, axis=1)
    M_ach = np.zeros(3)
    for i in range(p.N):
        M_ach += np.cross(R_pay @ p.attach_vecs[:, i], ff[:, i])
    return F_ach, M_ach


def R_from_net_force(ff):
    net  = np.sum(ff, axis=1)
    nmag = np.linalg.norm(net)
    if nmag < 1e-3:
        return np.eye(3)
    z_pay  = net / nmag
    z_w    = np.array([0.0, 0.0, 1.0])
    axis   = np.cross(z_w, z_pay)
    an     = np.linalg.norm(axis)
    if an < 1e-6:
        return np.eye(3)
    axis  /= an
    angle  = np.arccos(np.clip(np.dot(z_w, z_pay), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)


def wind_force_sphere(p, F_mag):
    if F_mag < 1e-3:
        return np.zeros(3)
    A = np.pi * p.R_disk**2
    v = np.sqrt(2.0 * F_mag / (p.rho * p.Cd_pay * A))
    return np.array([0.5 * p.rho * p.Cd_pay * A * v**2, 0.0, 0.0])


# ==============================================================================
def plot_wfw_map(ax):
    print("  [1/3] WFW map …")
    p     = make_params(lambda_shape=1.0, lambda_aero=0.0)
    m_tot = p.m_payload + p.m_liquid
    F_nom = max(m_tot * p.g, 1.0)
    M_nom = max(F_nom * getattr(p, 'pay_l', 1.0) * 0.5, 0.1)

    Fx_vals = np.linspace(0, 35, 28)
    Mz_vals = np.linspace(-15, 15, 28)
    feasible  = np.zeros((len(Mz_vals), len(Fx_vals)), dtype=bool)
    residuals = np.full_like(feasible, np.nan, dtype=float)

    for j, Fx in enumerate(Fx_vals):
        for i, Mz in enumerate(Mz_vals):
            ctx   = MockCtx(p.theta_ref, p.N)
            F_ext = np.array([Fx, 0.0, 0.0])
            T_safe, T_lim = tension_limits(p, F_ext_mag=Fx)
            try:
                tgt, _, _, ff = formation.compute_optimal_formation(
                    p, base_state(p),
                    acc_cmd_pay=np.zeros(3),
                    acc_ang_cmd_pay=np.array([0.0, 0.0, Mz / max(p.J[2,2], 0.01)]),
                    ref_yaw=0.0, force_attitude=(0.0, 0.0),
                    F_ext_total=F_ext, ctx=ctx, com_offset_body=p.CoM_offset,
                )
            except Exception:
                continue
            T  = np.linalg.norm(ff, axis=0)
            ok = np.all(T >= T_safe * 0.90) and np.all(T <= T_lim * 1.05)
            feasible[i, j] = ok
            if ok:
                F_ach, M_ach = reconstruct_wrench(ff, p)
                F_req = np.array([-Fx, 0.0, m_tot * p.g])
                M_req = np.array([0.0, 0.0, Mz])
                residuals[i, j] = (np.linalg.norm(F_ach - F_req) / F_nom +
                                   np.linalg.norm(M_ach - M_req) / M_nom)
        if (j + 1) % 7 == 0:
            print(f"\r    col {j+1}/{len(Fx_vals)}", end='', flush=True)
    print()

    feas_sm  = uniform_filter(feasible.astype(float), size=2) > 0.5
    res_plot = np.where(feas_sm,  residuals, np.nan)
    inf_plot = np.where(~feas_sm, 1.0,       np.nan)

    cmap_g = mcolors.LinearSegmentedColormap.from_list('g', ['#c8f5c8', '#005000'])
    ax.contourf(Fx_vals, Mz_vals, inf_plot, levels=[0.5, 1.5],
                colors=['#ffb3b3'], alpha=0.80)
    cf = ax.contourf(Fx_vals, Mz_vals, res_plot, levels=20, cmap=cmap_g)
    plt.colorbar(cf, ax=ax,
                 label=r'$\|\Delta F\|/F_{nom}+\|\Delta M\|/M_{nom}$')
    ax.set_xlabel(r'$F_x$ [N]')
    ax.set_ylabel(r'$M_z$ [N·m]')
    ax.set_title('Wrench-Feasible Workspace\n'
                 r'$\lambda_s{=}1,\ \lambda_a{=}0$  (hover)',
                 fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(color='#80c980', label='Feasible'),
        mpatches.Patch(color='#ffb3b3', label='Infeasible'),
    ], loc='upper right', fontsize=8)


# ==============================================================================
def plot_residuals(ax_f, ax_m):
    print("  [2-3/3] Residual sweep …")
    p_base  = make_params()
    m_tot   = p_base.m_payload + p_base.m_liquid
    F_nom   = max(m_tot * p_base.g, 1.0)
    M_nom   = max(F_nom * getattr(p_base, 'pay_l', 1.0) * 0.5, 0.1)
    wind_forces = np.linspace(0, 35, 36)

    strategies = [
        ('Rigid',          0.0, 0.0, 'dimgray',       '--'),
        ('Shape-Adaptive', 1.0, 0.0, 'mediumseagreen', '-'),
        ('Aero-Tilt',      0.0, 1.0, 'royalblue',      '-'),
    ]
    labels = {
        'Rigid':          r'Rigid  ($\lambda_s{=}0,\,\lambda_a{=}0$)',
        'Shape-Adaptive': r'Shape-Adaptive  ($\lambda_s{=}1$)',
        'Aero-Tilt':      r'Aero-Tilt  ($\lambda_a{=}1$)',
    }

    for name, lam_s, lam_a, color, ls in strategies:
        p   = copy.deepcopy(p_base)
        p.lambda_shape = lam_s
        p.lambda_aero  = lam_a
        ctx = MockCtx(p.theta_ref, p.N)
        rf_list, rm_list = [], []

        for F_mag in wind_forces:
            F_ext = wind_force_sphere(p, F_mag)
            try:
                tgt, _, _, ff = formation.compute_optimal_formation(
                    p, base_state(p),
                    acc_cmd_pay=np.zeros(3), acc_ang_cmd_pay=np.zeros(3),
                    ref_yaw=0.0, force_attitude=(0.0, 0.0),
                    F_ext_total=F_ext, ctx=ctx, com_offset_body=p.CoM_offset,
                )
            except Exception:
                rf_list.append(np.nan); rm_list.append(np.nan)
                continue
            F_req = np.array([-F_ext[0], -F_ext[1], m_tot * p.g])
            F_ach, M_ach = reconstruct_wrench(ff, p, R_from_net_force(ff))
            rf_list.append(np.linalg.norm(F_ach - F_req) / F_nom)
            rm_list.append(np.linalg.norm(M_ach)          / M_nom)

        ax_f.plot(wind_forces, rf_list, color=color, ls=ls, lw=2, label=labels[name])
        ax_m.plot(wind_forces, rm_list, color=color, ls=ls, lw=2, label=labels[name])
        print(f"    {name:16s}  F-res={np.nanmax(rf_list):.3f}  M-res={np.nanmax(rm_list):.3f}")

    for ax, title, ylabel in [
        (ax_f, 'Force Residual vs. Wind Force',
         r'$\|\mathbf{F}_{ach}-\mathbf{F}_{req}\|\;/\;F_{nom}$'),
        (ax_m, 'Moment Residual vs. Wind Force',
         r'$\|\mathbf{M}_{ach}\|\;/\;M_{nom}$'),
    ]:
        ax.set_xlabel('Nominal Wind Force [N]')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 35)
        ax.set_ylim(bottom=0)
        ax.axvspan(28, 35, color='salmon', alpha=0.15)
        ax.text(28.3, ax.get_ylim()[1] * 0.88,
                'Saturation', fontsize=8, color='firebrick', va='top')


# ==============================================================================
if __name__ == '__main__':
    print("\nComputing — please wait …")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        'Static Analysis — Wrench-Feasible Workspace and Tracking Residuals',
        fontsize=13, fontweight='bold'
    )
    plot_wfw_map(axes[0])
    plot_residuals(axes[1], axes[2])
    fig.tight_layout()
    plt.show()