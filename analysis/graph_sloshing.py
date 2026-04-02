"""
graph_sloshing.py
-----------------
Dynamic analysis: effectiveness of sloshing feedforward compensation.

Two parallel simulations are run with identical parameters except for
whether the sloshing feedforward is active:

  • Baseline    — sloshing ON, NO feedforward compensation (lambda_CoM = 0)
  • Compensated — sloshing ON, WITH feedforward compensation (lambda_CoM = 1)

Key design choices for a meaningful comparison
-----------------------------------------------
  1. slosh_freq is set near the natural pendulum frequency f_n = √(g/L) / (2π)
     so the liquid excitation actually drives the payload attitude.
  2. The simulation is run in pure hovering mode (no navigation trajectory),
     so there is no large programmed attitude bias that would bury the sloshing
     signal.  Phases 0-2 complete normally; from Phase 3 onward the reference
     holds position and attitude = (0, 0, 0).
  3. The angular error is measured as the deviation from zero attitude target,
     i.e. error_i = att_i - 0, isolating purely the sloshing disturbance.

Metrics plotted
---------------
  Panel A — Payload roll φ and pitch θ (absolute, target = 0 in hold)
  Panel B — |Δ(φ,θ)| oscillation amplitude (high-pass filtered, 5-s window)
  Panel C — Instantaneous liquid CoM eccentricity |e_CoM(t)| [cm]
  Panel D — Cable tension spread T_max − T_min [N]
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy, time
import numpy as np
import matplotlib.pyplot as plt

import formation as formation
from mission import MissionContext, equations_of_motion, update_guidance
from parameters import SysParams


# ==============================================================================
# PHASE SHADING
# ==============================================================================
PHASE_COLORS = {0:'#AAAAAA', 1:'#FFD700', 2:'#90EE90', 3:'#87CEEB',
                4:'#FFA07A', 5:'#DDA0DD', 6:'#98FB98'}

def shade_phases(ax, t, phases, alpha=0.10):
    for ph in np.unique(phases):
        mask = phases == ph
        idx  = np.where(mask)[0]
        ax.axvspan(t[idx[0]], t[idx[-1]],
                   color=PHASE_COLORS.get(ph, 'white'), alpha=alpha, zorder=0)


# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================
def run_sim(lambda_CoM_override: float, label: str):
    """
    Full-mission simulation with controlled sloshing parameters.
    lambda_CoM_override = 0  → no feedforward
    lambda_CoM_override = 1  → full compensation
    """
    p = SysParams()

    p.stop_after_phase = 3

    # ── Sloshing parameters (MODIFICATI PER MASSIMIZZARE IL DISTURBO) ────────
    p.CoM_offset = np.array([0.0, 0.0, 0.0])
    p.enable_sloshing = True
    
    # 1. Impatto Fisico: Contenitore leggerissimo, liquido dominante
    p.m_payload       = 1.0    # [kg] massa contenitore ridotta
    p.m_liquid        = 2.0    # [kg] massa liquido aumentata

    # Natural pendulum frequency for cable length L:  f_n = sqrt(g/L) / (2*pi)
    f_natural         = np.sqrt(p.g / p.L) / (2.0 * np.pi)
    p.slosh_freq      = round(f_natural * 0.95, 3)   
    
    # Ampiezza estrema per massimizzare la coppia di disturbo
    p.slosh_amp       = 0.4    # [m]  liquid centroid oscillation amplitude
    p.tau_slosh       = 0.4    # [s]  shorter time constant → more dynamic

    # 2. Indebolimento PID: Rende la Baseline incapace di gestire il disturbo
    p.kp_roll         = 4.0
    p.kp_pitch        = 4.0
    p.kp_yaw          = 2.0
    p.kv_rot          = 2.0
    p.ki_rot          = 0.1

    # 3. Ottimizzatore aggressivo: Permette ai droni di scattare per compensare
    p.w_smooth        = 10.0

    # ── CoM compensation toggle ───────────────────────────────────────────────
    p.lambda_CoM = lambda_CoM_override

    # ── Geometry ─────────────────────────────────────────────────────────────
    # (Il resto del codice rimane invariato da qui in poi...)
    p.uav_offsets, p.attach_vecs, geo_radius = formation.compute_geometry(p)
    dist_h = geo_radius - (min(p.pay_l, p.pay_w) / 2
                           if p.payload_shape in ['box', 'rect'] else p.R_disk)
    dist_h = max(dist_h, 0.1)
    p.safe_altitude_offset = np.sqrt(p.L**2 - dist_h**2) if p.L > dist_h else p.L

    ctx = MissionContext(p)

    # ── Initial conditions ───────────────────────────────────────────────────
    uav_pos_0 = np.zeros((3, p.N))
    for i in range(p.N):
        uav_pos_0[:, i] = p.home_pos + p.uav_offsets[:, i] + np.array([0, 0, p.floor_z])
    pay_pos_0 = p.home_pos + np.array([0, 0, p.floor_z + p.pay_h / 2.0])

    x0 = np.concatenate([
        uav_pos_0.flatten('F'), pay_pos_0, np.zeros(3), np.zeros(3 * p.N),
        np.zeros(3), np.zeros(3), np.zeros(3 * p.N), np.zeros(3)
    ])

    SIM_DURATION = 40.0
    dt    = 0.02
    steps = int(SIM_DURATION / dt)
    N     = p.N

    y_res     = np.zeros((len(x0), steps))
    t_res     = np.zeros(steps)
    phase_res = np.zeros(steps, dtype=int)
    # Log instantaneous liquid CoM offset in body frame
    ex_log    = np.zeros(steps)
    ey_log    = np.zeros(steps)

    current_x = x0.copy();  current_t = 0.0
    last_guid  = -1.0;       guid_dt   = p.optimization_dt

    print(f"  [{label}]  f_slosh={p.slosh_freq:.3f} Hz  f_natural={f_natural:.3f} Hz")
    t0_cpu = time.time()

    for k in range(steps):
        y_res[:, k]  = current_x
        t_res[k]     = current_t
        phase_res[k] = ctx.phase

        if getattr(p, 'enable_sloshing', False):
            omega_s = 2.0 * np.pi * p.slosh_freq
            m_tot   = p.m_payload + p.m_liquid
            ratio   = p.m_liquid / m_tot
            ex_log[k] = ratio * p.slosh_amp * np.sin(omega_s * current_t)
            ey_log[k] = ratio * (p.slosh_amp / 2.0) * np.cos(omega_s * current_t)
        else:
            ex_log[k] = 0.0
            ey_log[k] = 0.0

        if (current_t - last_guid >= guid_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guid = current_t

        n_sub  = 10 if ctx.phase <= 1 else 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            dx        = equations_of_motion(current_t, current_x, p, ctx)
            current_x = current_x + dx * dt_sub
            current_t += dt_sub

        if k % 250 == 0:
            print(f"\r    {current_t / SIM_DURATION * 100:.0f}%  phase={ctx.phase}", end='')

    print(f"\r    done in {time.time() - t0_cpu:.1f}s")
    com_log = np.sqrt(ex_log**2 + ey_log**2) #np.sqrt(ex_log**2 + ey_log**2)
    return t_res, y_res, phase_res, com_log, p


# ==============================================================================
# OSCILLATION AMPLITUDE — high-pass filter to isolate sloshing ripple
# ==============================================================================
def oscillation_amplitude(signal_deg: np.ndarray, window_s: float,
                          dt: float) -> np.ndarray:
    """
    Extract peak-to-peak oscillation amplitude over a sliding window.
    This removes slow trends (programmed attitude) and keeps only the
    fast sloshing ripple.
    """
    win = max(int(window_s / dt), 3)
    amp = np.zeros_like(signal_deg)
    for k in range(len(signal_deg)):
        sl = signal_deg[max(0, k - win): k + 1]
        amp[k] = np.max(sl) - np.min(sl)
    return amp


# ==============================================================================
# TENSION SPREAD
# ==============================================================================
def tension_spread(y, p, t):
    N    = p.N
    dt   = t[1] - t[0]
    uvel = y[3*N+6 : 6*N+6, :]
    g_v  = np.array([[0], [0], [-9.81]])
    T = []
    for i in range(N):
        acc_i = np.gradient(uvel[i*3 : (i+1)*3, :], dt, axis=1)
        F_i   = np.linalg.norm(p.m_drone * (acc_i - g_v), axis=0)
        T.append(F_i)
    T = np.array(T)
    return np.max(T, axis=0) - np.min(T, axis=0)

# ==============================================================================
# SYSTEM POWER (Energy waste proxy)
# ==============================================================================
def system_power(y, p, t):
    """
    Calcola un proxy della potenza meccanica totale del sistema.
    Nei multirotori, la potenza indotta è proporzionale a T^(3/2).
    Restituisce la potenza totale istantanea nel tempo.
    """
    N    = p.N
    dt   = t[1] - t[0]
    uvel = y[3*N+6 : 6*N+6, :]
    g_v  = np.array([[0], [0], [-9.81]])
    
    P_tot = np.zeros(len(t))
    for i in range(N):
        acc_i = np.gradient(uvel[i*3 : (i+1)*3, :], dt, axis=1)
        F_i   = np.linalg.norm(p.m_drone * (acc_i - g_v), axis=0)
        # Potenza del singolo drone proporzionale a T^(1.5)
        P_tot += F_i**(1.5)
    
    return P_tot


# ==============================================================================
# MAIN PLOT
# ==============================================================================
def plot_sloshing_comparison():
    print("Running sloshing analysis  (2 full simulations) …\n")
    t, y_base, ph_base, com_base, p = run_sim(0.0, "Baseline     (λ_CoM = 0)")
    t, y_comp, ph_comp, com_comp, _ = run_sim(1.0, "Compensated  (λ_CoM = 1)")
    print()

    N  = p.N
    dt = t[1] - t[0]

    def att(y): return y[3*N+3 : 3*N+6, :]

    att_base = att(y_base)
    att_comp = att(y_comp)

    phi_base_deg   = np.degrees(att_base[0, :])
    theta_base_deg = np.degrees(att_base[1, :])
    phi_comp_deg   = np.degrees(att_comp[0, :])
    theta_comp_deg = np.degrees(att_comp[1, :])

    # Oscillation amplitude (5-s sliding window → isolates sloshing ripple)
    WIN = 5.0
    osc_phi_base   = oscillation_amplitude(phi_base_deg,   WIN, dt)
    osc_theta_base = oscillation_amplitude(theta_base_deg, WIN, dt)
    osc_phi_comp   = oscillation_amplitude(phi_comp_deg,   WIN, dt)
    osc_theta_comp = oscillation_amplitude(theta_comp_deg, WIN, dt)

    # Combined roll+pitch oscillation magnitude
    osc_base = np.sqrt(osc_phi_base**2 + osc_theta_base**2)
    osc_comp = np.sqrt(osc_phi_comp**2 + osc_theta_comp**2)

    sp_base_raw = tension_spread(y_base, p, t)
    sp_comp_raw = tension_spread(y_comp, p, t)

    # Filtro a media mobile (finestra di 0.5 secondi) per smussare le tensioni nel grafico
    smooth_window = int(0.5 / dt) 
    kernel = np.ones(smooth_window) / smooth_window
    sp_base = np.convolve(sp_base_raw, kernel, mode='same')
    sp_comp = np.convolve(sp_comp_raw, kernel, mode='same')

    # ── Only show from Phase 2 onward (meaningful flight) ────────────────────
    mask = ph_base == 3
    t_show       = t[mask]
    ph_show      = ph_base[mask]
    com_show     = com_base[mask] * 100        # [cm]

    C_BASE = '#1D7EC2'    # vivid red '#1D7EC2'
    C_COMP = '#E63946'    # vivid blue
    ALPHA_LIGHT = 0.35

    fig, axs = plt.subplots(5, 1, figsize=(13, 17), sharex=True)


    # ── Panel A: Raw roll and pitch ───────────────────────────────────────────
    ax = axs[1]
    shade_phases(ax, t[mask], ph_show)
    ax.plot(t[mask], phi_base_deg[mask],   color=C_BASE,  lw=1.3, ls='--', alpha=0.3,
            label='Roll φ – baseline')
    ax.plot(t[mask], theta_base_deg[mask], color=C_BASE,  lw=1.3, ls=':', alpha=0.3, 
            label='Pitch θ – baseline')
    ax.plot(t[mask], phi_comp_deg[mask],   color=C_COMP,  lw=1.3, ls='--',
            label='Roll φ – compensated')
    ax.plot(t[mask], theta_comp_deg[mask], color=C_COMP,  lw=1.3, ls=':',
            label='Pitch θ – compensated')
    ax.axhline(0.0, color='black', lw=0.7, ls=':')
    ax.set_ylabel('Attitude [deg]', fontsize=10)
    ax.set_title('Payload Roll and Pitch',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, ls=':', alpha=0.5)

    # ── Panel B: Oscillation amplitude (sloshing ripple) ─────────────────────
    ax = axs[0]
    shade_phases(ax, t[mask], ph_show)
    ax.plot(t[mask], osc_base[mask], color=C_BASE, lw=1.5,
            label='Oscillation amplitude – baseline')
    ax.plot(t[mask], osc_comp[mask], color=C_COMP, lw=1.5,
            label='Oscillation amplitude – compensated')

    # Annotate mean reduction during stable hover (phase ≥ 3)
    # Annotate mean reduction during the final steady-state hover (last 10 seconds)
    t_end = t[-1]
    mask_steady = (t > t_end - 10.0) & mask
    
    if np.any(mask_steady):
        # Calcoliamo la media solo dove le curve sono diventate piatte e stabili
        mean_b = np.mean(osc_base[mask_steady])
        mean_c = np.mean(osc_comp[mask_steady])
        reduction = (1.0 - mean_c / max(mean_b, 1e-6)) * 100
        
        # Posizioniamo la freccia esattamente a metà della finestra stabile (es. a 35 secondi)
        x_target = t_end - 5.0

        # Disegna una freccia verticale perfetta tra le due linee stabili
        ax.annotate('', xy=(x_target, mean_b), xytext=(x_target, mean_c),
                    arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=1.8))
        
        # Posiziona il testo di fianco alla freccia
        ax.text(x_target - 0.5, (mean_b + mean_c) / 2, 
                f'Reduction: {reduction:.1f}%',
                fontsize=9, color='darkgreen', fontweight='bold', 
                ha='right', va='center')

    ax.set_ylabel('Peak-to-peak [deg]', fontsize=10)
    ax.set_title(
        f'Sloshing Ripple Amplitude',
        fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)

    # ── Panel C: CoM eccentricity ─────────────────────────────────────────────
    ax = axs[2]
    shade_phases(ax, t_show, ph_show)
    ax.plot(t_show, com_show, color='dimgray', lw=1.4,
            label=r'$|e_{\mathrm{CoM}}|$')
    # Annotate amplitude = ratio * amp
    m_tot = p.m_payload + p.m_liquid
    e_amp = (p.m_liquid / m_tot) * p.slosh_amp * 100   # [cm]
    ax.axhline(e_amp, color='steelblue', ls=':', lw=1.2,
               label=rf'Peak $|e_{{CoM}}|$')
    ax.set_ylabel('CoM Offset [cm]', fontsize=10)
    ax.set_title('Instantaneous Liquid CoM Eccentricity',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)

    # ── Panel D: Tension spread ───────────────────────────────────────────────
    ax = axs[3]
    shade_phases(ax, t[mask], ph_show)
    ax.plot(t[mask], sp_base[mask], color=C_BASE, lw=1.2,
            label='$T_{max} − T_{min}$ – baseline')
    ax.plot(t[mask], sp_comp[mask], color=C_COMP, lw=1.2,
            label='$T_{max} − T_{min}$ – compensated')
    #ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('$T_{max} − T_{min}$ [N]', fontsize=10)
    ax.set_title('Cable Tension Spread  (Load Imbalance)',
                 fontweight='bold', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)

    # ── Panel E: System Power (Energy Waste) ──────────────────────────────────
    ax = axs[4]
    shade_phases(ax, t[mask], ph_show)
    
    # Calcolo potenza pura
    pow_base_raw = system_power(y_base, p, t)
    pow_comp_raw = system_power(y_comp, p, t)
    
    # Smussamento (Media mobile di 1 secondo) per leggibilità
    smooth_window_p = int(1.0 / dt) 
    kernel_p = np.ones(smooth_window_p) / smooth_window_p
    pow_base = np.convolve(pow_base_raw, kernel_p, mode='same')
    pow_comp = np.convolve(pow_comp_raw, kernel_p, mode='same')

    ax.plot(t[mask], pow_base[mask], color=C_BASE, lw=1.5,
            label='System Power – baseline')
    ax.plot(t[mask], pow_comp[mask], color=C_COMP, lw=1.5,
            label='System Power – compensated')
            
    # Calcolo energia totale consumata (Integrale della potenza) nella fase di hover stabile
    mask_steady = (t > t[-1] - 10.0) & mask
    if np.any(mask_steady):
        energy_base = np.trapz(pow_base_raw[mask_steady], dx=dt)
        energy_comp = np.trapz(pow_comp_raw[mask_steady], dx=dt)
        
        # Calcolo dell'incremento percentuale di energia
        energy_diff = ((energy_comp / energy_base) - 1.0) * 100
        
        testo_energia = f"Energy overhead: +{abs(energy_diff):.1f}%" if energy_diff > 0 else f"Energy saving: {energy_diff:.1f}%"
        
        # Annotazione sul grafico
        y_pos = max(np.mean(pow_base[mask_steady]), np.mean(pow_comp[mask_steady])) * 1.02
        x_target = t[-1] - 8.0  
        ax.text(x_target, y_pos, testo_energia, 
                fontsize=10, color='indigo', fontweight='bold', ha='right')

    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel(r'Power ($\sum T^{1.5}$)', fontsize=10)
    ax.set_title('Average System Power (Energy Consumption)',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.5)

    ax.set_ylim(bottom=120.0, top=125.0)



    plt.xlim(right=38.0) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Lascia un piccolo margine extra
    
    # Oppure usa questo per un controllo millimetrico:
    plt.subplots_adjust(
        left=0.1,    # Margine sinistro
        right=0.95,  # Margine destro
        top=0.95,    # Margine superiore
        bottom=0.1,  # Aumenta questo valore (es. da 0.05 a 0.1) per la scritta "Time"
        hspace=0.3   # Spazio verticale tra i vari pannelli
    )
    
    plt.show()


if __name__ == "__main__":
    plot_sloshing_comparison()