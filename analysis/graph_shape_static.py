import sys
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

# Aggiungi il percorso radice per importare i moduli del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import formation as formation
from parameters import SysParams

# =============================================================================
# CONTEXT MOCK & FUNZIONI DI SUPPORTO
# =============================================================================
class MockCtx:
    """Oggetto contesto minimo per mantenere lo stato dell'ottimizzatore continuo."""
    def __init__(self, theta_ref_deg: float, N: int):
        self.last_theta_opt       = np.radians(theta_ref_deg)
        self.last_stable_yaw      = 0.0
        self.last_damp_correction = np.zeros(N)
        self.last_T_final         = np.zeros(N)
        self.dot_T_filt           = np.zeros(N)
        self.F_xy_filt            = np.zeros(2)
        self.e_filt_despun        = np.zeros(3)

def get_v_for_force(p, force_target_N):
    """Calcola la velocità del vento necessaria per ottenere una data forza in Newton."""
    if force_target_N < 1e-3: return 0.0
    
    if getattr(p, 'payload_shape', '') == 'sphere':
        pay_proj_area = np.pi * (p.R_disk**2)
    elif p.payload_shape in ['box', 'rect', 'square']:
        pay_proj_area = p.pay_w * p.pay_h
    else:
        pay_proj_area = 2.0 * p.R_disk * p.pay_h
        
    k_drag = 0.5 * p.rho * p.Cd_pay * pay_proj_area
    return np.sqrt(force_target_N / k_drag)

# =============================================================================
# MAIN SCRIPT
# =============================================================================
def plot_combined_analysis():
    print("Avvio simulazione di equilibrio statico...")
    
    p_base = SysParams()
    p_base.N = 4
    
    # Parametri base payload
    p_base.payload_shape = 'sphere' 
    p_base.R_disk = 0.6
    p_base.pay_h = 0.3
    p_base.m_payload = 2.0
    p_base.m_liquid = 1.0
    
    # Adattamento e limiti
    p_base.lambda_aero = 0.0 
    p_base.enable_sloshing = False
    p_base.F_ref = 10.0
    p_base.CoP = np.array([0.0, 0.0, 0.02]) 
    p_base.theta_ref = 30.0
    p_base.max_angle_variation = 10.0
    
    wind_forces_N = np.linspace(0.0, (p_base.m_payload+p_base.m_liquid)*9.8+5, 50)
    
    # Le 3 configurazioni richieste
    configs = [
        {
            'id': 0, 'label': r'$\lambda_{shape}=0$', 
            'l_shape': 0.0, 'l_par': 1.0, 'l_perp': 1.0, 
            'color': '#1D7EC2', 'ls': '-'
        },
        {
            'id': 1, 'label': r'$\lambda_{shape}=1$ ($\lambda_\parallel=2.0, \lambda_\perp=0.5$)', 
            'l_shape': 1.0, 'l_par': 2.0, 'l_perp': 0.5, 
            'color': '#E63946', 'ls': ':'
        },
        {
            'id': 2, 'label': r'$\lambda_{shape}=1$ ($\lambda_\parallel=0.5, \lambda_\perp=2.0$)', 
            'l_shape': 1.0, 'l_par': 0.5, 'l_perp': 2.0, 
            'color': '#E63946', 'ls': '--'
        }
    ]
    
    # Dizionario per i risultati (sostituito alpha_opt con aspect_ratio)
    results = {
        c['id']: {'wind_force': [], 'spread': [], 'power': [], 'max_thrust': [], 'moment_error': [], 'eccentricity': [], 'aspect_ratio': []} 
        for c in configs
    }
    
    for c in configs:
        # Inizializza il MockCtx separato per ogni configurazione
        ctx = MockCtx(p_base.theta_ref, p_base.N)
        print(f"Calcolo configurazione: {c['label']} ...")
        
        for f_target in wind_forces_N:
            v_mag = get_v_for_force(p_base, f_target)
            v_wind_vec = np.array([v_mag, 0.0, 0.0])
            F_wind_actual = np.array([f_target, 0.0, 0.0]) 
            
            p = copy.deepcopy(p_base)
            p.lambda_shape = c['l_shape']
            p.lambda_par = c['l_par']
            p.lambda_perp = c['l_perp']
            
            p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
            
            phi_tgt, theta_tgt = 0.0, 0.0 
            
            state = {
                'pay_pos': np.zeros(3), 
                'pay_att': np.array([phi_tgt, theta_tgt, 0.0]), 
                'pay_vel': np.zeros(3),
                'pay_omega': np.zeros(3), 
                'uav_vel': np.zeros((3, p.N)),
                'uav_pos': p.uav_offsets.copy(),
                'int_uav': np.zeros((3, p.N))
            }
            
            # Chiamata ottimizzatore, che restituisce targets (impronta X-Y) e alpha_opt
            targets, _, alpha_opt, ff_forces = formation.compute_optimal_formation(
                p, state, np.zeros(3), np.zeros(3), 0.0, 
                force_attitude=(phi_tgt, theta_tgt), 
                F_ext_total=F_wind_actual,
                com_offset_body=getattr(p, 'CoM_offset', np.zeros(3)),
                ctx=ctx
            )
            
            # --- 1. Tensioni ---
            tensions = np.linalg.norm(ff_forces, axis=0)
            spread = np.max(tensions) - np.min(tensions)
            
            # --- 2. Spinta e Potenza ---
            F_gravity_drones = np.array([0.0, 0.0, -p.m_drone * p.g])
            if v_mag > 1e-3:
                wind_dir = v_wind_vec / v_mag
                f_drag_single = 0.5 * p.rho * p.Cd_uav * p.A_uav * (v_mag**2)
                F_wind_drone = f_drag_single * wind_dir
            else:
                F_wind_drone = np.zeros(3)
                
            thrust_mags = []
            for i in range(p.N):
                t_vec = ff_forces[:, i] - F_gravity_drones - F_wind_drone
                thrust_mags.append(np.linalg.norm(t_vec))
                
            power = np.sum(np.array(thrust_mags)**1.5)
            max_thrust = np.max(thrust_mags)
            
            # --- 3. Momento Non Compensato ---
            M_wind = np.cross(p.CoP, F_wind_actual)
            M_cables = np.zeros(3)
            for i in range(p.N):
                r_arm = p.attach_vecs[:, i] - getattr(p, 'CoM_offset', np.zeros(3))
                M_cables += np.cross(r_arm, ff_forces[:, i])
                
            moment_error = np.linalg.norm(M_cables + M_wind)
            
            # --- 4. Eccentricity e Aspect Ratio ---
            xs = targets[0, :]
            ys = targets[1, :]
            dim_par  = max(np.max(xs) - np.min(xs), 1e-3)
            dim_perp = max(np.max(ys) - np.min(ys), 1e-3)
            
            ratio = dim_perp / dim_par
            
            a = max(dim_par, dim_perp) / 2.0
            b = min(dim_par, dim_perp) / 2.0
            e = np.sqrt(1.0 - (b / a) ** 2)
            
            # --- Salvataggio ---
            results[c['id']]['wind_force'].append(f_target)
            results[c['id']]['spread'].append(spread)
            results[c['id']]['power'].append(power)
            results[c['id']]['max_thrust'].append(max_thrust)
            results[c['id']]['moment_error'].append(moment_error)
            results[c['id']]['eccentricity'].append(e)
            results[c['id']]['aspect_ratio'].append(ratio)
            
    print("Dati calcolati con successo. Generazione dei grafici (6 plot in totale)...")

    # =========================================================================
    # PLOTTING DEI RISULTATI (2 righe x 3 colonne)
    # =========================================================================
    plt.close('all') # FORZA LA CHIUSURA DELLE FIGURE VECCHIE IN MEMORIA
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    for c in configs:
        cid = c['id']
        x_val = results[cid]['wind_force']
        kw = {'label': c['label'], 'color': c['color'], 'ls': c['ls'], 'lw': 2.5}
        
        # Prima riga (TENSION SPREAD, AVERAGE POWER, MAX DRONE THRUST)
        axs[0, 0].plot(x_val, results[cid]['spread'], **kw)
        axs[0, 1].plot(x_val, results[cid]['power'], **kw)
        axs[0, 2].plot(x_val, results[cid]['max_thrust'], **kw)
        
        # Seconda riga (UNCOMPENSATED PITCH TORQUE, ELLIPTICAL DEFORMATION, ASPECT RATIO)
        axs[1, 0].plot(x_val, results[cid]['moment_error'], **kw)
        axs[1, 1].plot(x_val, results[cid]['eccentricity'], **kw)
        axs[1, 2].plot(x_val, results[cid]['aspect_ratio'], **kw)

    # --- 1. TENSION SPREAD ---
    axs[0, 0].set_title('Cable Tension Spread', fontweight='bold')
    axs[0, 0].set_ylabel('$T_{max} - T_{min}$ [N]')
    
    # --- 2. AVERAGE POWER ---
    axs[0, 1].set_title('Average System Power', fontweight='bold')
    axs[0, 1].set_ylabel('Power ($\sum T^{1.5}$)')
    
    # --- 3. MAX DRONE THRUST ---
    axs[0, 2].set_title('Max Single Drone Thrust Required', fontweight='bold')
    axs[0, 2].set_ylabel('Max Thrust [N]')
    if hasattr(p_base, 'F_max_thrust'):
        axs[0, 2].axhline(p_base.F_max_thrust, color='gray', linestyle=':', lw=2, label='Hardware Limit')
    
    # --- 4. UNCOMPENSATED PITCH TORQUE ---
    axs[1, 0].set_title('Uncompensated Pitch Torque', fontweight='bold')
    axs[1, 0].set_ylabel('Moment Error [Nm]')
    max_err = max([max(results[c['id']]['moment_error']) for c in configs])
    if max_err < 1.0: max_err = 1.0
    axs[1, 0].axhspan(0.5, max_err + 1.0, color='red', alpha=0.1, label='Attitude Control Lost')
    axs[1, 0].set_ylim(-0.02, 0.7)
    
    # --- 5. ELLIPTICAL DEFORMATION ---
    axs[1, 1].set_title('Elliptical Deformation', fontweight='bold')
    axs[1, 1].set_ylabel('Eccentricity [0 - 1]')
    axs[1, 1].set_ylim(-0.05, 1.05)
    
    # --- 6. ASPECT RATIO ---
    axs[1, 2].set_title('Aspect Ratio', fontweight='bold')
    axs[1, 2].set_ylabel('Aspect Ratio $d_\perp / d_\parallel$')
    
    # --- Stile Comune ---
    for ax in axs.flat:
        ax.set_xlabel('Horizontal Force [N]')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_combined_analysis()