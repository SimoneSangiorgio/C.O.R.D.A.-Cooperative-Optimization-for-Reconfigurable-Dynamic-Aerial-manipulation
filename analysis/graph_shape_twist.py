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

# =============================================================================
# MAIN SCRIPT - TWIST ANALYSIS
# =============================================================================
def plot_twist_analysis():
    print("Avvio simulazione di equilibrio statico per TWIST (Coppia asse Z)...")
    
    p_base = SysParams()
    p_base.N = 4
    
    # Parametri base payload
    p_base.payload_shape = 'sphere' 
    p_base.R_disk = 0.6
    p_base.pay_h = 0.3
    p_base.m_payload = 1.5
    p_base.m_liquid = 1.5
    
    p_base.CoM_offset = np.array([0.05, 0.05, 0.0])

    # Adattamento e limiti
    p_base.lambda_aero = 0.0 
    p_base.lambda_shape = 0.0 # Disabilitiamo la deformazione ellittica per isolare il twist
    p_base.enable_sloshing = False
    p_base.theta_ref = 30.0
    p_base.max_angle_variation = 10.0
    
    # Sweep del momento torcente (da 0 a 15 Nm)
    # 15 Nm è un momento molto alto per un payload di 3kg, sufficiente a mandare in crisi il sistema standard
    moments_Z = np.linspace(0.0, 7.0, 50)
    
    # Le 2 configurazioni richieste: Twist OFF vs Twist ON
    configs = [
        {
            'id': 0, 'label': r'$\lambda_{twist}=0$ (Rigid)', 
            'l_twist': 0.0,
            'color': '#1D7EC2', 'ls': '-'
        },
        {
            'id': 1, 'label': r'$\lambda_{twist}=4.2$ (Adaptive)', 
            'l_twist': 4.2,
            'color': '#E63946', 'ls': '-'
        }
    ]
    
    # Dizionario per i risultati
    results = {
        c['id']: {
            'mz': [], 'spread': [], 'power': [], 
            'max_thrust': [], 'moment_error': [], 
            'twist_angle': [], 'min_tension': []
        } 
        for c in configs
    }
    
    for c in configs:
        # Inizializza il MockCtx separato per ogni configurazione
        ctx = MockCtx(p_base.theta_ref, p_base.N)
        print(f"Calcolo configurazione: {c['label']} ...")
        
        for mz in moments_Z:
            # Creiamo il momento disturbante puro sull'asse Z
            M_ext_actual = np.array([0.0, 0.0, mz])
            
            p = copy.deepcopy(p_base)
            p.lambda_twist = c['l_twist']
            
            p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
            
            # Equilibrio in Hovering
            state = {
                'pay_pos': np.zeros(3), 
                'pay_att': np.array([0.0, 0.0, 0.0]), 
                'pay_vel': np.zeros(3),
                'pay_omega': np.zeros(3), 
                'uav_vel': np.zeros((3, p.N)),
                'uav_pos': p.uav_offsets.copy(),
                'int_uav': np.zeros((3, p.N))
            }
            
            # Chiamata ottimizzatore, passiamo M_ext_total
            targets, _, alpha_opt, ff_forces = formation.compute_optimal_formation(
                p, state, np.zeros(3), np.zeros(3), 0.0, 
                force_attitude=(0.0, 0.0), 
                F_ext_total=np.zeros(3),      # Niente vento
                M_ext_total=M_ext_actual,     # Inseriamo il momento asse Z
                com_offset_body=np.zeros(3),
                ctx=ctx
            )
            
            # --- 1. Tensioni (Min, Max, Spread) ---
            tensions = np.linalg.norm(ff_forces, axis=0)
            spread = np.max(tensions) - np.min(tensions)
            min_tension = np.min(tensions)
            
            # --- 2. Spinta e Potenza ---
            F_gravity_drones = np.array([0.0, 0.0, -p.m_drone * p.g])
            thrust_mags = [np.linalg.norm(ff_forces[:, i] - F_gravity_drones) for i in range(p.N)]
                
            power = np.sum(np.array(thrust_mags)**1.5)
            max_thrust = np.max(thrust_mags)
            
            # --- 3. Momento Non Compensato (Residuo) ---
            # Momento generato dai cavi
            M_cables = np.zeros(3)
            for i in range(p.N):
                r_arm = p.attach_vecs[:, i]
                M_cables += np.cross(r_arm, ff_forces[:, i])
                
            # Errore (I cavi devono compensare esattamente M_ext_actual)
            # Affinché il payload sia fermo: M_cables + (-M_ext_actual_applicato) = 0
            # Poiché compute_optimal_formation gestisce M_req = ... - M_ext_total,
            # M_cables tenderà a uguagliare M_ext_actual.
            moment_error = np.linalg.norm(M_cables + M_ext_actual)
            
            # --- 4. Calcolo reale dell'angolo di Twist ---
            # Guardiamo di quanto si è spostato orizzontalmente il drone 0 rispetto al suo ancoraggio
            dx, dy = targets[0, 0], targets[1, 0]
            ax, ay = p.attach_vecs[0, 0], p.attach_vecs[1, 0]
            
            angle_uav = np.arctan2(dy, dx)
            angle_att = np.arctan2(ay, ax)
            # Differenza in [-pi, pi]
            twist_rad = (angle_uav - angle_att + np.pi) % (2*np.pi) - np.pi
            twist_deg = abs(np.degrees(twist_rad))
            
            # --- Salvataggio ---
            results[c['id']]['mz'].append(mz)
            results[c['id']]['spread'].append(spread)
            results[c['id']]['power'].append(power)
            results[c['id']]['max_thrust'].append(max_thrust)
            results[c['id']]['moment_error'].append(moment_error)
            results[c['id']]['twist_angle'].append(twist_deg)
            results[c['id']]['min_tension'].append(min_tension)
            
    print("Dati calcolati con successo. Generazione dei grafici (6 plot in totale)...")

    # =========================================================================
    # PLOTTING DEI RISULTATI (2 righe x 3 colonne)
    # =========================================================================
    plt.close('all') # FORZA LA CHIUSURA DELLE FIGURE VECCHIE IN MEMORIA
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    for c in configs:
        cid = c['id']
        x_val = results[cid]['mz']
        kw = {'label': c['label'], 'color': c['color'], 'ls': c['ls'], 'lw': 2.5}
        
        # Prima riga
        axs[0, 0].plot(x_val, results[cid]['spread'], **kw)
        axs[0, 1].plot(x_val, results[cid]['power'], **kw)
        axs[0, 2].plot(x_val, results[cid]['max_thrust'], **kw)
        
        # Seconda riga (Modificata per il Twist)
        axs[1, 0].plot(x_val, results[cid]['moment_error'], **kw)
        axs[1, 1].plot(x_val, results[cid]['twist_angle'], **kw)
        axs[1, 2].plot(x_val, results[cid]['min_tension'], **kw)

    # --- 1. TENSION SPREAD ---
    axs[0, 0].set_title('TENSION SPREAD', fontweight='bold')
    axs[0, 0].set_ylabel('Spread $T_{max} - T_{min}$ [N]')
    axs[0, 0].set_ylim(-0.2, 1.0)
    
    # --- 2. AVERAGE POWER ---
    axs[0, 1].set_title('AVERAGE SYSTEM POWER', fontweight='bold')
    axs[0, 1].set_ylabel('Power ($\sum T^{1.5}$)')
    
    # --- 3. MAX DRONE THRUST ---
    axs[0, 2].set_title('MAX DRONE THRUST', fontweight='bold')
    axs[0, 2].set_ylabel('Max Thrust [N]')
    
    # --- 4. UNCOMPENSATED YAW TORQUE ---
    axs[1, 0].set_title('UNCOMPENSATED YAW TORQUE', fontweight='bold')
    axs[1, 0].set_ylabel('Residual Moment [Nm]')
    max_err = max([max(results[c['id']]['moment_error']) for c in configs])
    axs[1, 0].set_ylim(-0.05, max(1.0, max_err * 1.2))
    axs[1, 0].axhspan(0.5, max(1.0, max_err * 1.5), color='red', alpha=0.1, label='Control limit')
    
    # --- 5. TWIST ANGLE DEFORMATION (Sostituisce Elliptical Def.) ---
    axs[1, 1].set_title('TWIST ANGLE DEFORMATION', fontweight='bold')
    axs[1, 1].set_ylabel(r'Twist Angle $\gamma$ [deg]')
    axs[1, 1].set_ylim(-2.0, 45.0)
    
    # --- 6. MINIMUM CABLE TENSION (Sostituisce Aspect Ratio / Cone Angle) ---
    axs[1, 2].set_title('MINIMUM CABLE TENSION', fontweight='bold')
    axs[1, 2].set_ylabel('Min Tension $T_{min}$ [N]')
    #axs[1, 2].axhline(0.0, color='red', linestyle='-', lw=2, label='Slack Line (Loss of Control)')
    
    # Calcolo T_safe teorico per mostrare la barriera dell'ottimizzatore
    T_safe_theory = p_base.k_safe * ((p_base.m_payload + p_base.m_liquid) * p_base.g / p_base.N)
    #axs[1, 2].axhline(T_safe_theory, color='orange', linestyle=':', lw=2, label='Safety Barrier $T_{safe}$')
    
    axs[1, 2].set_ylim(8.4, 9.0)
    
    # --- Stile Comune ---
    for ax in axs.flat:
        ax.set_xlabel('External Yaw Moment [Nm]')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_twist_analysis()