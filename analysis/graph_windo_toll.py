import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os

# Assicuriamo che la directory corrente sia nel path per importare i moduli caricati
sys.path.append('.')

try:
    from parameters import SysParams
    import formation as formation
    import physics
except ImportError:
    print("Errore: Assicurati che parameters.py, formation_coupled.py e physics.py siano nella stessa cartella.")
    sys.exit(1)

def calculate_static_thrust(p_in, wind_speed, l_shape, l_aero):
    """
    Simula una condizione di hovering statico con vento costante
    e calcola la spinta totale richiesta dai droni.
    """
    # Clona parametri per non modificare l'oggetto originale
    p = copy.deepcopy(p_in)
    p.lambda_shape = l_shape
    p.lambda_aero = l_aero
    
    # 1. Setup Geometria Iniziale
    p.uav_offsets, p.attach_vecs, p.geo_radius = formation.compute_geometry(p)
    
    # 2. Definisci Vento
    wind_vec = np.array([wind_speed, 0.0, 0.0])
    
    # Stato fittizio per l'ottimizzatore (hovering statico)
    state = {
        'pay_pos': np.zeros(3),
        'pay_att': np.zeros(3), 
        'pay_vel': np.zeros(3),
        'pay_omega': np.zeros(3),
        'uav_vel': np.zeros((3, p.N))
    }
    
    # 3. Stima Forza Vento sul Payload
    # Nota: L'ottimizzatore formation_coupled ricalcola l'assetto aerodinamico se lambda_aero=1
    # Qui forniamo una stima iniziale della forza esterna per far convergere il solver.
    if p.payload_shape in ['box', 'rect', 'square']:
        # Stima area frontale massima (caso peggiore iniziale)
        area_ref = p.pay_w * p.pay_h 
    elif p.payload_shape == 'sphere':
        area_ref = np.pi * p.R_disk**2
    else: # Cylinder
        area_ref = 2 * p.R_disk * p.pay_h
        
    f_drag_mag = 0.5 * p.rho * p.Cd_pay * area_ref * (wind_speed**2)
    F_wind_guess = np.array([f_drag_mag, 0.0, 0.0])
    
    # 4. Esegui Ottimizzatore
    # Restituisce, tra le altre cose, le forze ff_forces (Tensione * Versore) esercitate dai cavi SUL carico
    target_pos, L_winch, alpha_opt, ff_forces = formation.compute_optimal_formation(
        p, state, 
        acc_cmd_pay=np.zeros(3), 
        acc_ang_cmd_pay=np.zeros(3), 
        ref_yaw=0.0,
        force_attitude=(0.0, 0.0), # Se aero=1, questo viene ignorato/ricalcolato internamente
        F_ext_total=F_wind_guess
    )
    
    # 5. Calcolo Thrust Totale dei Droni
    # Dobbiamo considerare anche la resistenza del vento sui droni stessi
    f_drag_drone_mag = 0.5 * p.rho * p.Cd_uav * p.A_uav * (wind_speed**2)
    F_wind_drone = np.array([f_drag_drone_mag, 0.0, 0.0])
    F_g_drone = np.array([0.0, 0.0, -p.m_drone * p.g])
    
    total_thrust = 0.0
    
    for i in range(p.N):
        # La forza che il drone deve esercitare deve bilanciare:
        # - La tensione del cavo (che lo tira verso il carico)
        # - La sua gravità
        # - Il vento che lo spinge indietro
        
        # ff_forces[:, i] è la forza del cavo SUL carico.
        # La forza sul drone è opposta: -ff_forces
        F_cable_pull = -ff_forces[:, i]
        
        # Equilibrio vettoriale: T_thrust + F_cable + F_g + F_wind = 0
        T_req_vec = -(F_cable_pull + F_g_drone + F_wind_drone)
        
        total_thrust += np.linalg.norm(T_req_vec)
        
    return total_thrust

# ==========================================
# SETUP PARAMETRI E SIMULAZIONE
# ==========================================
wind_range = np.linspace(0, 25, 30) # Da 0 a 25 m/s

# --- GRAFICO 1: CONFRONTO STRATEGIE (Standard Box) ---
p_base = SysParams()
p_base.payload_shape = 'box'
p_base.pay_l = 0.5; p_base.pay_w = 0.5; p_base.pay_h = 0.5
p_base.m_payload = 3.0

y_shape = [] # Lambda Shape=1, Aero=0
y_aero = []  # Lambda Shape=0, Aero=1

for w in wind_range:
    y_shape.append(calculate_static_thrust(p_base, w, l_shape=1.0, l_aero=0.0))
    y_aero.append(calculate_static_thrust(p_base, w, l_shape=0.0, l_aero=1.0))

# --- GRAFICO 2: CONFRONTO GEOMETRIE (Box, Sfera, Cilindro) ---
# Usiamo una strategia ottimizzata fissa (Aero=1, Shape=1) per vedere le differenze fisiche

# 1. Sfera
p_sphere = SysParams()
p_sphere.payload_shape = 'sphere'
p_sphere.R_disk = 0.4 
p_sphere.m_payload = 3.0

# 2. Cilindro
p_cyl = SysParams()
p_cyl.payload_shape = 'cylinder'
p_cyl.R_disk = 0.3; p_cyl.pay_h = 1.0
p_cyl.m_payload = 3.0

# 3. Box Flat (Aerodinamico) -> Area esposta ~0.2 m^2
p_flat = SysParams()
p_flat.payload_shape = 'box'
p_flat.pay_l = 1.0 # X
p_flat.pay_w = 1.0 # Y
p_flat.pay_h = 0.2 # Z (Lato sottile)
p_flat.m_payload = 3.0

# 4. Box Tall (Muro) -> Area esposta ~1.0 m^2
p_tall = SysParams()
p_tall.payload_shape = 'box'
p_tall.pay_l = 0.2 # X
p_tall.pay_w = 1.0 # Y
p_tall.pay_h = 1.0 # Z (Lato alto)
p_tall.m_payload = 3.0

y_sphere_vals = []
y_cyl_vals = []
y_flat_vals = []
y_tall_vals = []

for w in wind_range:
    # Aero=1 permette al "Flat" di mettersi di taglio, ma non aiuta la Sfera
    y_sphere_vals.append(calculate_static_thrust(p_sphere, w, 1.0, 1.0))
    y_cyl_vals.append(calculate_static_thrust(p_cyl, w, 1.0, 1.0))
    y_flat_vals.append(calculate_static_thrust(p_flat, w, 1.0, 1.0))
    y_tall_vals.append(calculate_static_thrust(p_tall, w, 1.0, 1.0))

# ==========================================
# PLOTTING
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1
axs[0].plot(wind_range, y_shape, 'b--', linewidth=2, label=r'Shape Priority ($\lambda_{shape}=1, \lambda_{aero}=0$)')
axs[0].plot(wind_range, y_aero, 'r-', linewidth=2, label=r'Aero Priority ($\lambda_{shape}=0, \lambda_{aero}=1$)')
axs[0].set_title("Efficienza Controllo (Carico Cubico)", fontsize=14)
axs[0].set_xlabel("Velocità Vento [m/s]", fontsize=12)
axs[0].set_ylabel("Thrust Totale [N]", fontsize=12)
axs[0].grid(True, linestyle=':', alpha=0.6)
axs[0].legend()

# Plot 2
axs[1].plot(wind_range, y_sphere_vals, 'g-', linewidth=2, label='Sfera (R=0.4m)')
axs[1].plot(wind_range, y_cyl_vals, 'orange', linewidth=2, label='Cilindro (R=0.3, H=1.0)')
axs[1].plot(wind_range, y_flat_vals, 'purple', linestyle='-.', linewidth=2, label='Box Piatto (Area Min)')
axs[1].plot(wind_range, y_tall_vals, 'k:', linewidth=2, label='Box Alto (Area Max)')

axs[1].set_title("Impatto Geometria (Controllo Ottimizzato)", fontsize=14)
axs[1].set_xlabel("Velocità Vento [m/s]", fontsize=12)
axs[1].set_ylabel("Thrust Totale [N]", fontsize=12)
axs[1].grid(True, linestyle=':', alpha=0.6)
axs[1].legend()

plt.tight_layout()
plt.show()