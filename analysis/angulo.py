import numpy as np
import matplotlib.pyplot as plt

# 1. Definizione degli angoli (da 0 a 60 gradi per non far esplodere l'asintoto a 90)
alpha_deg = np.linspace(0, 60, 500)
alpha_rad = np.radians(alpha_deg)

# 2. Calcolo della "Tassa Energetica" (Thrust Penalty in %)
# Formula: (1 / cos(alpha) - 1) * 100
thrust_ratio = 1 / np.cos(alpha_rad)
penalty_percent = (thrust_ratio - 1) * 100

# 3. Creazione del grafico
plt.figure(figsize=(8, 5))
plt.plot(alpha_deg, penalty_percent, 'b-', linewidth=2, label='Energy Cost / Thrust Penalty')

# 4. Calcolo e plottaggio dei punti chiave (30° e 45°)
# Angolo 30°
penalty_30 = (1 / np.cos(np.radians(30)) - 1) * 100
plt.plot(30, penalty_30, 'ro', markersize=8)
plt.annotate(f'30° ($\\approx$ +{penalty_30:.0f}%)', 
             xy=(30, penalty_30), xytext=(15, penalty_30 + 15),
             fontsize=11, fontweight='bold', color='darkred',
             arrowprops=dict(facecolor='darkred', shrink=0.05, width=1.5, headwidth=7))

# Angolo 45°
penalty_45 = (1 / np.cos(np.radians(45)) - 1) * 100
plt.plot(45, penalty_45, 'ro', markersize=8)
plt.annotate(f'45° ($\\approx$ +{penalty_45:.0f}%)', 
             xy=(45, penalty_45), xytext=(30, penalty_45 + 20),
             fontsize=11, fontweight='bold', color='darkred',
             arrowprops=dict(facecolor='darkred', shrink=0.05, width=1.5, headwidth=7))

# 5. Formattazione estetica per la tesi
plt.title('Energy-Stability Trade-off: Thrust Penalty vs. Cone Angle ($\\alpha$)', fontsize=13, pad=15)
plt.xlabel('Cone Angle $\\alpha$ (degrees)', fontsize=12)
plt.ylabel('Extra Thrust Required (%)', fontsize=12)

# Abbellimenti
plt.grid(True, linestyle='--', alpha=0.7)
plt.fill_between(alpha_deg, 0, penalty_percent, color='blue', alpha=0.05) # Sfondo ombreggiato
plt.xlim(0, 60)
plt.ylim(0, 100)
plt.legend(loc='upper left', fontsize=11)

# Ottimizzazione dei margini per l'esportazione
plt.tight_layout()

# 6. Salva e mostra
# Salva il grafico in alta risoluzione, ideale per LaTeX
plt.savefig('energy_tradeoff_alpha.pdf', format='pdf', dpi=300)
plt.savefig('energy_tradeoff_alpha.png', format='png', dpi=300)
plt.show()