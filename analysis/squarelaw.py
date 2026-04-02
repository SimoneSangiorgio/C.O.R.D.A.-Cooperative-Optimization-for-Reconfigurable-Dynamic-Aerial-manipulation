import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')

# --- DATA FOR CHART 1: Scalability (Square-Cube Law vs Linear) ---
# Simulating the required lifting capacity from 1 to 10 units
required_load = np.linspace(1, 10, 100)

# Single UAV: Cost/mass grows exponentially due to the square-cube law
# (If you double the size for more lift, the volume/weight grows by the cube)
single_uav_cost_mass = required_load**1.8  

# CAM System: Adding drones scales the cost/mass purely linearly
cam_cost_mass = required_load * 2.5 

# --- DATA FOR CHART 2: Redundancy (Mechanical Decoupling) ---
number_of_failures = np.arange(0, 6)
# Single UAV: A critical failure drops lifting capacity to zero
single_capacity = [100, 0, 0, 0, 0, 0] 
# CAM System (e.g., fleet of 5 drones): Losing one drone only partially reduces capacity
cam_capacity = [100, 80, 60, 40, 20, 0] 

# --- FIGURE CREATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Inefficiency vs Scalability
ax1.plot(required_load, single_uav_cost_mass, label='Single Large UAV (Square-Cube Law)', color='#e63946', linewidth=3)
ax1.plot(required_load, cam_cost_mass, label='CAM System (Linear Scalability)', color='#2a9d8f', linewidth=3)
ax1.set_title("1. Cost/Mass vs Lifting Capacity\n(Scalability)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Desired Payload Capacity", fontsize=12)
ax1.set_ylabel("System Cost / Mass", fontsize=12)
ax1.legend(fontsize=11)

# Chart 2: Reliability and Redundancy
ax2.step(number_of_failures, single_capacity, label='Single UAV (No redundancy)', color='#e63946', where='post', linewidth=3)
ax2.plot(number_of_failures, cam_capacity, label='CAM System (Built-in redundancy)', color='#2a9d8f', marker='o', markersize=8, linewidth=3)
ax2.set_title("2. Failure Safety\n(Mechanical Decoupling)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Number of Motor / Agent Failures", fontsize=12)
ax2.set_ylabel("Remaining Lifting Capacity (%)", fontsize=12)
ax2.legend(fontsize=11)
ax2.set_xticks(number_of_failures)

plt.tight_layout()
plt.show()