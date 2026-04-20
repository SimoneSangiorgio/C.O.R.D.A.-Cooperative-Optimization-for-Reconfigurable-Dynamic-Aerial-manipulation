import numpy as np
from scipy.integrate import solve_ivp
import time

# Formation
import formation 

# Physics & Control
from mission import MissionContext, equations_of_motion, update_guidance
from parameters import SysParams
# Analysis
import system_simulation

# --- CONFIGURAZIONE VISUALIZZAZIONE ---
# Inserisci qui le fasi che vuoi vedere nell'animazione.
# Es: [4, 5] mostrerà solo dalla fase 4 alla 5 inclusa.
# Lascia None o una lista vuota per vedere tutto.
VISUALIZATION_PHASES = [0,1,2,3,4,5,6] 

if __name__ == "__main__":
    p = SysParams()
    # Inizializzazione Geometria
    p.uav_offsets, p.attach_vecs, geo_radius = formation.compute_geometry(p)
    dist_h = geo_radius - (min(p.pay_l, p.pay_w)/2 if p.payload_shape in ['box','rect'] else p.R_disk)
    if dist_h < 0.1: dist_h = 0.1 
    if p.L > dist_h: safe_h = np.sqrt(p.L**2 - dist_h**2)
    else: safe_h = p.L 
    p.safe_altitude_offset = safe_h 
    
    ctx = MissionContext(p)
    
    # Setup Stato Iniziale
    uav_pos_0 = np.zeros((3, p.N))
    for i in range(p.N):
        uav_pos_0[:, i] = p.home_pos + p.uav_offsets[:, i] + np.array([0,0, p.floor_z]) 
    pay_pos_0 = p.home_pos + np.array([0,0, p.floor_z + p.pay_h / 2.0])
    
    x0 = np.concatenate([
        uav_pos_0.flatten('F'), pay_pos_0, np.zeros(3), np.zeros(3*p.N),
        np.zeros(3), np.zeros(3), np.zeros(3*p.N), np.zeros(3)
    ])
    
    SIM_DURATION = 90.0   
    dt = 0.02
    steps = int(SIM_DURATION / dt)
    
    y_res = np.zeros((len(x0), steps))
    t_res = np.zeros(steps)
    phase_res = np.zeros(steps, dtype=int)
    thrust_res = np.zeros((p.N, steps))
    current_x = x0.copy()
    current_t = 0.0
    
    print(f"Starting Simulation N={p.N}...")
    start_cpu_time = time.time()
    
    last_phase = -1
    
    last_guidance_time = -1.0
    guidance_dt = p.optimization_dt  # Es: 0.04s (25 Hz)

    

    for k in range(steps):
        # 0. Salvataggio dati per plot
        y_res[:, k] = current_x
        t_res[k] = current_t
        phase_res[k] = ctx.phase

        # === 1. GUIDANCE STEP (Intelligenza) ===
        if (current_t - last_guidance_time >= guidance_dt) or (k == 0):
            update_guidance(current_t, current_x, p, ctx)
            last_guidance_time = current_t
        
        # === 2. PHYSICS SUB-STEPPING (High Freq Physics, Low Cost) ===
        # Spezziamo il dt grafico (0.01s) in 5 passi fisici (0.002s)
        # Questo rende i contatti col suolo stabili senza rallentare il PC.
        if ctx.phase <= 1:
            physics_substeps = 10
        else:
            physics_substeps = 5
            
        dt_sub = dt / physics_substeps
        
        for _ in range(physics_substeps):
            # Calcola le derivate
            dx = equations_of_motion(current_t, current_x, p, ctx)
            
            # Integrazione (Eulero)
            current_x = current_x + dx * dt_sub
            
            # Avanzamento tempo fisico
            current_t += dt_sub

        if hasattr(ctx, 'last_u_acc'):
            for i in range(p.N):
                acc_cmd = ctx.last_u_acc[:, i]
                # Aggiungiamo gravità per ottenere la spinta totale (in Newton)
                thrust_vec = p.m_drone * (acc_cmd + np.array([0, 0, 9.81]))
                thrust_res[i, k] = np.linalg.norm(thrust_vec)

        
        # Logging cambio fase
        if ctx.phase != last_phase:
             pass
        last_phase = ctx.phase

        if k % 50 == 0:
            percent = (current_t / SIM_DURATION) * 100
            print(f"\rProgress: {percent:5.1f}% | Time: {current_t:5.2f}s | Phase: {ctx.phase}", end="")

    print(f"\nSimulation finished in {time.time() - start_cpu_time:.2f}s")
    
    # --- FILTRAGGIO DATI PER L'ANIMAZIONE ---
    if VISUALIZATION_PHASES is not None and len(VISUALIZATION_PHASES) > 0:
        print(f"\nFiltering data for phases: {VISUALIZATION_PHASES}")
        # Maschera booleana: True dove la fase è nella lista richiesta
        mask = np.isin(phase_res, VISUALIZATION_PHASES)
        
        if np.any(mask):
            t_anim = t_res[mask]
            y_anim = y_res[:, mask]
            print(f"Showing {len(t_anim)} steps out of {steps} total.")
        else:
            print("WARNING: No data found for requested phases. Showing full simulation.")
            t_anim = t_res
            y_anim = y_res
    else:
        t_anim = t_res
        y_anim = y_res

    # --- RESAMPLING & ANIMATION ---
    vis_fps = 30
    # Calcoliamo sim_fps basandoci sul dt originale per coerenza
    sim_fps = int(1.0 / dt) 
    step_skip = max(1, int(sim_fps / vis_fps))
    
    if VISUALIZATION_PHASES is not None and len(VISUALIZATION_PHASES) > 0 and np.any(mask):
        thrust_anim = thrust_res[:, mask]
    else:
        thrust_anim = thrust_res

    print(f"Resampling data for animation: {sim_fps}Hz -> {vis_fps}Hz (Skip={step_skip})")
    
    # Passiamo t_anim e y_anim filtrati
    system_simulation.animate(t_anim[::step_skip], y_anim[:, ::step_skip], p, 
                              theta_log=ctx.theta_log, 
                              thrust_log=thrust_anim[:, ::step_skip]
                              )