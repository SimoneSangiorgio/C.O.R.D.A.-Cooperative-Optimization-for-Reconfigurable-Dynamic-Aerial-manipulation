import numpy as np

class SysParams:
    """
    System Parameters for Multi-UAV Payload Transportation.
    
    ORGANIZATION:
    1. Simulation Settings (Time, Phases)
    2. Environment (Gravity, Wind, Gusts)
    3. Multi-Agent Geometry (N, Cable Length)
    4. UAV Hardware (Mass, Thrust, Aero)
    5. Payload Properties (Mass, Inertia, Sloshing)
    6. Cable Physics (Stiffness, Damping)
    7. Control Gains (PID, Active Damping)
    8. Optimization & Formation (Cost Function, Weights)
    9. Mission & Navigation (Waypoints, Timing)
    """
    def __init__(self):

        # ==============================================================================
        # 1. SIMULATION & GENERAL SETTINGS
        # ==============================================================================
        # Impact: Controls the resolution of the physics engine.
        # - Lower: More accurate physics, but simulation runs slower.
        # - Higher: Faster simulation, but may miss collisions or cause instability.
        self.optimization_dt = 0.05  # Time step for the high-level formation optimizer [s]
        
        # Impact: Determines when the simulation automatically stops.
        # Phases: 0=Idle, 1=Lift, 2=Align, 3=Attitude, 4=Transport, 5=Winch, 6=Hold
        self.stop_after_phase = 6
        
        # Impact: Smooths out the control inputs to avoid "jerky" movements.
        self.input_shaping_enabled = True
        
        # Impact: Allows the formation to shift strictly based on momentum calculations.
        self.enable_momentum_shift = True

        # Impact: Shifts the drone formation center to align with the actual Center of Mass (CoM).
        self.enable_com_geometric_shift = True

        self.yaw_tracking_enabled = False


        # ==============================================================================
        # 2. ENVIRONMENT
        # ==============================================================================
        self.g = 9.81           # Gravity acceleration [m/s^2]
        self.rho = 1.225        # Air density [kg/m^3] (Standard sea level air)
        self.floor_z = 0.0      # Z-coordinate of the ground [m]

        # --- Wind Configuration ---
        # Impact: Sets the base constant wind velocity.
        # Note: If 'enable_gusts' is True, this base wind is added to the gusts.
        self.initial_wind_vec = np.array([3, 5, 0.0]) 
        
        # Impact: Rotates the wind vector over time (e.g., swirling wind).
        # - Non-zero: Wind direction changes continuously.
        self.Rot_wind = np.array([0.0, 0.0, 0.0])

        self.wind_vel = np.array([0.0, 0.0, 0.0]) # Internal variable (do not set manually)

        # --- Gusts Schedule ---
        # Impact: If True, the simulation uses the specific list of gusts below.
        self.enable_gusts = False

        self.gust_schedule = [
            # Gust 1: Lateral push
            {'t_start': 10.0, 'duration': 5.0, 'vec': np.array([0.0, 2.0, 0.0]), 'ramp': 1.0},
            # Gust 2: Strong headwind
            {'t_start': 25.0, 'duration': 8.0, 'vec': np.array([2.0, 0.0, 0.0]), 'ramp': 2.0},
            # Gust 3: Vertical turbulence
            {'t_start': 40.0, 'duration': 4.0, 'vec': np.array([1.0, 1.0, -1.0]), 'ramp': 0.5},
        ]


        # ==============================================================================
        # 3. SYSTEM GEOMETRY (Multi-Agent)
        # ==============================================================================
        self.N = 4   # Number of UAVs
        
        # Impact: The nominal length of the cables.
        # - Longer: More pendulum swing, slower oscillation frequency.
        # - Shorter: More rigid system, faster oscillations, higher risk of drone collision.
        self.L = 3.0          # Cable length [m]


        # ==============================================================================
        # 4. UAV HARDWARE PARAMETERS
        # ==============================================================================
        self.m_drone = 1.0      # Mass of a single UAV [kg]
        
        # --- Aerodynamics & Propulsion ---
        self.Cd_uav = 0.5       # Drag coefficient of the drone body
        self.A_uav = 0.1        # Reference Area for drag [m^2]
        self.prop_radius = 0.15 # Propeller radius [m] (Used for ground effect/downwash)
        self.downwash_cone_angle = np.radians(15) # Angle of air pushed down by props
        
        # --- Physical Limits ---
        # Impact: Limits how fast the drone can change acceleration.
        # - Lower: Drone moves smoother/robotic.
        # - Higher: Drone moves aggressively.
        self.jerk_limit = 3.0   # [m/s^3]
        
        # Impact: The ceiling of force a drone can produce.
        # If the optimizer requests more than this, the command is clipped (saturated).
        self.F_max_thrust = 45.0 # [N]


        # ==============================================================================
        # 5. PAYLOAD PARAMETERS
        # ==============================================================================
        # --- Physical Dimensions ---
        # Options: 'box', 'rect', 'square', 'cylinder' 'sphere'
        # Impact: Changes how the aerodynamic area and inertia are calculated.
        self.payload_shape = 'cylinder'
        
        self.m_payload = 1.5    # Mass of the solid container [kg]
        self.m_liquid = 1.5     # Mass of the liquid inside [kg]
        
        self.pay_h = 0.2        # Height [m]
        self.pay_l = 0.5        # Length [m] (x-axis)
        self.pay_w = 0.9        # Width [m] (y-axis)
        self.R_disk = 0.6       # Radius [m] (Only if shape is cylinder)
        
        # --- Aerodynamics ---
        if self.payload_shape in ['box', 'rect', 'square']:
            self.A_pay_z = self.pay_l * self.pay_w
        elif self.payload_shape == 'sphere':
            self.A_pay_z = np.pi * self.R_disk**2
        else:  # cylinder (default)
            self.A_pay_z = np.pi * self.R_disk**2
        self.Cd_pay = 0.8       # Drag coefficient (0.8 is typical for a box)

        # --- Inertia Tensor (J) ---
        # Impact: Determines how hard it is to rotate the payload.
        # - Higher Inertia: Payload is sluggish/stable against rotation.
        # - Lower Inertia: Payload spins easily.

        if self.payload_shape == 'sphere':
            # Sfera solida: I = (2/5) * m * r^2
            r = self.R_disk
            I_solid = (2.0/5.0) * self.m_payload * (r**2)
            J_solid = np.diag([I_solid, I_solid, I_solid])
            # Liquido approssimato come sfera: stessa formula
            I_liquid = (2.0/5.0) * self.m_liquid * (r**2)
            J_liquid_static = np.diag([I_liquid, I_liquid, I_liquid])

        elif self.payload_shape == 'cylinder':
            # Cilindro solido: Ixx=Iyy=(1/12)m(3R^2+h^2), Izz=(1/2)mR^2
            Ixx = (1/12) * self.m_payload * (3*self.R_disk**2 + self.pay_h**2)
            Iyy = Ixx
            Izz = 0.5  * self.m_payload * self.R_disk**2
            J_solid = np.diag([Ixx, Iyy, Izz])
            # Liquido approssimato come cilindro pieno
            I_lxx = (1/12) * self.m_liquid * (3*self.R_disk**2 + self.pay_h**2)
            J_liquid_static = np.diag([I_lxx, I_lxx, 0.5*self.m_liquid*self.R_disk**2])

        else:
            # Box / parallelepipedo: Ixx=(1/12)m(w^2+h^2), ecc.
            Ixx = (1/12) * self.m_payload * (self.pay_w**2 + self.pay_h**2)
            Iyy = (1/12) * self.m_payload * (self.pay_l**2 + self.pay_h**2)
            Izz = (1/12) * self.m_payload * (self.pay_l**2 + self.pay_w**2)
            J_solid = np.diag([Ixx, Iyy, Izz])
            # Liquido approssimato come parallelepipedo pieno
            I_lxx = (1/12) * self.m_liquid * (self.pay_w**2 + self.pay_h**2)
            I_lyy = (1/12) * self.m_liquid * (self.pay_l**2 + self.pay_h**2)
            I_lzz = (1/12) * self.m_liquid * (self.pay_l**2 + self.pay_w**2)
            J_liquid_static = np.diag([I_lxx, I_lyy, I_lzz])

        self.J = J_solid + J_liquid_static
            
        self.invJ = np.linalg.inv(self.J)       # Inertia Matrix

        # --- Center of Mass (CoM) & Pressure (CoP) ---
        self.CoP = np.array([0.0, 0.0, 0.0])    # Where wind pushes relative to center
        self.CoM_offset = np.array([0.0, 0.0, 0.0]) # Geometric offset of CoM

        # --- SLOSHING (Liquid Dynamics) ---
        # Impact: Simulates liquid moving inside the tank.
        # - True: Adds a disturbance force/torque based on sine waves.
        # - False: Acts as a solid block.
        self.enable_sloshing = True
        # Impact: How far the liquid moves side-to-side.
        # - Higher: Stronger disturbance, harder to stabilize.
        self.slosh_amp = 0.2    # Amplitude [m]
        
        # Impact: How fast the liquid sloshes.
        # - Should match resonance frequency for worst-case testing.
        self.slosh_freq = 0.15  # Frequency [Hz]
        self.tau_slosh = 0.9    # Smoothing time constant for the Slosh-Filter [s]
        self.max_com_shift = 1.0 # Safety limit for CoM compensation [m]


        # ==============================================================================
        # 6. CABLE PHYSICS & DYNAMICS
        # ==============================================================================
        # Impact: How "stretchy" the cable is.
        # - Higher (e.g., 2000): Steel cable. Rigid, transfers forces instantly, can cause numerical jitter.
        # - Lower (e.g., 100): Elastic band. Bouncy, introduces delay/lag.
        self.k_cable = 500.0     # Stiffness constant [N/m]
        self.L_ref_stiffness = 3.0 # Reference length for stiffness calculation
        
        # Impact: How much the cable absorbs energy.
        # - Higher: Cable stops oscillating quickly.
        # - Lower: Cable vibrates for a long time.
        effective_mass = (self.m_payload + self.m_liquid) / self.N
        self.gamma_damp = 3.0 * np.sqrt(self.k_cable * effective_mass) 
        
        # Impact: Simulates delay in the drone motors.
        # - Higher (e.g., 0.3): Motors are slow to spin up. Harder to control.
        # - Lower (e.g., 0.05): Motors are instant.
        self.alpha_motor_lag = 0.1 #s


        # ==============================================================================
        # 7. CONTROL SYSTEM (PID & Damping)
        # ==============================================================================
        # --- UAV Position PID ---
        # Controls how the UAVs chase their target positions.
        self.kp_pos = 25.0      # Proportional: Main drive. High = Fast but overshoot.
        self.kv_pos = 20.0      # Velocity (D): Damping. High = Slow/Sluggish. Low = Oscillating.
        self.ki_pos = 1.5      # Integral: Fixes steady-state error (wind).

        # --- Payload Position Correction ---
        # Additional logic to pull the payload to the correct spot.
        self.kp_pay_corr = 10.0  # Formation reacts to pull the payload towards the target.
        self.ki_pay_corr = 0.5  # Eliminates steady-state drift (e.g., pushes harder against constant wind)
        self.kd_pay_corr = 0.5
        self.pay_int_lim = 5.0  # Anti-windup limit

        # --- Attitude PID (Rotation) ---
        # Controls how the formation tilts the payload.
        self.kp_roll = 22.0     # High gain for Roll (Critical for stability) 8
        self.kp_pitch = 22.0    # Pitch gain 8
        self.kp_yaw = 22.0      # Yaw gain 3
        self.kv_rot = 22.0      # Damping for rotation 7
        self.ki_rot = 1.5      # Integral for rotation 1.5
        self.rot_int_lim = 20.0  # Limit for rotational integral 10

        # --- Active Damping ---
        # Impact: Artificial "friction" added by the controller to stop swinging.
        # - Higher: Very stable, but might feel "stuck" or slow to move.
        self.k_active_damp = 20.0   
        self.k_active_damp_z = 0.5
        self.pay_damping_lin = 2.5  # Linear drag simulation
        self.pay_damping_ang = 2.5  # Angular drag simulation
        
        # --- Smoothing ---
        self.alpha_acc_filter = 0.02       # Low-pass filter for acceleration commands
        self.formation_responsiveness = 0.04 # How fast formation shape updates ##################0.05
        self.beta_smooth = 20.0            # Smoothing factor for cable contact model
        self.alpha_moment_filter = 0.2     # Smoothing for momentum shift calculation

        # --- Winch / Cable Length Control ---
        # Impact: Parameters for the active winch damping (Phase 5).
        self.winch_damping_ratio = 0.7  # Zeta (0.7 = Optimal Critical Damping)
        self.winch_tau_relax = 0.3       # Time constant to relax tension
        self.winch_limit_correction = 0.2 # Max length change allowed [m]


        # ==============================================================================
        # 8. OPTIMIZATION & COST FUNCTION (The "Brain")
        # ==============================================================================
        # The optimizer minimizes: J_tot = w_T*J_T + w_ref*J_ref + w_smooth*J_smooth ...
        
        # Impact: How much we care about Equal Tensions.
        # - Higher: Drones try to share load perfectly (Good for safety).
        self.w_T = 2.00       
        
        # Impact: How much we care about keeping the cone angle near 'theta_ref'.
        # - Higher: Formation stays rigid at 30 degrees.
        # - Lower: Formation expands/contracts freely to handle wind.
        self.w_ref = 100 #50.0       
        
        # Impact: Penalizes rapid changes in formation shape.
        # - Higher: Slow, smooth transitions.
        # - Lower: Fast, jittery transitions.
        self.w_smooth = 10   #50
        
        # Impact: Safety Barrier. Penalizes tensions getting close to 0 (slack) or Max.
        self.w_barrier = 0.3  
        
        # Impact: Geometry condition number. Ensures the formation doesn't become "flat" or singular.
        self.w_cond = 1.0     #10
        
        # Impact: Penalizes errors in Force/Moment generation.
        # - Must be very high to ensure the physics requirements are met.
        self.w_resid_f = 10000.0  
        self.w_resid_m = 1000.0   

        # --- Safety Margins ---
        self.k_limit = 0.9  # Use only 70% of max thrust (Safety buffer)
        self.k_safe = 0.1   # Minimum tension as % of payload weight
        self.k_safe2 = 0.5  # Dynamic minimum tension buffer

        # --- Adaptation Weights ---
        # These control how the formation deforms into an ellipse against wind.
        self.lambda_static = 1.0 # Static CoM compensation gain
        self.lambda_perp = 2.0   # Stretch perpendicular to wind (Aerodynamic efficiency)
        self.lambda_par = 0.5   # Compress parallel to wind
        self.F_ref = 15.0  #8      # Reference force for scaling deformation
        
        # Blending factors (0.0 to 1.0)
        self.k_tilt = 1.0        # 1.0 = Payload Tilt = Formation Tilt
        self.lambda_shape = 1.0  # 1.0 = Full elliptical deformation enabled
        self.lambda_CoM = 1.0    # 1.0 = Full Center of Mass compensation
        self.lambda_aero = 0.0   # 1.0 = Tilt based on aerodynamics, 0.0 = Tilt based on geometry
        self.lambda_twist = 4.2
        self.lambda_traj = 0.0
        
        # --- Geometry Constraints ---
        self.theta_ref = 40.0    # Nominal cone angle [degrees]
        self.min_safe_angle = 10 # Minimum allowed angle [degrees]
        self.max_angle_variation = 0.1 # Max change per step [degrees]  1
        self.min_drone_z_rel = 1.5 # Drones cannot go below 1.5m above payload
        self.weight_tilt_scaling = 0.0 # Scales weights based on tilt (Advanced)

        # --- Soft Weather Vaning Configuration ---
        # Definisce quando ignorare la traiettoria per favorire l'aerodinamica.
        self.yaw_force_min = 5.0  # [N] Sotto questa forza, il sistema segue fedelmente la traiettoria (ref_yaw).
        self.yaw_force_max = 10.0  # [N] Sopra questa forza, il sistema si allinea totalmente alla forza (weather vane).
        self.alpha_force_filter = 0.2 # Costante del filtro passa-basso sulla forza (0.05 ~= 20 step di memoria).


        # ==============================================================================
        # 9. ADAPTIVE BEHAVIOR & MISSION PHASES
        # ==============================================================================
        # --- Waypoints ---
        self.home_pos = np.array([0.0, 0.0, 0.0])
        self.payload_goal_pos = np.array([20.0, 30.0, 10.0]) # Target for Phase 4
        self.safe_altitude_offset = 0.0 # Automatically calculated based on L

        # --- Phase Timing & Logic ---
        
        # Phase 0: Idle / Warmup
        self.t_idle = 0.0          # Time to wait before spinning motors [s]
        self.t_pretension = 5.0    # Time to gently pull cables tight [s]

        # Phase 1: Lift-Off
        self.t_lift = 10.0          # Duration of the lifting maneuver [s]

        # Phase 2: Alignment
        self.t_alignment = 3.0     # Time to stabilize after lift-off [s]

        # Phase 3: Attitude Settling
        self.t_attitude_1 = 6.0    # Time to test rotation before moving [s]
        self.traj_target_roll = np.radians(0)
        self.traj_target_pitch = np.radians(0)
        self.traj_target_yaw = np.radians(0)

        # Phase 4: Transport (Navigation)
        # Impact: Speed of the trajectory generator.
        self.nav_avg_vel = 1.8     # Average velocity [m/s]
        self.pendulum_correction = 1.0 

        # Phase 5: Winch / Drop
        self.t_attitude_2 = 12.0    # Duration of the drop/winch phase [s]
        self.final_target_roll = np.radians(0)
        self.final_target_pitch = np.radians(0) 
        self.final_target_yaw = np.radians(0)

        # --- Stabilization Checks ---
        # The mission waits for velocity to drop below these limits before advancing phase.
        self.stabilization_wait_time = 1.0
        self.stab_vel_limit = 1.5       # Max linear velocity [m/s]
        self.stab_omega_limit = 1.5     # Max angular velocity [rad/s]

        

