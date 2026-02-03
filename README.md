# Cooperative-Framework-for-Aerial-Manipulation
Development of a control system for a swarm of drones connected to a load via motorised cables (winches). The system operates as a parallel floating-base kinematic structure, capable of controlling the position and orientation of the load (6 DoF) independently of the drones' attitude.
The main contribution is a real-time optimization algorithm that continuously adapts the geometry of the formation in response to external disturbances and inertial forces.
By minimising a cost function based on cable tension, the algorithm calculates the optimal cone opening angle and the required cable length, ensuring stability and preventing slack even in windy conditions.
