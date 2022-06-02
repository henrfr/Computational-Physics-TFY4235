from physics import *
from euler import get_time_step, euler_numba
from scipy.integrate import simpson
from numba import njit

"""
This file contains the code used in the simulations in the tasks. All the functions are quite similar, but differ 
with regards to whether the potential is on/off, flashing is in/off, if it is compatible with numba and if
it is modified to use less memory.
"""

def simulate_particle_position_no_potential(particle: dict, N_particles: int, sim_time: float, ALPHA: float, TAU: float, tol: float) -> np.ndarray:
    """ 
    Simulates the particle diffusion when there is no potential. The positions should take the form of a gaussian.

    Parameters
    ----------------------------------------------
    particle : dict
        Contains the particle information.
    N_particles : int
        The number of particles in the simulation.
    sim_time : float
        How long the simulation will run in real units.
    ALPHA : float
        The asymmetry factor.
    TAU : float
        The flashing period in real units
    tol : float
        Modifies the size of the time step. Should in general be <0.1, but can be higher for no potential.

    Returns
    ----------------------------------------------
    np.ndarray : (N_steps, N_particles) - contains information about every step for every particle
    """
    D, gamma, omega = reduced_units(**particle) # Converts to reduced units
    dt = get_time_step(D, ALPHA, tol) # Finds time step.
    N_steps = int((sim_time*omega)/dt) # Converts the simulation time to reduced time and finds number of steps
    positions = np.zeros((N_steps, N_particles)) 
    for i in range(N_steps): # Solution loop
        next_step_no_potential(positions,i,dt,D) # Makes one step in the forward Euler method.
    return positions   

@njit(cache=True)
def next_step_no_potential(positions: np.ndarray, i: int, dt: float, D: float) -> None:
    """ 
    A helper function performing one step of the forward Euler method. It is wrapped to be
    compatible with using numba.

    Parameters
    ----------------------------------------------
    positions : np.ndarray
        The positions of the particles. Shape = (N_steps, N_particles).
    i : int
        The current time step.
    dt : float
        The length of the current time step.
    D : float
        The diffusion constant in reduced units.

    Returns
    ----------------------------------------------
    None : The function modifies positions in place.
    """
    f = np.zeros_like(positions[0]) # No potential means that the force is always 0.
    positions[i] = euler_numba(positions[i-1], dt, f, D)  # One step of the forward Euler

def simulate_particle_position_no_flashing(particle: dict, N_particles: int, sim_time: float, ALPHA: float, TAU: float, tol: float) -> np.ndarray:
    """ 
    Simulates the particle diffusion when the potential is on, but the flashing is off.
    Depending on the particle parameters, the particles might not diffuse at all or 
    they might take a random walk.

    Parameters
    ----------------------------------------------
    particle : dict
        Contains the particle information.
    N_particles : int
        The number of particles in the simulation.
    sim_time : float
        How long the simulation will run in real units.
    ALPHA : float
        The asymmetry factor.
    TAU : float
        The flashing period in real units
    tol : float
        Modifies the size of the time step. Should in general be <0.1.

    Returns
    ----------------------------------------------
    np.ndarray : (N_steps, N_particles) - contains information about every step for every particle
    """
    # See simulate_particle_positions_no_potential() for comments. It is almost the same.
    D, gamma, omega = reduced_units(**particle)
    dt = get_time_step(D, ALPHA, tol)
    N_steps = int((sim_time*omega)/dt)
    positions = np.zeros((N_steps, N_particles))
    for i in range(N_steps):
        next_step_no_flashing(positions,i,dt,ALPHA,TAU,omega,D)
    return positions

@njit(cache=True)
def next_step_no_flashing(positions: np.ndarray, i: int, dt: float, ALPHA: float, TAU: float, omega: float, D: float) -> None:
    """ 
    A helper function performing one step of the forward Euler method. It is wrapped to be
    compatible with using numba. The potential is on, but there is no flashing.

    Parameters
    ----------------------------------------------
    positions : np.ndarray
        The positions of the particles. Shape = (N_steps, N_particles).
    i : int
        The current time step.
    dt : float
        The length of the current time step.
    ALPHA : float
        The asymmetry factor
    TAU : float
        The flashing period in real units. Note: Not used when flashing=False
    omega : float
        Converts TAU to a reduced period. Note: Not used when flashing=False
    D : float
        The diffusion constant in reduced units.

    Returns
    ----------------------------------------------
    None : The function modifies positions in place.
    """
    f = force(positions[i-1], i*dt, ALPHA, TAU*omega, flashing=False) # Finds the foce at every particle position
    positions[i] = euler_numba(positions[i-1], dt, f, D)  

def simulate_particle_position_flashing_potential(particle: dict, N_particles: int, sim_time: float, ALPHA: float, TAU: float, tol: float) -> np.ndarray:
    """ 
    Simulates the particle diffusion when a flashing potential is applied. TAU will determine whether
    the particles will not diffuse, will travel stepwise to the right or do a random walk.
    For the code in the tasks, TAU ~ 1 will make it move stepwise to the right.

    Parameters
    ----------------------------------------------
    particle : dict
        Contains the particle information.
    N_particles : int
        The number of particles in the simulation.
    sim_time : float
        How long the simulation will run in real units.
    ALPHA : float
        The asymmetry factor.
    TAU : float
        The flashing period in real units.
    tol : float
        Modifies the size of the time step. Should in general be <0.1.

    Returns
    ----------------------------------------------
    np.ndarray : (N_steps, N_particles) - contains information about every step for every particle
    """
    # Almost the same as no_potential and no_flashing
    D, gamma, omega = reduced_units(**particle)
    dt = get_time_step(D, ALPHA, tol)
    N_steps = int((sim_time*omega)/dt)
    positions = np.zeros((N_steps, N_particles))
    for i in range(N_steps):
        next_step_flashing_potential(positions, i,dt, ALPHA, TAU,omega,D)
    return positions

@njit(cache=True)
def next_step_flashing_potential(positions: np.ndarray, i: int, dt: float, ALPHA: float, TAU: float, omega: float, D: float) -> None:
    """ 
    A helper function performing one step of the forward Euler method. It is wrapped to be
    compatible with numba. The flashing potential is on.

    Parameters
    ----------------------------------------------
    positions : np.ndarray
        The positions of the particles. Shape = (N_steps, N_particles).
    i : int
        The current time step.
    dt : float
        The length of the current time step.
    ALPHA : float
        The asymmetry factor
    TAU : float
        The flashing period in real units.
    omega : float
        Converts TAU to a reduced period.
    D : float
        The diffusion constant in reduced units.

    Returns
    ----------------------------------------------
    None : The function modifies positions in place.
    """
    f = force(positions[i-1], i*dt, ALPHA, TAU*omega, flashing=True) # Finds the force at the previous location at time=i*dt
    positions[i] = euler_numba(positions[i-1], dt, f, D)  

def simulate_particle_position_flashing_potential_less_memory(particle: dict, N_particles: int, sim_time: float, ALPHA: float, TAU: float, tol: float) -> np.ndarray:
    """ 
    Simulates the particle diffusion when a flashing potential is applied. TAU will determine whether
    the particles will not diffuse, will travel stepwise to the right or do a random walk.
    For the code in the tasks, TAU ~ 1 will make it move stepwise to the right.
    Note: This function only stores the current positions. This makes it memory efficient, but
    unsuitable for plotting trajectories.

    Parameters
    ----------------------------------------------
    particle : dict
        Contains the particle information.
    N_particles : int
        The number of particles in the simulation.
    sim_time : float
        How long the simulation will run in real units.
    ALPHA : float
        The asymmetry factor.
    TAU : float
        The flashing period in real units.
    tol : float
        Modifies the size of the time step. Should in general be <0.1.

    Returns
    ----------------------------------------------
    np.ndarray : (2, N_particles) - contains information about every step for every particle
    """    
    # This is still almost the same as the no_potential and flashing_potential, but memory efficient
    D, gamma, omega = reduced_units(**particle)
    print(omega)
    dt = get_time_step(D, ALPHA, tol)
    print(dt)
    N_steps = int((sim_time*omega)/dt)
    print(N_steps)
    print(sim_time)
    positions = np.zeros((2, N_particles), dtype=np.float32) # Allocates two rows insted of N_steps rows.
    for i in range(N_steps):
        next_step_flashing_potential_less_memory(positions, i,dt, ALPHA, TAU,omega,D)
    return positions[-1]
@njit(cache=True)
def next_step_flashing_potential_less_memory(positions: np.ndarray, i: int, dt: float, ALPHA: float, TAU: float, omega: float, D: float) -> None:
    """ 
    A helper function performing one step of the forward Euler method. It is wrapped to be
    compatible with numba. The flashing potential is on. It uses the current positions to find the
    next positions, before writing over the old positions to save memory.

    Parameters
    ----------------------------------------------
    positions : np.ndarray
        The positions of the particles. Shape = (2, N_particles).
    i : int
        The current time step.
    dt : float
        The length of the current time step.
    ALPHA : float
        The asymmetry factor
    TAU : float
        The flashing period in real units.
    omega : float
        Converts TAU to a reduced period.
    D : float
        The diffusion constant in reduced units.

    Returns
    ----------------------------------------------
    None : The function modifies positions in place.
    """    
    f = force(positions[0], i*dt, ALPHA, TAU*omega, flashing=True)
    positions[1] = euler_numba(positions[0], dt, f, D)
    positions[0] = positions[1]  


def normalize(distribution: np.ndarray) -> np.ndarray:
    """ 
    A helper function for normalizing a distribution

    Parameters
    ----------------------------------------------
    distribution : np.ndarray
        The positions of the particles. Shape = (N_particles,).

    Returns
    ----------------------------------------------
    np.ndarray : The normalized distribution. Shape = (N_particles,)
    """
    N = simpson(distribution, dx=1 / len(distribution)) # Integrates using Simpson
    return distribution/N # Makes the integral of the function equal 1