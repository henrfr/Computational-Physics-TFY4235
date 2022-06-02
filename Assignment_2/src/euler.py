import numpy as np
from physics import gaussian
from numba import njit

def euler(x: np.ndarray, dt: float, F: np.ndarray, D: float) -> np.ndarray:
    """ 
    Performs one step of the forward Euler method in reduced units.

    Parameters
    ----------------------------------------------
    x : np.ndarray
        The positions of the particles. Shape: (N_particles,)
    dt : float
        The time step
    F : np.ndarray
        The force at each position. Same shape as x
    D : float
        The diffusion constant in reduced units
    
    Returns
    ----------------------------------------------
    np.ndarray : The new positions after one step of the method
    """
    return x - F*dt + np.sqrt(2*D*dt)*gaussian(x.shape[0])

@njit(cache=True)
def euler_numba(x: np.ndarray, dt: float, F: np.ndarray, D: float) -> np.ndarray:
    """ 
    Performs one step of the forward Euler method in reduced units. It is the same as euler(),
    but modified to be used with numba.

    Parameters
    ----------------------------------------------
    x : np.ndarray
        The positions of the particles. Shape: (N_particles,)
    dt : float
        The time step
    F : np.ndarray
        The force at each position. Same shape as x
    D : float
        The diffusion constant in reduced units
    
    Returns
    ----------------------------------------------
    np.ndarray : The new positions after one step of the method
    """
    return x - F*dt + np.sqrt(2*D*dt)*np.random.normal(0,1,x.shape[0])

def get_time_step(D: float, ALPHA: float, tol: float) -> float:
    """ 
    Finds a time step that fulfills an equation guaranteeing that the
    particles will not tunnel across the potential barrier due to
    the stochastic term.

    Parameters
    ----------------------------------------------
    D : float
        The diffusion constant in reduced units
    ALPHA : float
        The asymmetry parameter 
    tol : float
        Ensures that the time step is sufficiently small. Should be < 0.1, but can be modified depending on application.

    Returns
    ----------------------------------------------
    float : A timestep fulfilling the equation
    """
    max_force = np.amax([1/ALPHA, 1/(1-ALPHA)]) # Finds the max force
    dt = 10 # Initializes dt. The large number is chosen for flexibility
    while max_force*dt + 4* np.sqrt(2*D*dt) > tol*ALPHA: # Reduces the time step until it is small enough
        dt *= 0.9
    return dt