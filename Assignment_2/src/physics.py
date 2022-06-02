import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit(cache=True)
def U_r(position: np.ndarray, ALPHA: float) -> np.ndarray:
    """ 
    Returns an array representing an asymetric sawtooth-potential.
    The function is implemented in reduced units.

    Parameters
    ----------------------------------------------
    position : np.ndarray
        An array containing positions in the x-direction. It is on the form [X_min, ..., X_max].
    ALPHA : float
        A parameter describing the assymetry of the potential.
    
    Returns
    ----------------------------------------------
    np.ndarray : A periodic assymetric sawtooth-potential.
    """
    position = position % 1 # Makes the position periodic
    return np.where(position<ALPHA, position/ALPHA, (1-position)/(1-ALPHA)) # Returns position/ALPHA if position<ALPHA else (1-position)/(1-ALPHA)

def U(position: np.ndarray, t: np.ndarray, ALPHA: float, TAU: float, flashing: bool=True) -> np.ndarray:
    """ 
    Combines U_r() and f() to form a potential that can include flashing.
    NB: This ended up not being used.

    Parameters
    ----------------------------------------------
    position : np.ndarray
        An array containing positions in the x-direction. It is on the form [X_min, ..., X_max].
    t : np.ndarray
        An array containing discrete time values. It is on the form [0, ..., t_max].
    ALPHA : int
        A parameter describing the assymetry of the potential.
    TAU : float
        The period of the flashing.
    flashing: bool
        If true, the potential is flashing, else it is constant.
    
    Returns
    ----------------------------------------------
    np.ndarray : A periodic assymetric sawtooth-potential, which can be flashing.
    """
    if not flashing:
        return U_r(position, ALPHA)
    return U_r(position, ALPHA)*f(t, TAU)
    
@njit(cache=True)
def force_r(position: np.ndarray, ALPHA: float) -> np.ndarray:
    """ 
    Returns an array representing the force in an asymetric sawtooth-potential.
    The function is implemented in reduced units and is the derivative of U_r().

    Parameters
    ----------------------------------------------
    position : np.ndarray
        An array containing positions in the x-direction. It is on the form [X_min, ..., X_max].
    ALPHA : float
        A parameter describing the assymetry of the potential.
    
    Returns
    ----------------------------------------------
    np.ndarray : Values for the force in the periodic assymetric sawtooth-potential.
    """
    a = position % 1
    return np.where(a<ALPHA, 1/ALPHA, 1/(ALPHA-1))

@njit(cache=True)
def f(t: float, TAU: float):
    """ 
    This function represent the flashing. It yields 0 when the potential is off, else 1.

    Parameters
    ----------------------------------------------
    t : float
        The current time
    TAU : float
        The period of the flashing. NB: Is passed as omega*TAU in later functions
    
    Returns
    ----------------------------------------------
    int : Describes whether the potential is on or off.
    """
    t = t % TAU # Makes the flashing periodic
    if t < 3*TAU/4: # The potential will be off 3/4 of the time.
        return 0
    return 1

@njit(cache=True)
def force(position: np.ndarray, t: np.ndarray, ALPHA: float, TAU: float, flashing: bool = True) -> np.ndarray:
    """ 
    Combines f_r() and f() to form a potentially flashing force.

    Parameters
    ----------------------------------------------
    position : np.ndarray
        An array containing positions in the x-direction. It is on the form [X_min, ..., X_max].
    t : np.ndarray
        An array containing discrete time values. It is on the form [0, ..., t_max].
    ALPHA : int
        A parameter describing the assymetry of the potential.
    TAU : float
        The period of the flashing.
    flashing: bool
        If true, the potential is flashing, else it is constant.
    
    Returns
    ----------------------------------------------
    np.ndarray : A periodic assymetric sawtooth-potential, which can be flashing.
    """
    if not flashing:
        return force_r(position, ALPHA)
    return force_r(position, ALPHA)*f(t, TAU)

def gaussian(N: int) -> np.ndarray:
    """
    Returns an array of normally distributed numbers with mean=0 and std=1 of size N.
    """
    return np.random.normal(0,1,N)

def boltzmann_dist(U: np.ndarray, du: float, kbT: float) -> np.ndarray:
    """ 
    Calculates the boltzmann distribution in reduced units.

    Parameters
    ----------------------------------------------
    U : np.ndarray
        U is the potential in reduced units. Shape = (N,)
    du : float
        The potential difference
    kbT : float
        The thermal energy
    
    Returns
    ----------------------------------------------
    np.ndarray : The probability according to the Boltzmann distribution of occupying a certain state
    """
    return np.exp(-U*du/kbT)/(kbT*(1-np.exp(-du/kbT)))

def reduced_units(**kwargs) -> list[float]:
    """ 
    Converts particle properties into reduced units.
    """
    r = kwargs.get("r")
    L = kwargs.get("L")
    eta = kwargs.get("eta")
    kbT = kwargs.get("kbT")
    delta_U = kwargs.get("delta_U")
    
    D = kbT/delta_U
    gamma = 6 * np.pi * eta * r
    omega = delta_U/(gamma*L**2)

    return D, gamma, omega

def diffusion_solution(x: np.ndarray, t: float, N: int, particle: dict) -> np.ndarray:
    """ 
    Calculates the solution to the diffusion equation for a given time t.

    Parameters
    ----------------------------------------------
    x : np.ndarray
        The initial position of the particles. In this project, it is taken to be [0, ... , 0]
    t : float
        Is the time which is solved for.
    N : int
        The number of particles
    particle : dict
        A dictionary containing all relevant parameters for the solution.
    
    Returns
    ----------------------------------------------
    np.ndarray : The number density of particles at the positions.
    """
    D_red, gamma, omega = reduced_units(**particle) # Converts to reduced units.
    D = particle["kbT"]/gamma # Finds the real diffusion constant
    return N*np.exp(-x**2/(4*D*t))/(np.sqrt(4*np.pi*D*t))

def set_plot_parameters(size: int=22) -> None:
    """
    A helper function for setting plot parameters, not really physics, but handy to have here.
    """
    plt.style.use('seaborn-bright')
    plt.rcParams['mathtext.fontset'] = 'cm'
    font = {'family' : 'serif', 
            'size': size}
    plt.rc('font', **font)
    plt.rc('lines', lw=2)


