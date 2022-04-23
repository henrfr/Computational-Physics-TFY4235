import numpy as np
from numba import njit


@njit(cache=True)
def analytical_decay(LAMBDA: float, N0: int, t: np.ndarray) -> np.ndarray:
    """
    Returns the analytical solution for the nuclear decay rate.

    Parameters
    ----------------------------------------------
    LAMBDA : float
        The decay constant for a specific particle.
    N0 : int
        The original number of particles
    t : np.ndarray
        The relevant times.
    
    Returns
    ----------------------------------------------
    np.ndarray : The number of particles of type N0 at each specified time.
    """
    return N0*np.exp(-LAMBDA*t)
@njit(cache=True)
def analytical_decay_3_step(LAMBDA1: float, LAMBDA2: float, N0: int, t: np.ndarray) -> np.ndarray:
    """
    Returns the analytical solution for the nuclear decay rate for a chain of
    3 isotopes. It is a solution of the Bateman equation
    (https://en.wikipedia.org/wiki/Bateman_equation).

    Parameters
    ----------------------------------------------
    LAMBDA1 : float
        The decay constant for the parent particle.
    LAMBDA2 : float
        The decay constant for the child particle
    N0 : int
        The original number of particles
    t : np.ndarray
        The relevant times.
    
    Returns
    ----------------------------------------------
    np.ndarray : The number of particles of type N0 and its child at each specified time.
    """
    parent = N0*np.exp(-LAMBDA1*t)
    child = (LAMBDA1/(LAMBDA2 - LAMBDA1))*(parent-N0*np.exp(-LAMBDA2*t))
    return parent, child

def secular_equilibrium(LAMBDA1: float, LAMBDA2: float, N0: int) -> float:
    """
    Returns the value for the secular equilibrium, that is, the point where the amounts of
    child particle is created from the parent particle equals the decay rate of the
    child particle. The value corresponds to how many child particles there are at secular
    equilibrium.

    Parameters
    ----------------------------------------------
    LAMBDA1 : float
        The decay constant associated with the parent particle.
    LAMBDA2 : float
        The decay constant associated with the child particle.
    N0 : int
        The starting number of parent particles.
    
    Returns
    ----------------------------------------------
    float : A number representing the secular equilibrium for the two particles. 
    """
    return (LAMBDA1/LAMBDA2)*N0

@njit(cache=True)
def p_no_decay(LAMBDA: float, delta_t: float) -> float:
    """
    Returns the probability of a particle not decaying during the time delta_t.

    Parameters
    ----------------------------------------------
    LAMBDA : float
        The decay constant associated with the particle.
    delta_t : float
        A time interval
    
    Returns
    ----------------------------------------------
    float : The probability of not decaying
    """    
    return np.exp(-LAMBDA*delta_t)

@njit(cache=True)
def simulate_decay(N0: int, t_max: float, delta_t: float, LAMBDA: float) -> np.ndarray:
    """
    Simulates the decay rate of a specific particle.

    Parameters
    ----------------------------------------------
    N0 : int
        The initial number of particles
    t_max : float
        The maximum simulation time
    delta_t : float
        A time interval
    LAMBDA : float
        The decay constant accosiated with the particle
    
    Returns
    ----------------------------------------------
    np.ndarray : The number of particles left at each point in time
    """ 
    N = int(t_max/delta_t) # Finds the number of steps
    N0_history = np.empty(N)
    N0_history[0] = N0 # Initializes the first time step.
    prob_no_decay = p_no_decay(LAMBDA, delta_t) # Finds the probability of not decaying during delta_t
    n_parent = N0
    for time_step in range(N):
        # Generates as many random numbers as there are parent atoms
        rnd_numbers = np.random.rand(n_parent)
        for particle in range(n_parent):
            # Reduces the number of parent atoms if the random number is greater than the 
            # probability of not decaying.
            if rnd_numbers[particle] > prob_no_decay:
                n_parent -= 1
        N0_history[time_step] = n_parent
    return N0_history

@njit(cache=True, parallel=True)
def simulate_decay_3_step(N0: int, t_max: float, delta_t: float, LAMBDA1: float, LAMBDA2: float) -> tuple:
    """
    Simulates the decay rate of a 3 step decay chain

    Parameters
    ----------------------------------------------
    N0 : int
        The initial number of particles
    t_max : float
        The maximum simulation time
    delta_t : float
        A time interval
    LAMBDA1 : float
        The decay constant accosiated with the parent particle
    LAMBDA2 : float
        The decay constant accosiated with the child particle
    
    Returns
    ----------------------------------------------
    tuple(ndarray,ndarray) : The number of particles left at each point in time
    """ 
    N = int(t_max/delta_t) # Finds number of time steps
    parent_history = np.empty(N)
    parent_history[0] = N0
    child_history = np.zeros(N)

    # Finds the probability constants
    prob_no_decay_parent = p_no_decay(LAMBDA1, delta_t)
    prob_no_decay_child = p_no_decay(LAMBDA2, delta_t)

    # Initializes number of particles
    n_parent = N0
    n_child = 0

    for i in range(N):
        # Generates as many random numbers as there are parent particles
        rnd_numbers = np.random.rand(n_parent)
        for j in range(n_parent):
            # Reduces number of parent and increase number of child
            # particles if decay happens.
            if rnd_numbers[j] > prob_no_decay_parent:
                n_parent -= 1
                n_child += 1
        # Generates as many random numbers as there are child particles
        rnd_numbers = np.random.rand(n_child)
        for k in range(n_child):
            if rnd_numbers[k] > prob_no_decay_child:
                n_child -= 1
        parent_history[i] = n_parent
        child_history[i] = n_child
    return parent_history, child_history