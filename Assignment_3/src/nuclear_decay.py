import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
from scipy.signal import savgol_filter
from plot_params import set_plot_parameters

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

def test_analytical_decay():
    """
    A short test showing the usage of the analytical decay function.
    """
    # Initializing parameters
    LAMBDA = 0.3
    N_0 = 250
    t = np.linspace(0,100,10000)

    # Plotting
    plt.plot(t, analytical_decay(LAMBDA, N_0, t))
    plt.show()

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
    number_of_particle0 = N0
    for time_step in range(N):
        # Generates as many random numbers as there are particles
        rnd_numbers = np.random.rand(number_of_particle0)
        for particle in range(number_of_particle0):
            # Reduces the number of particles if the random number is greater than the 
            # probability of not decaying.
            if rnd_numbers[particle] > prob_no_decay:
                number_of_particle0 -= 1
        N0_history[time_step] = number_of_particle0
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
    N0_history = np.empty(N)
    N0_history[0] = N0
    N1_history = np.zeros(N)

    # Finds the probability constants
    prob_no_decay = p_no_decay(LAMBDA1, delta_t)
    prob_no_decay1 = p_no_decay(LAMBDA2, delta_t)

    # Initializes number of particles
    number_of_particle0 = N0
    number_of_particle1 = 0

    for i in range(N):
        # Generates as many random numbers as there are parent particles
        rnd_numbers = np.random.rand(number_of_particle0)
        for j in range(number_of_particle0):
            # Reduces number of parent and increase number of child
            # particles if decay happens.
            if rnd_numbers[j] > prob_no_decay:
                number_of_particle0 -= 1
                number_of_particle1 += 1
        # Generates as many random numbers as there are child particles
        rnd_numbers = np.random.rand(number_of_particle1)
        for k in range(number_of_particle1):
            if rnd_numbers[k] > prob_no_decay1:
                number_of_particle1 -= 1
        N0_history[i] = number_of_particle0
        N1_history[i] = number_of_particle1
    return N0_history, N1_history

def test_decay():
    """
    Simulates and compares a 3-step nuclear decay chain with the analytical solution.
    """
    LAMBDA = 0.03
    N0 = 30000
    t_max = 100
    delta_t = 0.01
    LAMBDA2 = 20*LAMBDA # 5 can be good

    N1, N2 = simulate_decay_3_step(N0, t_max, delta_t, LAMBDA, LAMBDA2)
    t = np.linspace(0,t_max,len(N1))
    t_N1 = np.linspace(0, t_max,len(N1))
    parent, child = analytical_decay_3_step(LAMBDA, LAMBDA2, N0, t)
    plt.plot(t, parent, label="Analytical (Parent)")
    plt.plot(t, child, label="Analytical (Child)")
    plt.plot(t, [secular_equilibrium(LAMBDA, LAMBDA2, N0) for _ in t], label="Secular equilibrium", linestyle="dotted")
    plt.plot(t_N1, N1, label="Simulated parent")
    plt.plot(t_N1, N2, label="Simulated child")
    plt.yscale("log")
    plt.legend()
    plt.savefig("../plots/decay.png")
    plt.show()

@njit(cache=True, parallel=True)
def aggregate_statistics():
    """
    Simulates N distinct deacys of N0 particles and returns the error
    """
    N = 2000
    LAMBDA = 0.03
    N0 = 1000
    t_max = 150
    delta_t = 0.005
    N1 = np.empty((N, int(t_max/delta_t)))
    error_N1 = np.empty((N, int(t_max/delta_t)))

    for i in prange(N):
        N_1 = simulate_decay(N0, t_max, delta_t, LAMBDA)
        N1[i] = N_1
    t = np.linspace(0,t_max,len(N1[0]))
    parent = analytical_decay(LAMBDA, N0, t)
    for row in prange(len(N1)):
        error_N1[row] = N1[row] - parent
    return N1, error_N1, parent

def display_statistics():
    start = time.time()
    N1, error_N1, parent = aggregate_statistics()

    # Get total mean error and mean relative error
    error_N1 = np.mean(error_N1, axis=0)
    error_N1_relative = error_N1/parent

    print(f"Time spent: {time.time()-start} seconds.")
    t_max = 150
    t = np.linspace(0,t_max,len(N1[0]))

    # Plotting
    fig, axs = plt.subplots(1,3,figsize=(14,5))
    axs[0].plot(t, error_N1, label="Absolute error")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Error in number of particles")
    axs[1].plot(t, error_N1_relative, label="Relative error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Relative error (%)")
    axs[1].legend()
    axs[2].plot(t, parent, label="Analytical solution")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Number of particles")
    axs[0].legend()
    axs[2].legend()
    plt.tight_layout()
    plt.savefig("../plots/stats.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    set_plot_parameters()
    display_statistics()
    test_decay()