from euler import *
from physics import *
from simulate import *
import matplotlib.pyplot as plt
from seaborn import histplot


particle_1 = {
    "r": 12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}
# This particles has radius 3 times larger than the first. It is expected to be slower.
particle_2 = {
    "r": 3*12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}


def test_time_evolution():
    """ 
    This function plots the time evolution of the probability density for the two
    different particles. It will be slow if both N_particles and sim_time is high.
    At the moment, only final posiitons are plotted, but the entire time evolution
    could easily be obtained, but at the cost of memory.
    """
    # Initial parameters
    N_particles = 300
    sim_time = 100
    ALPHA = 0.2
    TAU = 0.55 # The previously found "optimal" period

    # Finds the final positions of the particles
    positions = simulate_particle_position_flashing_potential_less_memory(
        particle_1,N_particles, sim_time, ALPHA, TAU, tol=0.1)
    positions_2 = simulate_particle_position_flashing_potential_less_memory(
        particle_2,N_particles, sim_time, ALPHA, TAU, tol=0.1)

    # Converts to real units and plots the probability density
    plt.rcParams.update({'font.size': 16})
    histplot(positions*particle_1["L"]*10**6, label=r"$r_1$ = 12nm", kde=True, color="red", stat="density")
    histplot(positions_2*particle_2["L"]*10**6, label=r"$r_2$ = 36nm", kde=True, color="blue", stat="density")
    plt.xlabel(r"Position [$\mu m$]")
    plt.ylabel(r"Probability density")
    plt.title(f"Particle distribution after {sim_time} seconds")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("../plots/task_13_test5.png")
    plt.show()

if __name__ == "__main__":
    test_time_evolution()
