from euler import *
from physics import *
from simulate import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

particle_1 = {
    "r": 12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}
particle_2 = {
    "r": 3*12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}
def analytical_free_diffusion(N: int, t: float, particle: dict) -> tuple(np.ndarray, np.ndarray):
    """
    Returns the analytical probability density associated with positions, x, for free diffusion.
    """
    # Sets up range and converts to micrometers
    x = np.linspace(-60,60,1000)
    x = x*10**-6

    # Makes an array time values
    t = np.ones_like(x)*t

    # Solve and convert
    n = diffusion_solution(x, t, N, particle)
    p = n*10**-6/N # /N means that one looks at the probability density
    x = x*10**6 # Is done to plot the result in micrometers
    return p, x

def simulate_free_diffusion(N: int, t: float, particle: dict) -> np.ndarray:
    """
    Finds the position of simulated particles when there is no potential
    """

    # tol is not an important parameter in this application. High tol means faster code
    positions = simulate_particle_position_no_potential(particle,N,t,ALPHA=0.2,TAU=None, tol=20)

    # Finds the final position and converts to micrometers
    last_pos = positions[-1]*particle["L"]*10**6
    return last_pos

def test_time_evolution():
    """
    This plots the probability density associated with a position after a time t with no potential and
    compares it to the theoretical values. Also, the time evolution of the distribution is plotted.
    """

    # Sets initial parameters
    set_plot_parameters(18)
    N = 50000
    t = 4
    fig, axs = plt.subplots(1,3, figsize=(12,4))

    # First plot particle 1
    analytic_density, positions = analytical_free_diffusion(N, t, particle_1)
    simulated_density = simulate_free_diffusion(N, t, particle_1)
    axs[0].plot(positions, analytic_density, color="red",zorder=3, label="p(x,4)")
    dens_1 = sns.distplot(simulated_density, ax = axs[0], color="green", label=r"$\langle$p$\rangle$ (x,4)")
    dens_1.set(ylabel="p(x,t)")
    dens_1.set(xlabel=r"Position [$\mu m$]")
    axs[0].legend(loc = "upper right", prop={"size": 11})
    axs[0].set_title(r"$r=r_1=12nm$")

    # Then plot particle 2
    analytic_density, positions = analytical_free_diffusion(N, t, particle_2)
    simulated_density = simulate_free_diffusion(N, t, particle_2)
    axs[1].plot(positions, analytic_density, color="blue",zorder=3,label="p(x,4)")
    dens_2 = sns.distplot(simulated_density, ax = axs[1], color="green", label=r"$\langle$p$\rangle$ (x,4)")
    dens_2.set(ylabel=None)
    dens_2.set(xlabel=r"Position [$\mu m$]")
    axs[1].legend(loc = "upper right", prop={"size": 11})
    axs[1].set_title(r"$r=r_2=36nm$")

    # Lastly, plot analytic time evolution
    for time in range(1, t + 1):
        analytic_density_1, positions_1 = analytical_free_diffusion(N, time, particle_1)
        analytic_density_2, positions_2 = analytical_free_diffusion(N, time, particle_2)
        axs[2].plot(positions_1, analytic_density_1, color="red", zorder=3)
        axs[2].plot(positions_2, analytic_density_2, color="blue")
    axs[2].set_xlabel(r"Position [$\mu m$]")
    axs[2].legend([r"$r_1=12nm$", r"$r_2=36nm$"], prop={"size": 11})
    axs[2].set_title(r"$1-4$ $seconds$")
    fig.tight_layout()
    fig.savefig("../plots/task_12.png", dpi=300)

if __name__ == "__main__":
    test_time_evolution()