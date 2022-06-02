from physics import *
from simulate import *

"""
This file contains code for plotting used in task 7. The code is somewhat crowded and might
not be runnable if the necessary data is not available from the data folder. This could be
fixed if I have time later.
"""

# Defines particles
particle_0 = {
    "r": 12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}
particle_1 = {
    "r": 12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}

def test_different_delta_U():
    """
    PLots of average pos for different delta_U. This code investigates the average position of particles
    in asymmetric potentials of different magnitude. The positions should be slightly negative.
    """
    N_particles = 10000
    sim_time = 100
    set_plot_parameters(20)
    last_pos = []
    delta_U_factor = np.linspace(0.1,10.1,100)

    # Finds the last positions for the particles after the sim_time is passed for each delta_U
    for i in delta_U_factor:
        particle_1["delta_U"] = i*particle_0["kbT"]
        last_pos.append(simulate_particle_position_no_flashing(particle_1,N_particles,sim_time,0.2,1)[-1])

    # Calculate mean position for all particles and stores the data    
    means = np.array([np.mean(e) for e in last_pos])
    means = means*10**6*particle_0["L"] # Converts to real units (10**6 makes the position represented in micrometers)
    np.save("../data/means.npy", means)

    # Plotting
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(delta_U_factor, means)
    ax.set_xlabel(r"Multiples of $k_BT$")
    ax.set_ylabel(r"Average position [$\mu m$]")
    fig.tight_layout()
    fig.savefig("../plots/Avg_pos_diff_U.png", dpi=300)
    plt.show()

# Plots of trajectories
def test_plot_trajectories():
    """
    Plots trajectories for 2 particles in two potentials of different magnitude.
    The trajectories are plotted together with the mean positions from test_different_delta_U()
    """

    # Sets initial parameters
    set_plot_parameters(20)
    N_particles = 2
    sim_time = 100
    particle_0["delta_U"] = 0.1*particle_0["kbT"]
    particle_1["delta_U"] = 10*particle_1["kbT"]
    ALPHA = 0.2
    TAU = 1 # Not used but passed for conveniance

    # Finds the positions for all time steps for different potentials.
    positions = simulate_particle_position_no_flashing(particle_0,N_particles,sim_time,ALPHA,TAU)
    positions1 = simulate_particle_position_no_flashing(particle_1,N_particles,sim_time,ALPHA,TAU)

    # Reconstructs real time
    t = np.linspace(0, sim_time, len(positions[:,0]))
    t1 = np.linspace(0,sim_time, len(positions1[:,0]))

    # Plotting the results in real units.
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(t, positions[:,0]*1e6*particle_0["L"], color='blue', label=r"$\Delta U = 0.1 k_BT$")
    ax[0].plot(t1, positions1[:,0]*1e6*particle_1["L"], color='red',label=r"$\Delta U = 10 k_BT$")
    ax[0].plot(t, positions[:,1]*1e6*particle_0["L"], color='blue')
    ax[0].plot(t1, positions1[:,1]*1e6*particle_1["L"], color='red')
    ax[0].set_xlabel(r"Time [$s$]")
    ax[0].set_ylabel(r"Position [$\mu m$]")
    ax[0].legend()
    means = np.load("../data/means.npy", allow_pickle=True)
    delta_U_factor = np.linspace(0.1,10.1,100)
    ax[1].plot(delta_U_factor, means)
    ax[1].set_xlabel(r"Multiples of $k_BT$")
    ax[1].set_ylabel(r"Average position [$\mu m$]")
    fig.tight_layout()
    fig.savefig("../plots/trajectories_and_avg_pos.png", dpi=300)
    plt.show()

def test_plot_boltzmann():
    """
    Plots the theoretical boltzmann distribution along the distribution of potential obtained
    from simulations.
    """

    # Sets initial parameters
    N_particles = 1000000
    sim_time = 10
    N_particles_1 = 10000
    sim_time_1 = 10
    particle_0["delta_U"] = 0.1*particle_0["kbT"]
    particle_1["delta_U"] = 10*particle_1["kbT"]
    ALPHA = 0.2
    TAU = 1 # Not used, but passed for conveniance
    set_plot_parameters(20)

    # Finds the positions of the particles
    positions = simulate_particle_position_no_flashing(particle_0,N_particles,sim_time,ALPHA,TAU, tol=0.1)
    positions_1 = simulate_particle_position_no_flashing(particle_1, N_particles_1, sim_time_1, ALPHA, TAU, tol=0.1)

    # Finds the potetnial associated with the final position
    potential_dist_particle = U_r(positions[-1], particle_0["alpha"])
    potential_dist_particle_1 = U_r(positions_1[-1], particle_1["alpha"])

    # Plots the results

    # Makes a histogram, thus normalizing the distribution since density=True,
    # plots a curve and removes the bins for readability
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    potential_histogram = axs[0].hist(potential_dist_particle, density=True, bins=200)
    potential_histogram_1 = axs[1].hist(potential_dist_particle_1, density=True, bins=200)
    axs[0].clear()
    axs[1].clear()
    axs[0].plot(potential_histogram[1][1:], potential_histogram[0], label="Computed")
    axs[1].plot(potential_histogram_1[1][1:], potential_histogram_1[0], label="Computed")

    # Finds the theoretical boltzmann distribution, normalizes it and plots it
    potential_dist_linear = np.linspace(0,1,N_particles)
    potential_dist_linear_1 = np.linspace(0,1,N_particles_1)
    p = boltzmann_dist(potential_dist_linear, particle_0["delta_U"], particle_0["kbT"])
    p_1 = boltzmann_dist(potential_dist_linear_1, particle_1["delta_U"], particle_1["kbT"])
    p = normalize(p)
    p_1 = normalize(p_1)
    axs[0].plot(potential_dist_linear, p, label=r"Boltzmann", color="red")
    axs[1].plot(potential_dist_linear_1, p_1, label=r"Boltzmann", color="red")

    # Axes, labels, ...
    axs[0].set_title(r"$\Delta U = 0.1 k_BT$")
    axs[1].set_title(r"$\Delta U = 10 k_BT$")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlabel(r"$U$", fontsize=26)
    axs[0].set_ylabel(r"$p$", fontsize=26)
    axs[1].set_xlabel(r"$U$", fontsize=26)
    axs[1].set_ylabel(r"$p$", fontsize=26)       
    fig.tight_layout()
    fig.savefig("../plots/Boltzmann_dist.png", dpi=300)
    #plt.show()

if __name__ == "__main__":
    test_different_delta_U()
    test_plot_trajectories()
    test_plot_boltzmann()