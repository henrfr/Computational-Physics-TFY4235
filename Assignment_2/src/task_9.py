from euler import *
from physics import *
from simulate import *

"""
This file plots trajectories in a flashing potential and finds average velocities.
"""

particle = {
    "r": 12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}

# Plots of trajectories
def test_plot_trajectories():
    """
    Will plot trajectories of particles in flashing potential with different periods.
    """

    # Sets initial parameters
    set_plot_parameters(16)
    N_particles = [1,1,1]
    sim_time = [10,30,100]
    frequency = [25, 0.7, 0.025] # Will correspond to TAU as 1/f
    tols = [0.1, 0.4, 0.6] # These should ideally all be below 0.1 or so, but are increased for reduction in runtime.
    results = []

    # Finds the trajectories of the particles
    for N, t, f, tol in zip(N_particles, sim_time, frequency, tols):
        positions = simulate_particle_position_flashing_potential(particle,N,t,0.2,1/f, tol)
        t = np.linspace(0, t, len(positions[:,0]))
        results.append((positions, t))
    
    # Plots the results
    fig, axs = plt.subplots(1,3,figsize=(13,4))
    for i in range(3):
        pos = results[i][0]
        t = results[i][1]
        # Converts to real units
        axs[i].plot(t, pos[:,0]*1e6*particle["L"], label = fr"$\tau = {1/frequency[i]:.2f}$")
        axs[i].set_xlabel(r"Time [$s$]")
        axs[i].legend()
    axs[0].set_ylabel(r"Position [$\mu m$]")
    fig.tight_layout()
    fig.savefig("../plots/trajectories_flashing3.png", dpi=300)
    plt.show()
#test_plot_trajectories()




def test_avg_drift_velocity(save_sol=True):
    """ 
    This function calculates the average drift velocity of a particle in flashing potentials
    of different periods. The mean is found by dividing the final position by the total time. The
    sim_time must thus be adjusted such that if the period is long, the sim_time is long enough.
    Note: the function uses data generated in task_11.
    The function will be left untidy at the moment, but the commented lines can be commented out
    in order to generate new data.
    """
    set_plot_parameters(20)
    N_particles = 100
    sim_time = 20
    # taus = np.linspace(0.1, 5, 20)
    # start = time.time()
    # avg_lst = []
    # for tau in taus:
    #     positions = simulate_particle_position_flashing_potential_less_memory(particle,N_particles,sim_time,0.2,tau, 0.9)
    #     avg_lst.append(np.mean(positions)*1e6*particle["L"]/sim_time)
    # print(f"Sim time is {time.time()-start:.2f} seconds")
    # if save_sol:
    #     np.save("../data/mean_velocity_0_5.npy", (avg_lst, taus))
    data = np.load("../data/mean_velocity_0_5.npy", allow_pickle=True)
    avg_lst = data[0]
    taus = data[1]
    data_1 = np.load("../data/mean_velocity_042_065_20.npy", allow_pickle=True)
    avg_lst_2 = data_1[0]
    taus_2 = data_1[1]
    def running_mean(a):
        """ Helper function which smooths data"""
        new = []
        for i in range(1, len(a)-1):
            new.append(np.mean([a[i-1], a[i], a[i+1]]))
        return new
    # taus_2 = np.linspace(0.42, 0.65, 20)
    # start = time.time()
    # N_particles = 100
    # sim_time = 20
    # avg_lst_2 = []
    # for tau in taus_2:
    #     positions = simulate_particle_position_flashing_potential_less_memory(particle,N_particles,sim_time,0.2,tau, 0.9)
    #     avg_lst_2.append(np.mean(positions)*1e6*particle["L"]/sim_time)
    # print(f"Sim time is {time.time()-start:.3f} seconds")
    avg_lst_2 = running_mean(avg_lst_2)
    taus_2 = running_mean(taus_2)
    # if save_sol:
    #     np.save("../data/mean_velocity_042_065_many_mean.npy", (avg_lst_2, taus_2))
    fig, axs = plt.subplots(1,3, figsize=(13,5))
    axs[0].plot(taus, avg_lst, label=r"$r = r_1$")
    axs[1].plot(taus_2, avg_lst_2, label=r"$r = r_1$")

    data_pred = np.load("../data/mean_velocity_0_5_particle2_pred.npy", allow_pickle=True)
    particle_2_vel_pred = data_pred[0]
    particle_2_taus_pred = data_pred[1]

    data = np.load("../data/mean_velocity_0_5_particle2.npy", allow_pickle=True)      
    particle_2_vel = data[0]
    particle_2_taus = data[1]

    axs[2].plot(particle_2_taus_pred[:len(particle_2_taus_pred)-15], particle_2_vel_pred[:len(particle_2_taus_pred)-15], label=r"$Predicted$")
    axs[2].plot(particle_2_taus[:len(particle_2_taus)-28], particle_2_vel[:len(particle_2_taus)-28], label=r"$r = r_2$")

    for ax in axs:
        ax.set_xlabel(r"$\tau [s]$")
        ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=14)
    axs[0].set_ylabel(r"$\langle v \rangle$ $[\mu m/s]$")
    fig.tight_layout()
    fig.savefig("../plots/mean_velocity.png", dpi=300)
    #plt.show()

if __name__ == "__main__":
    test_plot_trajectories()
    test_avg_drift_velocity()