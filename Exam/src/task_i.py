from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
from solve_ode import evolve_spins_pbc_square
import numpy as np
from equations import get_magnetization, get_timeavg_magnetization
import time

"""This code is a little messy, since it involves saving of files and loading etc. It
simply calculates temporal magnetization averages at many different T and B_0."""

def task_i():
    gamma = 0.176 # 1/(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # ps
    k_b = 0.08617 # meV/K
    alpha = 0.5 # A little damping
    #T = 10 # K
    #B = np.array([0,0,B_0])
    e_z = np.array([0,0,1])
    J = 1 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    d_z = 0 
    #B = np.array([0,0,0])


    N = 45000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 30
    N_particles_y = 30
    N_spin_components = 3


    # This block of code stores all average magnetization data for every time step for many T's and B's.
    # Change this
    T_ = [76, 78]
    B_0_ = [0.3, 1, 2]
    for b in range(len(B_0_)):
        B = np.array([0,0,B_0_[b]*B_0])

        for l in range(len(T_)):
            # Initializes without padding
            data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

            # Initialize all states on a square lattice to the z direction 
            data[0,:,:,2] = 1

            data = evolve_spins_pbc_square(data, N_steps, delta_t, mu, d_z, e_z,
                    B, J, alpha, k_b, T_[l], gamma, shape=(N_particles_x,N_particles_y))


            M = np.zeros(data.shape[0])
            for i in range(len(M)):
                M[i] = get_magnetization(data[i])

            np.save(f"../data/{int(3*B_0_[b])}_{T_[l]}.npy", M)    
            


def load_and_plot():
    T_ = np.arange(1,40,2)
    #T_ = np.append(T_, [72,74, 76, 78,80,82,84,86,88])
    B_0_ = [0.3, 1, 4]
    M_t_ = np.zeros(len(T_))
    M_t_std_ = np.zeros(len(T_))
    M_t_1 = np.zeros(len(T_))
    M_t_std_1 = np.zeros(len(T_))
    M_t_2 = np.zeros(len(T_))
    M_t_std_2 = np.zeros(len(T_))
    T_3= np.arange(1,80,5)
    B_0_3 = [1]
    M_t_3 = np.zeros(len(T_3))
    M_t_std_3 = np.zeros(len(T_3))

    # This is the important part
    # It calculates the time averaged magnetization for different B and T
    for l in range(len(T_)):
        M = np.load(f"../data/{int(3*B_0_[1])}_{T_[l]}_alt.npy", allow_pickle=True)
        stable_point = 20000
        M_timeavg = get_timeavg_magnetization(M[stable_point:])
        M_t_std_[l] = np.std(M[stable_point:])
        M_t_[l] = M_timeavg
        M1 = np.load(f"../data/{int(3*B_0_[2])}_{T_[l]}_alt.npy", allow_pickle=True)
        stable_point = 20000
        M_timeavg1 = get_timeavg_magnetization(M1[stable_point:])
        M_t_std_1[l] = np.std(M1[stable_point:])
        M_t_1[l] = M_timeavg1
        M2 = np.load(f"../data/{int(3*B_0_[0])}_{T_[l]}_alt.npy", allow_pickle=True)
        stable_point = 20000
        M_timeavg2 = get_timeavg_magnetization(M2[stable_point:])
        M_t_std_2[l] = np.std(M2[stable_point:])
        M_t_2[l] = M_timeavg2
    set_plot_parameters()
    for l in range(len(T_3)):
        M3 = np.load(f"../data/{int(3*B_0_3[0])}_{T_3[l]}_-J.npy", allow_pickle=True)
        stable_point = 20000
        M_timeavg3 = get_timeavg_magnetization(M3[stable_point:])
        M_t_std_3[l] = np.std(M3[stable_point:])
        M_t_3[l] = M_timeavg3

    fig, axs = plt.subplots(1,3, sharey=True, figsize=(14,4))
    axs[0].plot(T_, M_t_2, label=r"$M(T)$")
    axs[0].plot(T_, M_t_, label=r"$M(T), B = B_0$", linestyle="dashed")
    axs[0].fill_between(T_, M_t_2 - M_t_std_2, M_t_2 + M_t_std_2, alpha=0.3)
    axs[0].set_title(r"$B = 0.3B_0$")
    axs[0].set_xlabel("Temperature (K)")
    axs[0].set_ylabel("Magnetization")
    axs[0].plot(T_, np.zeros_like(T_), color="red")
    axs[0].legend()
    
    axs[1].plot(T_, M_t_, label=r"$M(T)$")
    axs[1].fill_between(T_, M_t_ - M_t_std_, M_t_ + M_t_std_, alpha=0.3)
    axs[1].set_title(r"$B = B_0$")
    axs[1].set_xlabel("Temperature (K)")
    axs[1].plot(T_, np.zeros_like(T_), color="red")
    axs[1].legend()

    axs[2].plot(T_, M_t_1, label=r"$M(T)$")
    axs[2].plot(T_, M_t_, label=r"$M(T), B = B_0$", linestyle="dashed")
    axs[2].fill_between(T_, M_t_1 - M_t_std_1, M_t_1 + M_t_std_1, alpha=0.3)
    axs[2].set_title(r"$B = 4B_0$")
    axs[2].set_xlabel("Temperature (K)")
    axs[2].plot(T_, np.zeros_like(T_), color="red")
    axs[2].legend()
    #fig.suptitle(r"J < 0, $d_z$ > 0")

    fig.tight_layout()
    fig.savefig("../plots/task_ialtplot.png", dpi=300)
    plt.show()

    plt.plot(T_3, M_t_3, label=r"$M(T)$, J < 0")
    plt.show()
    # plt.plot(T_, M_t_1, label=r"$\langle$M(T,t)$\rangle_t, B = 2B_0$")
    # plt.errorbar(T_, M_t_1, yerr = M_t_std_1,fmt='o',ecolor = 'red',color='yellow')
    # plt.plot(T_, M_t_, label=r"$\langle$M(T,t)$\rangle_t$")
    # plt.errorbar(T_, M_t_, yerr = M_t_std_,fmt='o',ecolor = 'red',color='yellow')
    # plt.plot(T_, M_t_2, label=r"$\langle$M(T,t)$\rangle_t$")
    # plt.errorbar(T_, M_t_2, yerr = M_t_std_2,fmt='o',ecolor = 'red',color='yellow')
    # plt.fill_between(T_, M_t_2 - M_t_std_2, M_t_2 + M_t_std_2)
    # plt.title("Phase diagram")
    # plt.legend()
    # plt.xlabel("Temperature (K)")
    # plt.ylabel("Magnetization")
    # plt.savefig("../plots/phase_diagram_B0.png", dpi=300)
    # plt.show()
if __name__ == "__main__":
    load_and_plot()
    start = time.time()
    task_i()
    print(f"Task h) took {time.time()-start:.2f} seconds.")
