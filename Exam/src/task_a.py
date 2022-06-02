from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
import numpy as np
from equations import normalize
from solve_ode import evolve_spins_old, evolve_spins
import time

def task_a():
    gamma = 0.176 # 1/(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # ps
    k_b = 0.08617  # meV/K
    alpha = 0 # Undamped
    T = 0 # K
    B = np.array([0,0,B_0])
    e_z = np.array([0,0,1])
    J = 0 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    d_z = 0 
    #B = np.array([0,0,0])


    N = 100000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)

    # Initializes with padding
    data = np.zeros(shape=(N_steps, 3, 3, 3))

    # Initialize first state. (it will be in the center of the padding)
    data[0][1][1] = normalize(np.array([1,0,5]))

    # Solve the ODE
    print("Start")
    start = time.time()
    data = evolve_spins(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma)
    print(f"End: {time.time() -start:.2f} seconds.")
    print(data.shape[0])
    t = np.arange(data.shape[0])*delta_t

    # Plotting
    set_plot_parameters(20,20)
    fig, axs = plt.subplots(figsize=(8,6))
    axs.plot(t, data[:,1,1,0], label=r"$S_x$")
    axs.plot(t, data[:,1,1,1], label=r"$S_y$")
    axs.plot(t, data[:,1,1,2], label=r"$S_z$")
    axs.set_xlabel("Time (ps)")
    axs.set_ylabel("Spin value")
    axs.legend()
    fig.tight_layout()
    #fig.savefig("../plots/task_a.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    task_a()