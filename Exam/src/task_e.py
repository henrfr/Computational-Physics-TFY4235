from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
#from single_spin import evolve_spins
from solve_ode import evolve_spins, evolve_spins_old_but_working_pbc_linear
import numpy as np
from equations import normalize
import time


"""
Task e is now working 29.04 10:24"""

def task_e():
    gamma = 0.176 # 1.76*10**-1 T^-1 ps^-1 # /(T* ps)
    J = 1 # meV
    d_z = 0.1 # meV
    mu = 0.05788 # 5.788*10**-2 # meV*T^-1 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # 1*10**-3 # ps
    k_b = 0.08617 # 8.617*10**-2 # meV K^-1 # meV/K
    alpha = 0 # 0.05
    T = 0
    B = np.array([0,0,B_0])
    #B = np.array([0,0,0])
    e_z = np.array([0,0,1])
    d_z = 0 # Will only plot with one effect
    #J = 0 If J is 0, no spin will be transmitted


    N = 20000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 30
    N_particles_y = 1
    N_spin_components = 3

    # Initializes without padding padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a line 
    data[0,:,0] = normalize(np.array([0,0,1]))

    # Initialize first state with tilt
    data[0][0][0] = normalize(np.array([1,0,10]))

    data = evolve_spins_old_but_working_pbc_linear(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    t = np.arange(data.shape[0])

    set_plot_parameters()

    fig, axs = plt.subplots(1,3, sharey=True)
    all_x = data[:,:,0,0]
    all_y = data[:,:,0,1]
    all_z = data[:,:,0,2]

    axs[0].imshow(all_x, aspect="auto")
    axs[1].imshow(all_y, aspect="auto")
    axs[2].imshow(all_z, aspect="auto")
    plt.show()

    x = data[:,1,0,0]
    plt.plot(t, x, label=r"$S_x$")
    plt.plot(t, data[:,1,0,1], label=r"$S_y$")
    plt.plot(t, data[:,1,0,2], label=r"$S_z$")
    plt.legend()
    plt.show()

start = time.time()
task_e()
print(f"Task e) took {time.time()-start:.2f} seconds.")