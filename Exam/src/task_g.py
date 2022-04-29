from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
from solve_ode import evolve_spins, evolve_spins_old_but_working_pbc_square
import numpy as np
from equations import normalize, make_random_spins_square, get_magnetization, get_timeavg_magnetization
import time
import matplotlib.colors as plc

"""
This works and will find the ground state for both ferromagnetic and antiferromagnetic
Ferromagnetic will make boundaries that make collapse when there is no B-field.
Ferromagnetic will orient upwards when there is B-field
Antiferromagnetic will make spins opposing each other in Z. Just tune, J, B and d_z to see this.
"""

def task_g_assert_same_as_f():
    gamma = 0.176 # 1.76*10**-1 T^-1 ps^-1 # /(T* ps)
    J = 1 # meV
    d_z = 0.1 # meV
    mu = 0.05788 # 5.788*10**-2 # meV*T^-1 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # 1*10**-3 # ps
    k_b = 0.08617 # 8.617*10**-2 # meV K^-1 # meV/K
    alpha = 0.2 # 0.05
    T = 0
    B = np.array([0,0,B_0])
    #B = np.array([0,0,0])
    e_z = np.array([0,0,1])
    d_z = 0 # Will only plot with one effect
    #J = 0 If J is 0, no spin will be transmitted


    N = 30000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 4
    N_particles_y = 4
    N_spin_components = 3

    # Initializes without padding padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a line 
    data[0,:,:] = make_random_spins_square(N_particles_x, N_particles_y)
    #print(data[0])

    data = evolve_spins_old_but_working_pbc_square(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    t = np.arange(data.shape[0])

    set_plot_parameters()

    fig, axs = plt.subplots(1,3, sharey=True)
    all_x = data[-1,:,:,0]
    all_y = data[-1,:,:,1]
    all_z = data[-1,:,:,2]

    print(all_z)

    axs[0].imshow(all_x, aspect="auto")
    axs[1].imshow(all_y, aspect="auto")
    axs[2].imshow(all_z, aspect="auto", norm=plc.Normalize(-1, 1))
    plt.show()

    x = data[:,:,0,0]
    #plt.plot(t, x, label=r"$S_x$")
    #plt.plot(t, data[:,:,0,1], label=r"$S_y$")
    plt.plot(t, data[:,:,0,2], label=r"$S_z$")
    plt.plot(t, data[:,0,:,2], label=r"$S_z$")
    plt.legend()
    plt.show()


def task_g():
    gamma = 0.176 # 1.76*10**-1 T^-1 ps^-1 # /(T* ps)
    J = 1 # meV
    d_z = 0.1 # meV
    mu = 0.05788 # 5.788*10**-2 # meV*T^-1 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # 1*10**-3 # ps
    k_b = 0.08617 # 8.617*10**-2 # meV K^-1 # meV/K
    alpha = 0.2 # 0.05
    T = 1
    B = np.array([0,0,B_0])
    #B = np.array([0,0,0])
    e_z = np.array([0,0,1])
    d_z = 0 # Will only plot with one effect
    #J = 0 If J is 0, no spin will be transmitted


    N = 60000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 6
    N_particles_y = 6
    N_spin_components = 3

    # Initializes without padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a square lattice to the z direction 
    data[0,:,:,2] = 1
    print(data[0])

    data = evolve_spins_old_but_working_pbc_square(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    t = np.arange(data.shape[0])

    set_plot_parameters()

    fig, axs = plt.subplots(1,3, sharey=True)
    all_x = data[-1,:,:,0]
    all_y = data[-1,:,:,1]
    all_z = data[-1,:,:,2]

    print(all_z)

    axs[0].imshow(all_x, aspect="auto")
    axs[1].imshow(all_y, aspect="auto")
    axs[2].imshow(all_z, aspect="auto", norm=plc.Normalize(-1, 1))
    plt.show()

    # x = data[:,:,0,0]
    # #plt.plot(t, x, label=r"$S_x$")
    # #plt.plot(t, data[:,:,0,1], label=r"$S_y$")
    # plt.plot(t, data[:,:,0,2], label=r"$S_z$")
    # plt.plot(t, data[:,0,:,2], label=r"$S_z$")
    # plt.legend()
    # plt.show()

    M = np.zeros(data.shape[0])
    for i in range(len(M)):
        M[i] = get_magnetization(data[i])
    
    plt.plot(t, M, label=r"M(T,t)")
    #plt.ylim(-1, 1)
    plt.title(f"T = {T} K")
    plt.legend()
    plt.show()

    stable_point = int(input("When is M(T,t) fluctuating? (Answer in int): "))

    M_timeavg = get_timeavg_magnetization(M[stable_point:])
    M_timeavg = M_timeavg*np.ones_like(t)
    plt.plot(t, M, label=r"M(T,t)")
    plt.plot(t, M_timeavg, label=r"M(T)")
    plt.title(f"T = {T} K")
    plt.legend()
    plt.show()

def test_random_spins():
    delta_t = 0.001
    N = 120000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 10
    N_particles_y = 10
    N_spin_components = 3

    # Initializes without padding padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a line 
    rnd = make_random_spins_square(N_particles_x, N_particles_y)
    print(rnd)
    data[0,:,:] = rnd
    print(data[0])

    fig, axs = plt.subplots(1,3, sharey=True)
    all_x = data[0,:,:,0]
    all_y = data[0,:,:,1]
    all_z = data[0,:,:,2]

    print(all_x)

    axs[0].imshow(all_x, aspect="auto")
    axs[1].imshow(all_y, aspect="auto")
    axs[2].imshow(all_z, aspect="auto")
    plt.show()

#test_random_spins()
#start = time.time()
#task_g_assert_same_as_f()
#print(f"Task g_assert) took {time.time()-start:.2f} seconds.")
start = time.time()
task_g()
print(f"Task g) took {time.time()-start:.2f} seconds.")