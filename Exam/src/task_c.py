from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
#from single_spin import evolve_spins
from solve_ode import evolve_spins, evolve_spins_old_but_working
import numpy as np
from equations import normalize
import time

def task_c():
    gamma = 0.176 # 1.76*10**-1 T^-1 ps^-1 # /(T* ps)
    J = 1 # meV
    d_z = 0.1 # meV
    mu = 0.05788 # 5.788*10**-2 # meV*T^-1 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # 1*10**-3 # ps
    k_b = 0.08617 # 8.617*10**-2 # meV K^-1 # meV/K
    alpha = 0.1 # 0.05
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

    # Initializes with padding
    data = np.zeros(shape=(N_steps, N_particles_x + 2, N_particles_y + 2, N_spin_components))

    # Initialize all states on a line 
    data[0,:,1] = normalize(np.array([0,0,1]))
    # Remove left and right, such that padding is conserved
    data[0,0,1] = np.array([0,0,0])
    data[0,-1,1] = np.array([0,0,0])

    # Initialize first state with tilt
    data[0][1][1] = normalize(np.array([1,0,4]))


    data = evolve_spins_old_but_working(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    t = np.arange(data.shape[0])

    x = data[:,1,1,0]

    set_plot_parameters()

    # print(data[:,0,:,1])
    # data[:,0,:,1] = np.array([0,0,0])
    # data[:,-1,:,1] = np.array([0,0,0])

    fig, axs = plt.subplots(1,3, sharey=True)
    all_x = data[:,1:N_particles_x+1,1,0]
    all_y = data[:,1:N_particles_x+1,1,1]
    all_z = data[:,1:N_particles_x+1,1,2]

    axs[0].imshow(all_x, aspect="auto")
    axs[1].imshow(all_y, aspect="auto")
    axs[2].imshow(all_z, aspect="auto")
    plt.show()

    plt.plot(t, x, label=r"$S_x$")
    plt.plot(t, data[:,1,1,1], label=r"$S_y$")
    plt.plot(t, data[:,1,1,2], label=r"$S_z$")
    plt.legend()
    plt.show()



def test():
    # Initializes with padding
    data = np.zeros(shape=(2, 5, 3, 3))

    # Initialize first state
    data[0][1][1] = normalize(np.array([1,0,2]))
    data[0][2][1] = normalize(np.array([0,0,1]))
    data[0][3][1] = normalize(np.array([0,0,1]))

    data[0][2][0] = normalize(np.array([0,1,1]))
    print(data.shape)
    print(data)
    i = 2
    neighbours = np.array([data[0][i][1+1],  # Down
                            data[0][i][1-1],  # Up
                            data[0][i+1][1],  # Right
                            data[0][i-1][1]]) # Left
    
    print(neighbours)
    print(np.sum(neighbours, axis=0))

def test_2():
    # Initializes with padding
    data = np.zeros(shape=(2, 5, 3, 3))

    # Initialize all states
    data[0,:,1] = normalize(np.array([0,0,1]))

    # Remove left and right
    data[0,0,1] = np.array([0,0,0])
    data[0,-1,1] = np.array([0,0,0])
    print(data)


    # Initialize first state
    data[0][1][1] = normalize(np.array([1,0,2]))
    print(data)

    print(data.shape)
    #print(data)
    i = 2
    neighbours = np.array([data[0][i][1+1],  # Down
                            data[0][i][1-1],  # Up
                            data[0][i+1][1],  # Right
                            data[0][i-1][1]]) # Left
    
    print(neighbours)
    print(np.sum(neighbours, axis=0))
#test_2()
#test()
start = time.time()
task_c()
print(f"Task c took {time.time()-start:.2f} seconds.")