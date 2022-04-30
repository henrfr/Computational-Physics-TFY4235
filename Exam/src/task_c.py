from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
from solve_ode import evolve_spins, evolve_spins_old
import numpy as np
from equations import normalize
import time

def task_c():
    gamma = 0.176 # 1.76*10**-1 T^-1 ps^-1 # /(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # 5.788*10**-2 # meV*T^-1 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # 1*10**-3 # ps
    k_b = 0.08617 # 8.617*10**-2 # meV K^-1 # meV/K
    alpha = 0.05 # Sligthly damped
    T = 0 # K
    B = np.array([0,0,B_0])
    e_z = np.array([0,0,1])
    J = 1 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    #d_z = 0 
    B = np.array([0,0,0])

    N = 10000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 20
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
    data[0][1][1] = normalize(np.array([1,0,5]))


    data = evolve_spins(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    t = np.arange(data.shape[0])*delta_t


    #set_plot_parameters()

    # print(data[:,0,:,1])
    # data[:,0,:,1] = np.array([0,0,0])
    # data[:,-1,:,1] = np.array([0,0,0])

    set_plot_parameters()
    fig, axs = plt.subplots(1,3, sharey=True, figsize=(14,4))
    all_x = data[:,1:N_particles_x+1,1,0]
    all_y = data[:,1:N_particles_x+1,1,1]
    all_z = data[:,1:N_particles_x+1,1,2]

    im0 = axs[0].imshow(all_x, aspect="auto")
    axs[0].set_title(r"$S_x$")
    axs[0].set_xlabel("Spin number")
    axs[0].set_ylabel("Time (fs)")
    im1 = axs[1].imshow(all_y, aspect="auto")
    axs[1].set_title(r"$S_y$")
    axs[1].set_xlabel("Spin number")
    axs[1].set_ylabel("Time (fs)")
    im2 = axs[2].imshow(all_z, aspect="auto")
    axs[2].set_title(r"$S_z$")
    axs[2].set_xlabel("Spin number")
    axs[2].set_ylabel("Time (fs)")
    plt.colorbar(im0,ax = axs[0])
    plt.colorbar(im1,ax = axs[1])
    plt.colorbar(im2,ax = axs[2])
    fig.tight_layout()
    fig.savefig("../plots/task_cimshow.png", dpi=300)
    plt.show()

    fig, axs = plt.subplots(1,1)
    axs.plot(t, data[:,1:4,1,0], label=r"$S_x$")
    axs.set_xlabel("Time (ps)")
    axs.set_ylabel("Spin value")
    #axs[1].plot(t, data[:,1:5,1,1], label=r"$S_y$")
    #axs[2].plot(t, data[:,1:5,1,2], label=r"$S_z$")
    axs.legend()
    #axs[1].legend()
    #axs[2].legend()
    fig.tight_layout()
    fig.savefig("../plots/task_cplot.png", dpi=300)
    plt.show()

    from matplotlib import animation

    X, Y = np.mgrid[:4:5j,:1:1j]
    U = data[0,1:6,1:2,0]
    V = data[0,1:6,1:2,1]

    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X, Y, U, V, pivot='mid', color='r', units='inches')

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 1)

    def update_quiver(num, Q, X, Y):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """

        U = data[num*10,1:6,1:2,0]
        V = data[num*10,1:6,1:2,1]
        C = data[num*10,1:6,1:2,2]
        Q.set_UVC(U,V,C)

        return Q,

    # you need to set blit=False, or the first set of arrows never gets
    # cleared on subsequent frames
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                                interval=400, blit=True)
    fig.tight_layout()
    plt.show()

start = time.time()
task_c()
print(f"Task c took {time.time()-start:.2f} seconds.")