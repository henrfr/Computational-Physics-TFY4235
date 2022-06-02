from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
from solve_ode import evolve_spins_pbc_linear, evolve_spins_pbc_linear_old
import numpy as np
from equations import normalize
import time

def task_e():
    gamma = 0.176 # 1/(T* ps)
    J = 1 # meV
    d_z = 0.1 # meV
    mu = 0.05788 #  meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 #  ps
    k_b = 0.08617 #  meV/K
    alpha = 0 # 0.05
    T = 0
    B = np.array([0,0,B_0])
    #B = np.array([0,0,0])
    e_z = np.array([0,0,1])

    d_z = 0 # Will only plot with one effect
  


    N = 10000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 20
    N_particles_y = 1
    N_spin_components = 3

    # Initializes without padding padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a line 
    data[0,:,0] = normalize(np.array([0,0,1]))

    # Initialize first state with tilt
    data[0][0][0] = normalize(np.array([1,0,10]))

    data = evolve_spins_pbc_linear(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    t = np.arange(data.shape[0])*delta_t

    # Plotting
    set_plot_parameters()
    fig, axs = plt.subplots(1,3, sharey=True, figsize=(14,4))
    all_x = data[:,:,0,0]
    all_y = data[:,:,0,1]
    all_z = data[:,:,0,2]

    im0 = axs[0].imshow(all_x, aspect="auto", interpolation="none")
    axs[0].set_title(r"$S_x$")
    axs[0].set_xlabel("Spin number")
    axs[0].set_ylabel("Time (fs)")
    im1 = axs[1].imshow(all_y, aspect="auto", interpolation="none")
    axs[1].set_title(r"$S_y$")
    axs[1].set_xlabel("Spin number")
    im2 = axs[2].imshow(all_z, aspect="auto", interpolation="none")
    axs[2].set_title(r"$S_z$")
    axs[2].set_xlabel("Spin number")
    plt.colorbar(im0,ax = axs[0])
    plt.colorbar(im1,ax = axs[1])
    plt.colorbar(im2,ax = axs[2])
    fig.tight_layout()
    #fig.savefig("../plots/task_eimshow.png", dpi=300)
    plt.show()

    fig, axs = plt.subplots(1,1)
    axs.plot(t, data[:,-2,0,0], label=r"$S_{N-1}$", lw=4)
    axs.plot(t, data[:,-1,0,0], label=r"$S_{N}$", lw=4)
    axs.plot(t, data[:,0,0,0], label=r"$S_{0}$", lw=4)
    axs.plot(t, data[:,1,0,0], label=r"$S_{1}$", linestyle="dotted", lw=4)
    axs.plot(t, data[:,2,0,0], label=r"$S_{2}$", linestyle="dotted", lw=4)
    axs.set_xlabel("Time (ps)")
    axs.set_ylabel(r"Spin value ($S_x$)")
    axs.legend()
    fig.tight_layout()
    #fig.savefig("../plots/task_eplot_alt.png", dpi=300)
    plt.show()


    fig = plt.figure(figsize=(12,6))
    axs1 = fig.add_subplot(121)
    axs1.plot(t, data[:,-2,0,0], label=r"$S_{N-1}$", lw=4)
    axs1.plot(t, data[:,-1,0,0], label=r"$S_{N}$", lw=4)
    axs1.plot(t, data[:,0,0,0], label=r"$S_{1}$", lw=4)
    axs1.plot(t, data[:,1,0,0], label=r"$S_{2}$", linestyle="dotted", lw=4)
    axs1.plot(t, data[:,2,0,0], label=r"$S_{3}$", linestyle="dotted", lw=4)
    axs1.set_xlabel("Time (ps)")
    axs1.set_ylabel(r"Spin value ($S_x$)")
    axs1.legend()

    axs = fig.add_subplot(122)
    axs.plot(data[3500:5000,-2,0,0], data[3500:5000,-2,0,1], label=r"$S_{N-1}$", lw=4)
    axs.plot(data[3500:5000,-1,0,0], data[3500:5000,-1,0,1], label=r"$S_{N}$", lw=4)
    axs.plot(data[3500:5000,0,0,0], data[3500:5000,0,0,1], label=r"$S_1$", lw=4)
    axs.plot(data[3500:5000,1,0,0], data[3500:5000,1,0,1], label=r"$S_2$", lw=4, linestyle="dotted")
    axs.plot(data[3500:5000,2,0,0], data[3500:5000,2,0,1], label=r"$S_3$", lw=4, linestyle="dotted")
    axs.set_xlabel(r"$S_x$")
    axs.set_ylabel(r"$S_y$")
    axs.legend()

    fig.tight_layout()
    #fig.savefig("../plots/task_edouble.png", dpi=300)
    plt.show()

    from matplotlib import animation

    X, Y = np.mgrid[:19:20j,:1:1j]
    U = data[0,:,:,0]
    V = data[0,:,:,1]

    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X, Y, U, V, pivot='tail',angles='xy', scale_units='xy', scale=0.1)

    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 1)

    def update_quiver(num, Q, X, Y):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """

        U = data[num*10,:,:,0]
        V = data[num*10,:,:,1]
        C = data[num*10,:,:,2]
        Q.set_UVC(U,V,C)

        return Q,

    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                                interval=40, blit=True)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    start = time.time()
    task_e()
    print(f"Task e) took {time.time()-start:.2f} seconds.")