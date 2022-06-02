from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
from solve_ode import evolve_spins_pbc_square
import numpy as np
from equations import make_random_spins_square, get_magnetization, get_timeavg_magnetization
import time

def task_g_assert_same_as_f():
    gamma = 0.176 # 1/(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # ps
    k_b = 0.08617 # meV/K
    alpha = 0.5 # A little damping
    T = 0 # K
    B = np.array([0,0,B_0])
    e_z = np.array([0,0,1])
    J = -1 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    #d_z = 0 
    B = np.array([0,0,0])

    N = 10000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 10
    N_particles_y = 10
    N_spin_components = 3

    # Initializes without padding padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a lattice with random directions
    data[0,:,:] = make_random_spins_square(N_particles_x, N_particles_y)

    start = time.time()
    data = evolve_spins_pbc_square(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    print(f"Task g_assert) took {time.time()-start:.2f} seconds.")
    t = np.arange(data.shape[0])

    # Plotting
    set_plot_parameters()
    # fig, axs = plt.subplots(1,3, sharey=True, figsize=(14,4))
    # all_x = data[-1,:,:,0]
    # all_y = data[-1,:,:,1]
    # all_z = data[-1,:,:,2]

    # im0 = axs[0].imshow(all_x, aspect="auto", interpolation="none")
    # axs[0].set_title(r"$S_x$")
    # axs[0].set_xlabel("Spin number")
    # axs[0].set_ylabel("Spin number")
    # im1 = axs[1].imshow(all_y, aspect="auto", interpolation="none")
    # axs[1].set_title(r"$S_y$")
    # axs[1].set_xlabel("Spin number")
    # im2 = axs[2].imshow(all_z, aspect="auto", interpolation="none")
    # axs[2].set_title(r"$S_z$")
    # axs[2].set_xlabel("Spin number")
    # fig.suptitle(r"J < 0, $d_z$ > 0")
    # plt.colorbar(im0,ax = axs[0])
    # plt.colorbar(im1,ax = axs[1])
    # plt.colorbar(im2,ax = axs[2])
    # fig.tight_layout()
    # fig.savefig("../plots/task_gimshow_-J_dz_20.png", dpi=300)
    # plt.show()

    from matplotlib import animation

    X, Y = np.mgrid[:9:10j,:9:10j]
    U = data[0,:,:,0]
    V = data[0,:,:,1]

    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X, Y, U, V, pivot='tail',angles='xy', scale_units='xy', scale=1)

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)

    def update_quiver(num, Q, X, Y):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """

        U = data[num*30,:,:,0]
        V = data[num*30,:,:,1]
        C = np.floor(data[num*30,:,:,2])
        Q.set_UVC(U,V,C)

        return Q,

    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                                interval=1, blit=True)
    f = r"animation.gif" 
    writergif = animation.PillowWriter(fps=10) 
    anim.save(f, writer=writergif)
    fig.tight_layout()
    plt.show()
    # TODO: make an animation!


def task_g():
    gamma = 0.176 # 1/(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # ps
    k_b = 0.08617 # meV/K
    alpha = 0.5 # A little damping
    T = 60 # K
    B = np.array([0,0,B_0])
    e_z = np.array([0,0,1])
    J = 1 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    d_z = 0 
    #B = np.array([0,0,0])

    N = 15000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 30
    N_particles_y = 30
    N_spin_components = 3

    # Initializes without padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a square lattice to the z direction 
    data[0,:,:,2] = 1
    start = time.time()
    data = evolve_spins_pbc_square(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    print(f"Task g) took {time.time()-start:.2f} seconds.")
    t = np.arange(data.shape[0])*delta_t

    set_plot_parameters()

    M = np.zeros(data.shape[0])
    for i in range(len(M)):
        M[i] = get_magnetization(data[i])
    
    # Plots the magnetization and finds a point of stability by visual inspection.
    plt.plot(t, M, label=r"M(T,t)")
    plt.title(f"T = {T} K")
    plt.legend()
    plt.show()

    #stable_point = int(input("When is M(T,t) fluctuating? (Answer in int): "))
    stable_point = 10000
    M_timeavg = get_timeavg_magnetization(M[stable_point:])
    M_timeavg = M_timeavg*np.ones_like(t)
    plt.plot(t, M, label=r"M(T,t)")
    plt.plot(t, M_timeavg, label=r"M(T)")
    plt.xlabel("Time (ps)")
    plt.ylabel("Magnetization")
    plt.title(f"T = {T} K")
    plt.tight_layout()
    plt.legend()
    plt.savefig("../plots/task_g.png")
    plt.show()


def make_spin_wave():
    gamma = 0.176 # 1/(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # ps
    k_b = 0.08617 # meV/K
    alpha = 0.5 # A little damping
    T = 0 # K
    B = np.array([0,0,B_0])
    e_z = np.array([0,0,1])
    J = -1 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    #d_z = 0 
    B = np.array([0,0,0])

    N = 10000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)
    N_particles_x = 10
    N_particles_y = 10
    N_spin_components = 3

    # Initializes without padding padding
    data = np.zeros(shape=(N_steps, N_particles_x, N_particles_y, N_spin_components))

    # Initialize all states on a lattice with random directions
    #data[0,:,:] = make_random_spins_square(N_particles_x, N_particles_y)
    data[0,:,:] = [0,0,1]
    data[0,0,0] = [1,0,5]

    start = time.time()
    data = evolve_spins_pbc_square(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma, shape=(N_particles_x,N_particles_y))
    print(f"Task g_assert) took {time.time()-start:.2f} seconds.")
    t = np.arange(data.shape[0])

    # Plotting
    set_plot_parameters()
    # fig, axs = plt.subplots(1,3, sharey=True, figsize=(14,4))
    # all_x = data[-1,:,:,0]
    # all_y = data[-1,:,:,1]
    # all_z = data[-1,:,:,2]

    # im0 = axs[0].imshow(all_x, aspect="auto", interpolation="none")
    # axs[0].set_title(r"$S_x$")
    # axs[0].set_xlabel("Spin number")
    # axs[0].set_ylabel("Spin number")
    # im1 = axs[1].imshow(all_y, aspect="auto", interpolation="none")
    # axs[1].set_title(r"$S_y$")
    # axs[1].set_xlabel("Spin number")
    # im2 = axs[2].imshow(all_z, aspect="auto", interpolation="none")
    # axs[2].set_title(r"$S_z$")
    # axs[2].set_xlabel("Spin number")
    # fig.suptitle(r"J < 0, $d_z$ > 0")
    # plt.colorbar(im0,ax = axs[0])
    # plt.colorbar(im1,ax = axs[1])
    # plt.colorbar(im2,ax = axs[2])
    # fig.tight_layout()
    # fig.savefig("../plots/task_gimshow_-J_dz_20.png", dpi=300)
    # plt.show()

    from matplotlib import animation

    X, Y = np.mgrid[:9:10j,:9:10j]
    U = data[0,:,:,0]
    V = data[0,:,:,1]

    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X, Y, U, V, pivot='tail',angles='xy', scale_units='xy', scale=1)

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)

    def update_quiver(num, Q, X, Y):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """

        U = data[num*30,:,:,0]
        V = data[num*30,:,:,1]
        C = np.floor(data[num*30,:,:,2])
        Q.set_UVC(U,V,C)

        return Q,

    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                                interval=1, blit=True)
    f = r"spin_wave.gif" 
    writergif = animation.PillowWriter(fps=10) 
    anim.save(f, writer=writergif)
    fig.tight_layout()
    plt.show()
    # TODO: make an animation!

if __name__ == "__main__":
    #make_spin_wave()
    # start = time.time()
    # task_g_assert_same_as_f()
    # print(f"Task g_assert) took {time.time()-start:.2f} seconds.")
    start = time.time()
    task_g()
    print(f"Task g) took {time.time()-start:.2f} seconds.")