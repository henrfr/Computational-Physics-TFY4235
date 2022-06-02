from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from equations import normalize
from solve_ode import evolve_spins_old, evolve_spins

def task_b():
    gamma = 0.176 # 1/(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # ps
    k_b = 0.08617 #  meV/K
    alpha = 0.05 # Sligthly damped
    T = 0 # K
    B = np.array([0,0,B_0])
    e_z = np.array([0,0,1])
    J = 0 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    d_z = 0 
    #B = np.array([0,0,0])

    N = 300000 # Number of time_steps
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)

    # Initializes with padding
    data = np.zeros(shape=(N_steps, 3, 3, 3))

    # Initialize first state
    data[0][1][1] = normalize(np.array([1,0,5]))

    data = evolve_spins(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma)
    t = np.arange(data.shape[0])*delta_t

    x = data[:,1,1,0]
    y = data[:,1,1,1]
    z = data[:,1,1,2]

    def f(x, omega, tau_inv):
        """Fits the data to a cos function modulated by an exponential with the amplitude known"""
        return data[0,1,1,0]*np.cos(omega*x)*np.exp(-tau_inv*x)

    # Uses scipy's curve_fit to find optimal parameters.
    popt, pcov = curve_fit(f, t, x,bounds=(0,np.inf))
    omega = popt[0]
    tau = 1/popt[1]
    print(f"Tau is: {tau:.2f}")
    print(f"Omega is: {omega:.2f}")
    error = (tau-(1/(alpha*omega)))/tau
    print(f"The error is {error:.2f}")

    def exp_decay(t, tau):
        """For plotting exponential decay"""
        return data[0,1,1,0]*np.exp(-t/tau)

    # Plotting
    set_plot_parameters(20,20)
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(t,x, label=r"$S_x$")
    axs[0].plot(t, exp_decay(t, tau), label=r"$e^{(-t/\tau)}$")
    axs[0].plot(t, f(t, omega, 1/tau), label=r"Fitted $S_x$", linestyle="dashed")
    axs[0].plot(t, -exp_decay(t, 1/(alpha*omega)), label=r"$-e^{(-\alpha\omega t)}$")
    axs[1].plot(t,z, label=r"$S_z$")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylabel("Spin value")
    axs[0].set_xlabel("Time (ps)")
    axs[1].set_xlabel("Time (ps)")
    fig.tight_layout()
    #fig.savefig("../plots/task_b.png", dpi=300)
    plt.show()

def compare_a_and_b():
    """This copy-pasted code makes one plot. Ignore it"""
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
    J = 0 # meV

    # Keep one of these to use only 1 symmetry-breaking term
    d_z = 0 
    #B = np.array([0,0,0])

    N = 300000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)

    # Initializes with padding
    data = np.zeros(shape=(N_steps, 3, 3, 3))

    # Initialize first state
    data[0][1][1] = normalize(np.array([1,0,1]))

    data = evolve_spins(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma)
    t = np.arange(data.shape[0])*delta_t

    x = data[:,1,1,0]
    y = data[:,1,1,1]
    z = data[:,1,1,2]

    gamma = 0.176 # 1.76*10**-1 T^-1 ps^-1 # /(T* ps)
    d_z = 0.1 # meV
    mu = 0.05788 # 5.788*10**-2 # meV*T^-1 # meV/T
    B_0 = 1.72 # T
    delta_t = 0.001 # 1*10**-3 # ps
    k_b = 0.08617 # 8.617*10**-2 # meV K^-1 # meV/K
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
    data[0][1][1] = normalize(np.array([1,0,1]))

    # Solve the ODE
    data1 = evolve_spins(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma)
    t = np.arange(data.shape[0])*delta_t
    x1 = data1[:,1,1,0]
    y1 = data1[:,1,1,1]
    z1 = data1[:,1,1,2]

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x2 = r*sin(phi)*cos(theta)
    y2 = r*sin(phi)*sin(theta)
    z2 = r*cos(phi)

    set_plot_parameters()
    #Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    ax.plot_surface(
        x2, y2, z2,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    ax.plot(x,y,z)
    ax.set_title(r"$\alpha = 0.05$")
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_box_aspect((1,1,1))
    ax.set_axis_off()

    ax1 = fig.add_subplot(122, projection='3d')
    ax1.plot_surface(
        x2, y2, z2,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    ax1.plot(x1,y1,z1)
    ax1.set_title(r"$\alpha = 0$")
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.set_box_aspect((1,1,1))
    ax1.set_axis_off()
    fig.tight_layout()
    #fig.savefig("../plots/task_b2.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    task_b()
    #compare_a_and_b()