from plot_params import set_plot_parameters
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from equations import normalize, F
from solve_ode import heun, evolve_spins_old_but_working

# def evolve_spins(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
#     # For each time step
#     for i in range(N_steps-1):
#         # For every value in row
#         for row_val in range(1,shape[0]+1):
#             # For every value in column
#             for col_val in range(1,shape[1]+1):
#                 # Find current spin values
#                 S_j = data[i][row_val][col_val]

#                 # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
#                 neighbours = np.array([data[i][row_val][col_val+1],  # Down
#                                        data[i][row_val][col_val-1],  # Up
#                                        data[i][row_val+1][col_val],  # Right
#                                        data[i][row_val-1][col_val]]) # Left

#                 # Find the effective field at current position
#                 F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)

#                 # Perform a step with Heun
#                 data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
#     return data

def task_b():
    gamma = 0.176 # 1.76*10**-1 T^-1 ps^-1 # /(T* ps)
    J = 1 # meV Should be zero, but there are no neighbours so doesn't matter
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
    J = 0
    d_z = 0 # Will only plot with one effect



    N = 40000
    sim_time = N*delta_t
    N_steps = int(sim_time/delta_t)

    # Initializes with padding
    data = np.zeros(shape=(N_steps, 3, 3, 3))

    # Initialize first state
    data[0][1][1] = normalize(np.array([1,0,2]))

    data = evolve_spins_old_but_working(data, N_steps, delta_t, mu, d_z, e_z,
            B, J, alpha, k_b, T, gamma)
    t = np.arange(data.shape[0])

    x = data[:,1,1,0]

    def f(x, omega, tau_inv):
        return data[0,1,1,0]*np.cos(omega*x)*np.exp(-tau_inv*x)

    popt, pcov = curve_fit(f, t, x)
    print(popt)
    print(f"1/alpha is: {1/alpha:.2f}")
    print(f"omega*tau is: {popt[0]/popt[1]}")

    set_plot_parameters()
    plt.plot(t,x, label=r"$S_x$")
    plt.plot(t, f(t, popt[0], popt[1]), label=r"Fitted $S_x$")
    plt.legend()
    plt.show()

task_b()