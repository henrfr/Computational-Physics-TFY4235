import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_area(l,n):
    """
    Finds the area based on the original square side length (n+1)*l**4. Since all
    transformations add and remove the same amount of are, the area of the fractal
    is simply the originial sidelegnth squared.
    """
    return ((n+1)*l**4)**2

def N_omega(evals):
    """
    Finds the number of eigenstates with omega less than
    itself for every omega in evals, which is the eigenvalues.
    """
    # Some hardcoding to avoid problems with machine precision
    evals_int = evals*10**7
    evals_int = evals_int.astype(int)

    # If a value is smaller than the inspected, the number of
    # states below it is incremented.
    n_omega = np.zeros_like(evals)
    for i in range(len(evals)):
        for j in range(len(evals)):
            if evals_int[j] < evals_int[i]:
                n_omega[i] += 1
    return n_omega

def delta_N_omega(evals, n_omega,l,n):
    """
    Finds the delta_N correction. Evals are already squared, 
    which is why they appear in first order. n_omega is the IDOS,
    l the generation and n the number of points on each segment.
    """
    A = get_area(l,n)

    return [(A/(4*np.pi))*eval - n_val for n_val, eval in zip(n_omega, evals)]

def f(x,a, b):
    """a will represent d when fitted to logarithmic data"""
    return a*x + b

def f_non_log(x, a, b):
    """The same as f, but not logarithmic. Used for plotting."""
    return np.e**f(x,a,b)

def find_scaling_parameter(solution, l, n, k):
    """Finds the scaling parameter d by fitting the logarithm of the
    evals and dN_omega to a linear function. Solution is the file with
    eigenvalues, l is the generation, n the number of intermediate points
    and k the number of eigenvalues. If an ValueError is thrown, the correction
    term, dN_omega, is decreasing and becoming negative. Inspect the values
    and reduce the number of eigenvalues used if necessary."""
    # Load the data
    sol_current = solution[f"{l}_{n}_{k}"]
    evals = sol_current["evals"]

    # Find the IDOS, n_omega, and the correction term, dN_omega.
    n_omega= N_omega(evals)
    dN_omega = delta_N_omega(evals, n_omega, l, n)
    print(f"dN_omega is: {dN_omega}")

    # A quick sanity check of the eigenvalues
    plt.plot(np.sqrt(evals), dN_omega)
    plt.show()

    evals_log = np.log(np.sqrt(evals))
    try:
        dN_omega_log = np.log(dN_omega)
    except ValueError:
        print("Tried taking the logarithm of a negative number. Try reducing the number of eigenvalues.")

    popt, pcov = curve_fit(f, evals_log, dN_omega_log)

    # This is d
    print(popt)    
    print(f"d is: {popt[0]:.2f}")

    # Plotting values vs the best fit in a loglog plot
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.loglog(evals,dN_omega)
    ax.loglog(evals, f_non_log(evals_log, *popt), 'r-', label=f"Best fit for f(x)=ax+b: a = {popt[0]:.2f}, b = {popt[1]:.2f}")
    ax.legend()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\Delta N(\omega)$")
    plt.show()

def make_plot(solution):
    # Load the data
    sol_current = solution[f"{4}_{5}_{30}"]
    evals = sol_current["evals"]
    print(evals)
    # Find the IDOS, n_omega, and the correction term, dN_omega.
    n_omega= N_omega(evals)
    dN_omega = delta_N_omega(evals, n_omega, 4, 5)
    print(f"dN_omega is: {dN_omega}")

    # A quick sanity check of the eigenvalues
    plt.plot(np.sqrt(evals), dN_omega)
    plt.show()

    evals_log = np.log(np.sqrt(evals))
    try:
        dN_omega_log = np.log(dN_omega)
    except:
        print("Tried taking the logarithm of a negative number. Try reducing the number of eigenvalues.")

    popt, pcov = curve_fit(f, evals_log, dN_omega_log)

    # This is d
    print(popt)    
    print(f"d is: {popt[0]:.2f}")

    # Plotting values vs the best fit in a loglog plot
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].loglog(evals,dN_omega,label=r"$\Delta N(\omega)$ (Computed)")
    ax[0].loglog(evals, f_non_log(evals_log, *popt), 'r-', label=f"Best fit")
    ax[0].legend()
    ax[0].set_xlabel(r"$\omega$")
    ax[0].set_ylabel(r"$\Delta N(\omega)$")


    # Load the data
    sol_current = solution[f"{2}_{1}_{300}"]
    evals = sol_current["evals"][:250]

    # Find the IDOS, n_omega, and the correction term, dN_omega.
    n_omega= N_omega(evals)
    dN_omega = delta_N_omega(evals, n_omega, 2, 1)
    print(f"dN_omega is: {dN_omega}")

    evals_log = np.log(np.sqrt(evals))
    try:
        dN_omega_log = np.log(dN_omega)
    except:
        print("Tried taking the logarithm of a negative number. Try reducing the number of eigenvalues.")

    popt, pcov = curve_fit(f, evals_log, dN_omega_log)

    # This is d
    print(popt)    
    print(f"d is: {popt[0]:.2f}")

    ax[1].loglog(evals,dN_omega, label=r"$\Delta N(\omega)$ (Computed)")
    ax[1].legend()
    ax[1].set_xlabel(r"$\omega$")
    ax[1].set_ylabel(r"$\Delta N(\omega)$")

    ax[0].set_title(r"(l,n) = (4,5), $N_{\omega}$ = 30")
    ax[1].set_title(r"(l,n) = (2,1), $N_{\omega}$ = 250")
    plt.tight_layout()
    plt.savefig("../plots/scaling.png", dpi=300)
    plt.show()

def set_plot_parameters(size: int=16) -> None:
    """
    A helper function for setting plot parameters.
    """
    plt.style.use('seaborn-bright')
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rc('legend',fontsize=size-4)
    font = {'family' : 'serif', 
            'size': size}
    plt.rc('font', **font)
    plt.rc('lines', lw=2)

if __name__ == "__main__":
    sol = np.load("solutions.npy", allow_pickle=True).item()
    print(sol.keys())
    l = -1
    while not l >= 0:
        l = int(input("Generation (0,->): "))
    n = -1
    while not n >= 0:
        n = int(input("Number of points between each corner (0,->): "))
    k = -1
    while not k >= 0:
        k = int(input("Number of smallest eigenvalues (1,->): "))
    set_plot_parameters(size=20)
    #find_scaling_parameter(sol, l, n, k)
    make_plot(sol)