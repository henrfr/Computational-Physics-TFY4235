import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from physics import *
from scipy.integrate import simpson

def test_potential():
    """
    Should return a plot showing one symmetric and one assymetric potential.
    """
    x = np.linspace(-2,2, 2000)
    u_r = U_r(x, 0.2)
    u_r_symmetric = U_r(x,0.5)
    plt.plot(x, u_r, label=r'$\alpha = 0.2$')
    plt.plot(x, u_r_symmetric,linestyle="dashed", color="red", label=r'$\alpha = 0.5$')
    plt.legend(loc="upper right")
    plt.tight_layout()
    #plt.savefig("/plots/potential.png", dpi=300)
    plt.show()

def test_gaussian():
    """
    Should return gaussian random numbers with mean of 0 and std. of 1
    """
    rnd = []
    for i in range(1,1001, 30):
        rnd.append(gaussian(i*10000))
    x = [len(e) for e in rnd]
    mean = [np.mean(e) for e in rnd]
    std = [np.std(e) for e in rnd]
    fig, axs = plt.subplots(2,1,figsize=(6,8), sharex=True)
    axs[0].plot(x, mean)
    axs[0].set_ylabel("Mean")
    axs[1].set_xlabel("N")
    axs[1].set_ylabel("Standard deviation")
    axs[0].set_xscale('log')
    axs[1].plot(x, std)
    fig.tight_layout()
    #fig.savefig("/plots/random.png", dpi=300)
    plt.show()

def test_boltzmann():
    """
    Should return a normalized boltzmann distribution
    """
    particle = {
        "r": 12e-9,
        "L": 20e-6,
        "eta": 1e-3,
        "kbT": 4.1656e-21,
        "delta_U": 1.2817e-17,
        "alpha": 0.2
    }
    def normalize(distribution):
        N = simpson(distribution, dx=1 / len(distribution))
        return distribution/N
    u_linear = np.linspace(0,1,1000)
    p = boltzmann_dist(u_linear, 10*particle["kbT"], particle["kbT"])
    p = normalize(p)
    plt.plot(u_linear, p)
    plt.ylabel("p")
    plt.xlabel("U")
    plt.show()

if __name__ == "__main__":
    set_plot_parameters()
    test_potential()
    test_gaussian()
    test_boltzmann()
