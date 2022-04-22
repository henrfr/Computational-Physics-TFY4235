import numpy as np
from scipy.constants import hbar, alpha, c, m_e
import matplotlib.pyplot as plt
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.klein_nishina import klein_nishina_eq
from src.plot_params import set_plot_parameters

def test_distribution(save=False):
    THETA = np.linspace(0, 2*np.pi, 360)
    E_gamma = 0.2*10**6 # The energy of the incoming electron in eV
    E_e = 511000 # in eV (m_e*c**2 = 0.511MeV) , the rest mass of an electron
    k = E_gamma/E_e
    init_angle = np.pi/2 # Can be passed to klein-nishina() to change the initial angle    
    prefactor_klein_nishina = (hbar*alpha)**2/(2*(m_e*c)**2)
    N = 500

    dist = klein_nishina_eq(k, THETA, prefactor_klein_nishina, init_angle=-init_angle)
    dist = dist/np.sum(dist)
    THETA = THETA*180/np.pi

    fig, axs = plt.subplots(1,1, figsize=(8,6))
    axs.plot(THETA, dist, lw=4)

    angles = np.empty(N*len(THETA))
    for i in range(N*len(THETA)):
        angles[i] = np.random.choice(THETA, p=dist)
    axs.hist(angles, bins=360, density=True)
    axs.set_xlabel(r"$\theta_{new}$ (degrees)")
    axs.set_ylabel(r"Probability density")
    plt.title(r"$E_{\gamma} = 0.2$ MeV, $\theta_{init} = 90 \degree$")
    plt.tight_layout()
    if save:
        plt.savefig("../plots/test_distribution.png", dpi=300)
    plt.show()

def test_klein_nishina_eq(save=False):
    """
    Shows an example of usage. It should return a continous polar plot with decreasing circumferance
    for increasing E_gamma. Physically, this means that for increasing incoming energy, the electron
    is less likely to be deflected backwards, but will continue almost straight forward.
    """
    THETA = np.linspace(0, 2*np.pi, 360)
    prefactor_klein_nishina = (hbar*alpha)**2/(2*(m_e*c)**2)
    E_gamma = np.array([(energy*10**6) for energy in np.linspace(0.05,1.5, 6)]) # The energy of the incoming electron
    E_e = 511000 # in eV (m_e*c**2 = 0.511MeV) , the rest mass of an electron
    k = E_gamma/E_e
    init_angle = 0 # Can be passed to klein-nishina() to change the initial angle

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, polar=True)
    for k_value in k:        
        dist = klein_nishina_eq(k_value, THETA, prefactor_klein_nishina, init_angle=init_angle)
        ax1.plot(THETA, dist)

    ax2 = fig.add_subplot(122)
    for k_value, energy in zip(k, E_gamma/10**6):        
        dist = klein_nishina_eq(k_value, THETA, prefactor_klein_nishina, init_angle=init_angle)
        ax2.plot(THETA*180/np.pi, dist, label=f"{energy:.2f} MeV")
    ax2.legend(loc="upper right")
    ax2.set_xlabel(r'$\theta$ (degrees)')
    ax2.set_ylabel(r'd$\sigma$/d$\Omega$/m$^2$sr$^-1$')
    plt.tight_layout()
    if save:
        plt.savefig("../plots/klein_nishina_eq.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    set_plot_parameters(size=20)
    test_klein_nishina_eq()
    set_plot_parameters(size=18)
    test_distribution()