import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, hbar, alpha, c, m_e

def klein_nishina(k: float, THETA: np.ndarray, hbar: float, ALPHA: float, m_e: float, c:float, init_angle: float=0) -> np.ndarray:
    """
    Finds d(sigma)/d(Omega) using the Klein-Nishina equation. The equation is broken up into different factors
    for readability and avoiding performing the same calculations many times.
    Possible improvement: Send in the prefactor, factor0, istead of calculating it every time.
    
    Parameters
    ----------------------------------------------
    k : float
        The ratio between the energy of the incoming electron and the electron rest mass energy
    THETA : np.ndarray
        An array of angles between 0 and 2*pi.
    hbar : float
        The reduced Planck constant
    ALPHA : float
        The fine structure constant (~1/137.035)
    m_e : float
        The electron mass
    c : float
        The speed of light in vacuum.
    init_angle : float
        Which angle the electron is coming from. 0 by default. Should be between 0 and 2*pi.
    
    Returns
    ----------------------------------------------
    np.ndarray : d(sigma)/d(Omega) for angles between 0 and 2*pi for a specific incoming electron energy.
    """
    factor0 = (hbar*ALPHA)**2/(2*(m_e*c)**2) # A prefactor consisting of constants
    THETA = THETA + init_angle # Gives clockwise rotation of init_angle
    factor1 = k*(1-np.cos(THETA))
    factor2 = 1/(1+factor1)
    factor3 = (factor2 + 1 + factor1 - (np.sin(THETA))**2)
    return factor0*(factor2**2)*factor3

def test_klein_nishina():
    """
    Shows an example of usage. It should return a continous polar plot with decreasing circumferance
    for increasing E_gamma. Physically, this means that for increasing incoming energy, the electron
    is less likely to be deflected backwards, but will continue almost straight forward.
    """
    THETA = np.linspace(0, 2*np.pi, 360)
    E_gamma = np.array([(energy*10**6)*e for energy in np.linspace(0.05,1.5, 6)]) # The energy of the incoming electron
    E_e = m_e*c**2 # (0.511*10**6)*e 0.511MeV, the rest mass of an electron
    k = E_gamma/E_e
    init_angle = np.pi # Can be passed to klein-nishina() to change the initial angle

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    for k_value in k:        
        dist = klein_nishina(k_value, THETA, hbar, alpha, m_e, c)
        ax.plot(THETA, dist)
    plt.show()
test_klein_nishina()
