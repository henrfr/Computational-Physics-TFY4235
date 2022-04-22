import numpy as np
from numba import njit

@njit(cache=True)
def klein_nishina_eq(k: float, THETA: np.ndarray, prefactor: float, init_angle: float=0) -> np.ndarray:
    """
    Finds d(sigma)/d(Omega) using the Klein-Nishina equation. The equation is broken up into different factors
    for readability and avoiding performing the same calculations many times.
    
    Parameters
    ----------------------------------------------
    k : float
        The ratio between the energy of the incoming electron and the electron rest mass energy
    THETA : np.ndarray
        An array of angles between 0 and 2*pi.
    prefactor : float
        (hbar*ALPHA)**2/(2*(m_e*c)**2). The constants are pre-calculated.
    init_angle : float
        Which angle the electron is coming from. 0 by default. Should be between 0 and 2*pi.
    
    Returns
    ----------------------------------------------
    np.ndarray : d(sigma)/d(Omega) for angles between 0 and 2*pi for a specific incoming electron energy.
    """
    THETA = THETA + init_angle # Gives clockwise rotation of init_angle
    factor1 = k*(1-np.cos(THETA))
    factor2 = 1/(1+factor1)
    factor3 = (factor2 + 1 + factor1 - (np.sin(THETA))**2)
    return prefactor*(factor2**2)*factor3
