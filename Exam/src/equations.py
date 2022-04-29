import numpy as np # Use numpy to operate on arrays. numpy is written in C
from numba import njit # Used for JIT-compilation

@njit()
def normalize(spin_coords: np.ndarray) -> np.ndarray:
    """Normalizes the size of a vector.

    Args:
        spin_coords (np.ndarray): The spin coordinates. Ex: [S_x, S_y, S_z]

    Returns:
        np.ndarray: A vector with length 1
    """
    return spin_coords / np.sqrt(np.sum(spin_coords**2))

@njit()
def get_magnetization(spin_grid: np.ndarray) -> float:
    """Takes a 2D grid of spins and returns the mean of the z-components.

    Args:
        spin_grid (np.ndarray): A 2D grid of spins. shape=(N_x, N_y, 3), where 3 are the components

    Returns:
        float: _description_
    """
    s_z_ = spin_grid[:,:,2] # Gets all z-components
    return np.mean(s_z_)

@njit()
def get_timeavg_magnetization(magnetization_: np.ndarray) -> float:
    """Finds the time_average of the magnetization.

    Args:
        magnetization (np.ndarray): A 1D array of magnetizations

    Returns:
        float: The time averaged magnetization.
    """
    return np.mean(magnetization_)

@njit()
def F_th(thermal_constant: float, rand_vec: np.ndarray) -> np.ndarray:
    """The thermal contribution to the electric field.

    Args:
        thermal_constant (float): The thermal constant calculated according to make_thermal_constant()
        rand_vec (np.ndarray): A vector with three normally distributed numbers
    Returns:
        np.ndarray: An [x,y,z] array modelling the thermal contribution.
    """
    #rnd_vec = make_3D_gaussian_alt()
    return rand_vec*thermal_constant

@njit()
def make_thermal_constant(alpha: float, k_b: float, T: float, gamma: float, mu: float, delta_t: float) -> float:
    """Calculates the thermal constant, which is constant for every simulation.

    Args:
        alpha (float): The damping parameter
        k_b (float): The boltzmann constant
        T (float): The temperature
        gamma (float): The gyromagnetic ratio
        mu (float): The magnetic moment
        delta_t (float): The timestep
    Returns:
        np.ndarray: An [x,y,z] array modelling the thermal contribution.
    """
    return np.sqrt((2*alpha*k_b*T)/(gamma*mu*delta_t))
@njit()
def LLG(S_j: np.ndarray, F_j: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """The Landau-Lifshitz-Gilbert equation describing the precessional motion of magnetization
    in a solid. 

    Args:
        S_j (np.ndarray): The spin components at the current location ([x,y,z])
        F_j (np.ndarray): The effective field at the current location ([x,y,z])
        gamma (float): The gyromagnetic ratio
        alpha (float): The damping constant

    Returns:
        np.ndarray: The change in spin at the current position with time. [x,y,z]
    """
    prefactor = -gamma/(1+alpha**2)
    factor1 = np.cross(S_j, F_j) # Stored for convenience
    factor2 = np.cross(alpha*S_j, factor1)
    return prefactor*(factor1 + factor2)

@njit()
def F_eff(mu: float, S_j: np.ndarray, d_z: float, e_z: np.ndarray, B: np.ndarray, J: float, neighbours: np.ndarray) -> np.ndarray:
    """A term related to the functional derivative of the Hamiltionian.
    The np.sum() sums over all the neighbouring spins.

    Args:
        mu (float): The magnetic moment
        S_j (np.ndarray): The spin components at the current location ([s_x, s_y, s_z])
        d_z (float): The anisotropy constant
        e_z (np.ndarray): The unit vector in the z-direction ([0,0,1])
        B (np.ndarray): The magnetic field (Always [0,0,B_0] or [0,0,0])
        J (float): The coupling constant
        neighbours (np.ndarray): The spin components of the nearest neighbours

    Returns:
        np.ndarray: Returns a field vector on the form [f_x, f_y, f_z]
    """
    return (1/mu) * (J*np.sum(neighbours, axis=0) + 2*d_z*S_j[2]*e_z + mu*B)

@njit()
def F(mu, S_j, d_z, e_z, B, J, neighbours, thermal_constant, rand_vec: np.ndarray) -> np.ndarray:
    """I will not comment this, as it is only the sum of the two field terms,
    constituting the effective field at a position."""
    return F_eff(mu, S_j, d_z, e_z, B, J, neighbours) + F_th(thermal_constant, rand_vec)

@njit()
def random_matrix(shape):
    rand_mat = np.empty(shape=(shape[0], shape[1], 3))
    for i in range(shape[0]):
        for j in range(shape[1]):
            # rand_mat[i][j] = random method
            rand_mat[i][j] = make_3D_gaussian()
    return rand_mat
@njit()
def make_3D_gaussian():
    rnd_spin = np.zeros(3)
    theta = np.random.rand()*2*np.pi
    phi = np.random.rand()*np.pi

    s_x = (np.sin(phi)*np.cos(theta))
    s_y = (np.sin(phi)*np.sin(theta))
    s_z = (np.cos(phi))  

    rnd_spin[0] = s_x
    rnd_spin[1] = s_y
    rnd_spin[2] = s_z

    rnd_spin = rnd_spin*np.sin(phi)*np.random.normal()

    return rnd_spin

@njit()
def make_3D_gaussian_alt():
    rnd_spin = np.zeros(3)
    rnd_spin[0] = np.random.normal()
    rnd_spin[1] = np.random.normal()
    rnd_spin[2] = np.random.normal()
    return normalize(rnd_spin)

@njit()
def make_random_spins_linear(N):
    rnd_spins = np.zeros((N, 3))

    for i in range(N):
        theta = np.random.rand()*2*np.pi
        phi = np.random.rand()*np.pi

        s_x = (np.sin(phi)*np.cos(theta))
        s_y = (np.sin(phi)*np.sin(theta))
        s_z = (np.cos(phi))    

        rnd_spins[i] = [s_x, s_y, s_z]
    return rnd_spins


def make_random_spins_square(N_x, N_y):
    rnd_spins = np.zeros((N_x, N_y, 3))

    for i in range(N_x):
        for j in range(N_y):
            theta = np.random.rand()*2*np.pi
            phi = np.random.rand()*np.pi

            s_x = (np.sin(phi)*np.cos(theta))
            s_y = (np.sin(phi)*np.sin(theta))
            s_z = (np.cos(phi))    

            rnd_spins[i][j] = [s_x, s_y, s_z]
    return rnd_spins

def test():
    # (x, y, S)
    a = np.zeros((3,3,3))
    a[1][1][2] = 1
    print(a)
    print(a[0][0])
# test()

# print(normalize(np.array([1,1,100])))