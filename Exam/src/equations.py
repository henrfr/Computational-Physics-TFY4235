import numpy as np # Use numpy to operate on arrays. numpy is written in C
from numba import njit

@njit()
def normalize(coords):
    return coords / np.sqrt(np.sum(coords**2))

@njit()
def get_magnetization(spin_grid):
    s_z_ = spin_grid[:,:,2]
    return np.mean(s_z_)

@njit()
def get_timeavg_magnetization(magnetization):
    return np.mean(magnetization)

@njit()
def F_th(alpha, k_b, T, gamma, mu, delta_t):
    rnd_num = make_3D_gaussian_alt()
    return rnd_num*np.sqrt((2*alpha*k_b*T)/(gamma*mu*delta_t))

@njit()
def LLG(S_j, F_j, gamma, alpha):
    prefactor = -gamma/(1+alpha**2)
    factor1 = np.cross(S_j, F_j)
    factor2 = np.cross(alpha*S_j, factor1)
    return prefactor*(factor1 + factor2)

@njit()
def F_eff(mu, S_j, d_z, e_z, B, J, neighbours):
    return (1/mu) * (J*np.sum(neighbours, axis=0) + 2*d_z*S_j[2]*e_z + mu*B)

@njit()
def F(mu, S_j, d_z, e_z, B, J, neighbours, alpha, k_b, T, gamma, delta_t):
    return F_eff(mu, S_j, d_z, e_z, B, J, neighbours) + F_th(alpha, k_b, T, gamma, mu, delta_t)

#@njit(cache=True)
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