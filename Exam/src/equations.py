import numpy as np # Use numpy to operate on arrays. numpy is written in C
from numba import njit

@njit(cache=True)
def normalize(coords):
    return coords / np.sqrt(np.sum(coords**2))

@njit(cache=True)
def F_th(alpha, k_b, T, gamma, mu, delta_t):
    rnd_num = np.random.normal()
    return rnd_num*np.sqrt((2*alpha*k_b*T)/(gamma*mu*delta_t))

# def hamiltonian_single_spin(coords, d_z, mu, e_z, B):
#     return - d_z*(np.dot(coords, e_z))**2 - mu*np.dot(coords, B)

@njit(cache=True)
def LLG(S_j, F_j, gamma, alpha):
    prefactor = -gamma/(1+alpha**2)
    factor1 = np.cross(S_j, F_j)
    factor2 = np.cross(alpha*S_j, factor1)
    return prefactor*(factor1 + factor2)

@njit(cache=True)
def F_eff(mu, S_j, d_z, e_z, B, J, neighbours):
    return (1/mu) * (J*np.sum(neighbours, axis=0) + 2*d_z*S_j[2]*e_z + mu*B)

@njit(cache=True)
def F(mu, S_j, d_z, e_z, B, J, neighbours, alpha, k_b, T, gamma, delta_t):
    return F_eff(mu, S_j, d_z, e_z, B, J, neighbours) + F_th(alpha, k_b, T, gamma, mu, delta_t)

def test():
    # (x, y, S)
    a = np.zeros((3,3,3))
    a[1][1][2] = 1
    print(a)
    print(a[0][0])
# test()

# print(normalize(np.array([1,1,100])))