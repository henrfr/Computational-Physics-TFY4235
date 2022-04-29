import numpy as np
from equations import LLG, normalize, F
from numba import njit, prange
# Heun's Method

@njit(cache=True)
def heun(delta_t, S_j, F_j, gamma, alpha):
    # x is coordinates (as a vector)
    # h is timestep
    # f(x) is a function that returns the derivative
    # "Slopes"
    # Vurdere om du må sende inn nye F_j verdier når du skal korrigere prediksjonen.
    k1  = LLG(S_j,                F_j, gamma, alpha)
    k2  = LLG(S_j + k1*delta_t,   F_j, gamma, alpha)
    # Update time and position
    S_j_ = S_j + delta_t*(k1 + k2)/2
    return normalize(S_j_)


# def test():
#     a = np.zeros((4,3))
#     a[0] = [1,2,3]
#     a[1] = [2,1,0]
#     print(a)
#     print(np.sum(a, axis=0))
# test()

def evolve_spins_pbc(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    TODO: implement periodic boundary conditions
    """
    # For each time step
    for i in range(N_steps-1):
        # For every value in row
        for row_val in range(1,shape[0]+1):
            # For every value in column
            for col_val in range(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.array([data[i][row_val][col_val+1],  # Down
                                       data[i][row_val][col_val-1],  # Up
                                       data[i][row_val+1][col_val],  # Right
                                       data[i][row_val-1][col_val]]) # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)

                # Perform a step with Heun
                data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
    return data




@njit(cache=True)
def evolve_spins(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for non-periodic boundary conditions
    """
    # For each time step
    for i in range(N_steps-1):
        # For every value in row
        F_j_ = np.zeros((shape[0], shape[1], 3))
        F_j_pred = np.zeros((shape[0], shape[1], 3))
        S_j_pred = np.zeros((shape[0], shape[1], 3))
        #update_S(data, i, shape, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, delta_t, F_j_, S_j_pred)
        for row_val in range(1,shape[0]+1):
            # For every value in column
            for col_val in range(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((4,3))
                neighbours[0] = data[i][row_val][col_val+1] # Down
                neighbours[1] = data[i][row_val][col_val-1] # Up
                neighbours[2] = data[i][row_val+1][col_val] # Right
                neighbours[3] = data[i][row_val-1][col_val] # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)
                F_j_[row_val][col_val] = F_j

                S_j_pred[row_val][col_val] = S_j + delta_t * LLG(S_j,                F_j, gamma, alpha)
        #update_F(shape, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, delta_t, F_j_pred, S_j_pred)
        for row_val in range(1,shape[0]+1):
            # For every value in column
            for col_val in range(1,shape[1]+1):
                # Find current spin values
                S_j_predicted = S_j_pred[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((4,3))
                neighbours[0] = S_j_pred[row_val][col_val+1] # Down
                neighbours[1] = S_j_pred[row_val][col_val-1] # Up
                neighbours[2] = S_j_pred[row_val+1][col_val] # Right
                neighbours[3] = S_j_pred[row_val-1][col_val] # Left

                # Find the effective field at current position
                F_j_pred[row_val][col_val] = F(mu, S_j_predicted,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)
        #quasi_heun(data, i, shape, gamma, alpha, delta_t, F_j_, F_j_pred, S_j_pred)
        for row_val in range(1,shape[0]+1):
            # For every value in column
            for col_val in range(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                F_j = F_j_[row_val][col_val]
                F_j_predicted = F_j_pred[row_val][col_val] 
                S_j_predicted = S_j_pred[row_val][col_val]

                data[i+1][row_val][col_val] = S_j + (delta_t/2)*(LLG(S_j, F_j, gamma, alpha) + LLG(S_j_predicted,
                F_j_predicted, gamma, alpha))
    return data

@njit(cache=True)
def update_S(data, i, shape, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, delta_t, F_j_, S_j_pred):
    for row_val in range(1,shape[0]+1):
        # For every value in column
        for col_val in range(1,shape[1]+1):
            # Find current spin values
            S_j = data[i][row_val][col_val]

            # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
            neighbours = np.zeros((4,3))
            neighbours[0] = data[i][row_val][col_val+1] # Down
            neighbours[1] = data[i][row_val][col_val-1] # Up
            neighbours[2] = data[i][row_val+1][col_val] # Right
            neighbours[3] = data[i][row_val-1][col_val] # Left

            # Find the effective field at current position
            F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)
            F_j_[row_val][col_val] = F_j

            S_j_pred[row_val][col_val] = S_j + delta_t * LLG(S_j,                F_j, gamma, alpha)

@njit(cache=True)
def update_F(shape, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, delta_t, F_j_pred, S_j_pred):
    for row_val in range(1,shape[0]+1):
        # For every value in column
        for col_val in range(1,shape[1]+1):
            # Find current spin values
            S_j_predicted = S_j_pred[row_val][col_val]

            # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
            neighbours = np.zeros((4,3))
            neighbours[0] = S_j_pred[row_val][col_val+1] # Down
            neighbours[1] = S_j_pred[row_val][col_val-1] # Up
            neighbours[2] = S_j_pred[row_val+1][col_val] # Right
            neighbours[3] = S_j_pred[row_val-1][col_val] # Left

            # Find the effective field at current position
            F_j_pred[row_val][col_val] = F(mu, S_j_predicted,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)
@njit(cache=True)            
def quasi_heun(data, i, shape, gamma, alpha, delta_t, F_j_, F_j_pred, S_j_pred):
    for row_val in range(1,shape[0]+1):
        # For every value in column
        for col_val in range(1,shape[1]+1):
            # Find current spin values
            S_j = data[i][row_val][col_val]
            F_j = F_j_[row_val][col_val]
            F_j_predicted = F_j_pred[row_val][col_val] 
            S_j_predicted = S_j_pred[row_val][col_val]

            data[i+1][row_val][col_val] = S_j + (delta_t/2)*(LLG(S_j, F_j, gamma, alpha) + LLG(S_j_predicted,
            F_j_predicted, gamma, alpha))

@njit(cache=True)
def evolve_spins_old_but_working(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for non-periodic boundary conditions. It does not update the F_j for the prediction.
    """
    # For each time step
    for i in range(N_steps-1):
        # For every value in row
        for row_val in range(1,shape[0]+1):
            # For every value in column
            for col_val in range(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((4,3))
                neighbours[0] = data[i][row_val][col_val+1] # Down
                neighbours[1] = data[i][row_val][col_val-1] # Up
                neighbours[2] = data[i][row_val+1][col_val] # Right
                neighbours[3] = data[i][row_val-1][col_val] # Left

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                # neighbours = np.array([data[i][row_val][col_val+1],  # Down
                #                        data[i][row_val][col_val-1],  # Up
                #                        data[i][row_val+1][col_val],  # Right
                #                        data[i][row_val-1][col_val]]) # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)

                # Perform a step with Heun
                data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
    return data

@njit(cache=True)
def evolve_spins_old_but_working_pbc_linear(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for non-periodic boundary conditions. It does not update the F_j for the prediction.
    """
    print(shape[0])
    print(shape[1])
    print(1%1)
    # For each time step
    for i in range(N_steps-1):
        # For every value in row
        for row_val in range(shape[0]):
            # For every value in column
            for col_val in range(shape[1]):
                # Find current spin values
                S_j = data[i][row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((2,3))
                row_val_end = row_val % shape[0]

                neighbours[0] = data[i][row_val_end+1][col_val] # Right
                neighbours[1] = data[i][row_val-1][col_val] # Left
                if i in [0,1] and row_val in [0,1]:
                    print(data[i][row_val][1])
                    print(neighbours)

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,alpha,k_b,T,gamma,delta_t)

                # Perform a step with Heun
                data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
    return data