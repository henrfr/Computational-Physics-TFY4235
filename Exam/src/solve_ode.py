import numpy as np
from equations import LLG, normalize, F, random_matrix, make_thermal_constant
from numba import njit, prange

@njit()
def heun(delta_t, S_j, F_j, gamma, alpha):
    k1  = LLG(S_j,                F_j, gamma, alpha)
    k2  = LLG(S_j + k1*delta_t,   F_j, gamma, alpha)
    # Update time and position
    S_j_ = S_j + delta_t*(k1 + k2)/2
    return normalize(S_j_)

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




@njit()
def evolve_spins(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for non-periodic boundary conditions and used for a,b,c and d
    """
    thermal_constant = make_thermal_constant(alpha, k_b, T, gamma, mu, delta_t)

    # For each time step
    for i in range(N_steps-1):
        # For every value in row
        F_j_ = np.zeros((shape[0], shape[1], 3))
        F_j_pred = np.zeros((shape[0], shape[1], 3))
        S_j_pred = np.zeros((shape[0], shape[1], 3))
        rand_mat = random_matrix(shape)
        for row_val in prange(1,shape[0]+1):
            # For every value in column
            for col_val in prange(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((4,3))
                neighbours[0] = data[i][row_val][col_val+1] # Down
                neighbours[1] = data[i][row_val][col_val-1] # Up
                neighbours[2] = data[i][row_val+1][col_val] # Right
                neighbours[3] = data[i][row_val-1][col_val] # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)
                F_j_[row_val][col_val] = F_j

                S_j_pred[row_val][col_val] = S_j + delta_t * LLG(S_j, F_j, gamma, alpha)
        for row_val in prange(1,shape[0]+1):
            # For every value in column
            for col_val in prange(1,shape[1]+1):
                # Find current spin values
                S_j_predicted = S_j_pred[row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]
                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((4,3))
                neighbours[0] = S_j_pred[row_val][col_val+1] # Down
                neighbours[1] = S_j_pred[row_val][col_val-1] # Up
                neighbours[2] = S_j_pred[row_val+1][col_val] # Right
                neighbours[3] = S_j_pred[row_val-1][col_val] # Left

                # Find the predicted effective field at current position
                F_j_pred[row_val][col_val] = F(mu, S_j_predicted,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)
        for row_val in prange(1,shape[0]+1):
            # For every value in column
            for col_val in prange(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                F_j = F_j_[row_val][col_val]
                F_j_predicted = F_j_pred[row_val][col_val] 
                S_j_predicted = S_j_pred[row_val][col_val]

                data[i+1][row_val][col_val] = S_j + (delta_t/2)*(LLG(S_j, F_j, gamma, alpha) + LLG(S_j_predicted,
                F_j_predicted, gamma, alpha))
    return data


#@njit(cache=True)
def evolve_spins_old_but_working(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for non-periodic boundary conditions. It does not update the F_j for the prediction.
    """
    thermal_constant = make_thermal_constant(alpha, k_b, T, gamma, mu, delta_t)
    # For each time step
    for i in range(N_steps-1):
        #print("First time step")
        # For every value in row
        rand_mat = random_matrix(shape)
        for row_val in range(1,shape[0]+1):
            # For every value in column
            for col_val in range(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((4,3))
                neighbours[0] = data[i][row_val][col_val+1] # Down
                neighbours[1] = data[i][row_val][col_val-1] # Up
                neighbours[2] = data[i][row_val+1][col_val] # Right
                neighbours[3] = data[i][row_val-1][col_val] # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)

                # Perform a step with Heun
                data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
    return data

@njit()
def evolve_spins_old_but_working_pbc_linear(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for periodic boundary conditions. It does not update the F_j for the prediction.
    """
    thermal_constant = make_thermal_constant(alpha, k_b, T, gamma, mu, delta_t)
    # For each time step
    for i in range(N_steps-1):
        rand_mat = random_matrix(shape)
        # For every value in row
        if i % 1000 == 0:
            print(i)
        for row_val in range(shape[0]):
            # For every value in column
            for col_val in range(shape[1]):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((2,3))
                #print(row_val)
                #print(shape[0])
                if row_val % (shape[0] - 1) == 0:
                    neighbours[0] = data[i][0][col_val] # Right
                else:
                    neighbours[0] = data[i][row_val+1][col_val] # Right
                neighbours[1] = data[i][row_val-1][col_val] # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)

                # Perform a step with Heun
                data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
    return data

@njit()
def evolve_spins_old_but_working_pbc_square(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for non-periodic boundary conditions. It does not update the F_j for the prediction.
    """
    thermal_constant = make_thermal_constant(alpha, k_b, T, mu, gamma, delta_t)
    # For each time step
    print(shape[0])
    for i in range(N_steps-1):
        rand_mat = random_matrix(shape)
        if i%10000 == 0:
            print(i)
        # For every value in row
        for row_val in range(shape[0]):
            # For every value in column
            for col_val in range(shape[1]):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((4,3))
                row_val_end = row_val % shape[0]
                col_val_end = col_val % shape[0]

                if row_val % (shape[0] - 1) == 0:
                    neighbours[2] = data[i][0][col_val] # Right
                else:
                    neighbours[2] = data[i][row_val+1][col_val] # Right
                if col_val % (shape[0] - 1) == 0:
                    neighbours[0] = data[i][0][col_val] # Down
                else:
                    neighbours[0] = data[i][row_val][col_val+1] # Down

                #neighbours[0] = data[i][row_val_end][col_val_end + 1]  # Down
                neighbours[1] = data[i][row_val][col_val - 1] # Left
                #neighbours[2] = data[i][row_val_end+1][col_val] # Right
                neighbours[3] = data[i][row_val-1][col_val] # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)

                # Perform a step with Heun
                data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
    return data