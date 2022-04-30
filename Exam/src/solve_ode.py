import numpy as np
from equations import LLG, normalize, F, random_matrix, make_thermal_constant
from numba import njit, prange

@njit()
def evolve_spins(data: np.ndarray, N_steps: int, delta_t: float, mu: float, d_z: float, e_z: np.ndarray, B: np.ndarray, J: float, alpha: float, k_b: float,T: float, gamma: float, shape: tuple =(1,1)) -> np.ndarray:
    """Finds the time evolution of spins for a 1D lattice with non-periodic boundary conditions.
    For every time step, it initializes matrices to store the predicted S_j and the calculated F_j
    and the predicted F_j. 
    
    1. It then firstly fill the S_j_pred with the prediction, performing the
    first step of the Heun scheme.
    2. Then it finds the new predicted effective field at every location and stores it in F_j_pred
    3. Lastly, it performs the final step in the Heun scheme using the initial and predicted spins and fields.

    This is one step using the Heun scheme. It looks slow, but using numba and parallelisation with pranga
    for steps that do not depend on others, it is quite fast, but sometimes it terminates quickly. If this
    is a problem, use the "old_but_working" functions in the bottom of the file. 

    The data takes the shape: [0,0,0] [0,0,0] [0,0,0], which is what is meant by padding.
                              [0,0,0] [x,y,z] [0,0,0]
                              [0,0,0] [0,0,0] [0,0,0]   
    Args:
        data (np.ndarray): An [N_steps, N_x+2, N_y+2, 3] array with the +2 being padding of 0s
        N_steps (int): The number of steps
        delta_t (float): The size of the timestep
        mu (float): The magnetic moment
        d_z (float): The anisotropy term
        e_z (np.ndarray): The unit vector in the z-direction
        B (np.ndarray): The magnetic field vector [0,0,B_0]
        J (float): The coupling constant
        alpha (float): The damping constant
        k_b (float): The boltzmann constant
        T (float): The temperature
        gamma (float): The gyromagnetic ratio
        shape (tuple, optional): The actual number of spins. Defaults to (1,1).

    Returns:
        np.ndarray: The time evolution of all spins and spin components.
    """

    # Makes the thermal constant once
    thermal_constant = make_thermal_constant(alpha, k_b, T, gamma, mu, delta_t)

    # For each time step
    for i in range(N_steps-1):

        # Allocate memory
        F_j_ = np.zeros((shape[0], shape[1], 3))
        F_j_pred = np.zeros((shape[0], shape[1], 3))
        S_j_pred = np.zeros((shape[0], shape[1], 3))
        rand_mat = random_matrix(shape)

        # Predicts the new spin components at every location.
        # For every value in row
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
                F_j_[row_val][col_val] = F_j

                # This corresponds to Eq. 7a in the exam.
                S_j_pred[row_val][col_val] = normalize(S_j + delta_t * LLG(S_j, F_j, gamma, alpha))
        
        # Finds the predicted effective field at every location.
        for row_val in range(1,shape[0]+1):
            for col_val in range(1,shape[1]+1):
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
        
        # Updates the spin components at every location using current and predicted values
        for row_val in range(1,shape[0]+1):
            for col_val in range(1,shape[1]+1):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                F_j = F_j_[row_val][col_val]
                F_j_predicted = F_j_pred[row_val][col_val] 
                S_j_predicted = S_j_pred[row_val][col_val]

                # This correspond to Eq. 7b in the exam.
                data[i+1][row_val][col_val] = normalize(S_j + (delta_t/2)*(LLG(S_j, F_j, gamma, alpha) + LLG(S_j_predicted,
                F_j_predicted, gamma, alpha)))
    return data


@njit()
def evolve_spins_pbc_linear(data: np.ndarray, N_steps: int, delta_t: float, mu: float, d_z: float, e_z: np.ndarray, B: np.ndarray, J: float, alpha: float, k_b: float,T: float, gamma: float, shape: tuple =(1,1)) -> np.ndarray:
    """Finds the time evolution of spins for a 1D lattice with non-periodic boundary conditions.
    For every time step, it initializes matrices to store the predicted S_j and the calculated F_j
    and the predicted F_j. 
    
    1. It then firstly fill the S_j_pred with the prediction, performing the
    first step of the Heun scheme.
    2. Then it finds the new predicted effective field at every location and stores it in F_j_pred
    3. Lastly, it performs the final step in the Heun scheme using the initial and predicted spins and fields.

    This is one step using the Heun scheme. It looks slow, but using numba and parallelisation with pranga
    for steps that do not depend on others, it is quite fast, but sometimes it terminates quickly. If this
    is a problem, use the "old_but_working" functions in the bottom of the file. 

    The data takes the shape: [0,0,0] [0,0,0] [0,0,0], which is what is meant by padding.
                              [0,0,0] [x,y,z] [0,0,0]
                              [0,0,0] [0,0,0] [0,0,0]   
    Args:
        data (np.ndarray): An [N_steps, N_x+2, N_y+2, 3] array with the +2 being padding of 0s
        N_steps (int): The number of steps
        delta_t (float): The size of the timestep
        mu (float): The magnetic moment
        d_z (float): The anisotropy term
        e_z (np.ndarray): The unit vector in the z-direction
        B (np.ndarray): The magnetic field vector [0,0,B_0]
        J (float): The coupling constant
        alpha (float): The damping constant
        k_b (float): The boltzmann constant
        T (float): The temperature
        gamma (float): The gyromagnetic ratio
        shape (tuple, optional): The actual number of spins. Defaults to (1,1).

    Returns:
        np.ndarray: The time evolution of all spins and spin components.
    """

    # Makes the thermal constant once
    thermal_constant = make_thermal_constant(alpha, k_b, T, gamma, mu, delta_t)

    # For each time step
    for i in range(N_steps-1):
        if i %10000 == 0:
            print(i)
        # Allocate memory
        F_j_ = np.zeros((shape[0], shape[1], 3))
        F_j_pred = np.zeros((shape[0], shape[1], 3))
        S_j_pred = np.zeros((shape[0], shape[1], 3))
        rand_mat = random_matrix(shape)

        # Predicts the new spin components at every location.
        # For every value in row
        for row_val in range(shape[0]):
            # For every value in column
            for col_val in range(shape[1]):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((2,3))
                if row_val % (shape[0] - 1) == 0:
                    neighbours[0] = data[i][0][col_val] # Right
                else:
                    neighbours[0] = data[i][row_val+1][col_val] # Right
                neighbours[1] = data[i][row_val-1][col_val] # Left

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)
                F_j_[row_val][col_val] = F_j

                # This corresponds to Eq. 7a in the exam.
                S_j_pred[row_val][col_val] = normalize(S_j + delta_t * LLG(S_j, F_j, gamma, alpha))
        
        # Finds the predicted effective field at every location.
        for row_val in range(shape[0]):
            for col_val in range(shape[1]):
                # Find current spin values
                S_j_predicted = S_j_pred[row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((2,3))
                if row_val % (shape[0] - 1) == 0:
                    neighbours[0] = data[i][0][col_val] # Right
                else:
                    neighbours[0] = data[i][row_val+1][col_val] # Right
                neighbours[1] = data[i][row_val-1][col_val] # Left

                # Find the predicted effective field at current position
                F_j_pred[row_val][col_val] = F(mu, S_j_predicted,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)
        
        # Updates the spin components at every location using current and predicted values
        for row_val in range(shape[0]):
            for col_val in range(shape[1]):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                F_j = F_j_[row_val][col_val]
                F_j_predicted = F_j_pred[row_val][col_val] 
                S_j_predicted = S_j_pred[row_val][col_val]

                # This correspond to Eq. 7b in the exam.
                data[i+1][row_val][col_val] = normalize(S_j + (delta_t/2)*(LLG(S_j, F_j, gamma, alpha) + LLG(S_j_predicted,
                F_j_predicted, gamma, alpha)))
    return data    








#############################################################################
######## OLD AND A LITTLE WRONG CODE, BUT YIELDS THE SAME RESULTS ###########
#############################################################################

@njit()
def heun(delta_t: float, S_j: np.ndarray, F_j: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """NOTE: This one is not 100% correct, as it do not update the effective field between
    iterations. It yields the same results as the more complex.

    Args:
        delta_t (float): The timestep
        S_j (np.ndarray): The spin components at the current position
        F_j (np.ndarray): The effective field at the current position
        gamma (float): The gyromagnetic ratio
        alpha (float): The damping constant

    Returns:
        np.ndarray: Normalized updated spin components for the iteration.
    """
    k1  = LLG(S_j,                F_j, gamma, alpha) # Predicts new spins
    k2  = LLG(S_j + k1*delta_t,   F_j, gamma, alpha) # Uses LLG with new spins, but NOT new F-field.
    # Update spin components
    S_j_ = S_j + delta_t*(k1 + k2)/2
    return normalize(S_j_)

@njit()
def evolve_spins_old(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
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
def evolve_spins_pbc_linear_old(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for periodic boundary conditions. It does not update the F_j for the prediction.
    """
    thermal_constant = make_thermal_constant(alpha, k_b, T, gamma, mu, delta_t)
    # For each time step
    for i in range(N_steps-1):
        rand_mat = random_matrix(shape)
        # For every value in row
        if i % 10000 == 0:
            print(i)
        for row_val in range(shape[0]):
            # For every value in column
            for col_val in range(shape[1]):
                # Find current spin values
                S_j = data[i][row_val][col_val]
                rand_vec = rand_mat[row_val][col_val]

                # Find the neighbour spin values, since the data is padded, it generalizes to 0D/1D
                neighbours = np.zeros((2,3))
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
def evolve_spins_pbc_square_old(data, N_steps, delta_t, mu, d_z, e_z, B, J, alpha, k_b, T, gamma, shape=(1,1)):
    """
    This one is for non-periodic boundary conditions. It does not update the F_j for the prediction.
    """
    thermal_constant = make_thermal_constant(alpha, k_b, T, mu, gamma, delta_t)
    # For each time step
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

                if row_val % (shape[0] - 1) == 0:
                    neighbours[2] = data[i][0][col_val] # Right
                else:
                    neighbours[2] = data[i][row_val+1][col_val] # Right
                if col_val % (shape[0] - 1) == 0:
                    neighbours[0] = data[i][0][col_val] # Down
                else:
                    neighbours[0] = data[i][row_val][col_val+1] # Down

                neighbours[1] = data[i][row_val][col_val - 1] # Left
                neighbours[3] = data[i][row_val-1][col_val] # Up

                # Find the effective field at current position
                F_j = F(mu, S_j,d_z,e_z,B,J,neighbours,thermal_constant, rand_vec)

                # Perform a step with Heun
                data[i+1][row_val][col_val] = heun(delta_t, S_j, F_j, gamma, alpha)
    return data