import numpy as np

def middle_square(seed: int, N: int) -> np.ndarray:
    """
    This method uses the middle square method to generate random numbers.
    The seed must be 4 digits long. It will inevitably repeat itself after
    8**4 numbers (probably quicker). Never use it!
    Parameters
    ----------------------------------------------
    seed : int
        A 4-digit number to start of the sequence.
    N : int
        The number of random numbers generated
    
    Returns
    ----------------------------------------------
    np.ndarray : An array with N random numbers
    """
    rnd = np.empty(N) # Initialize array of size N
    rnd[0] = seed
    for i in range(1,N):
        prev = rnd[i-1]

        # Fills the front digits if necessary and keeps the center four digits
        rnd[i] = int(str(int((prev**2))).zfill(8)[2:6])
    return rnd

def linear_congruential_generator(modulus: int, a: int, c: int, seed: int, N: int) -> np.ndarray:
    """
    This method linear congruential generator to generate random numbers.
    It can perform alright for good parameters, but really bad for others.
    It is still not a very random method.
    Parameters
    ----------------------------------------------
    seed : int
        A number to start of the sequence
    modulus, a, c : int
        Mathematical constants used in the algorithm
    N : int
        The number of random numbers generated
    
    Returns
    ----------------------------------------------
    np.ndarray : An array with N random numbers
    """
    rnd = np.empty(N)
    rnd[0] = seed
    for i in range(1,N):
        rnd[i] = (a*rnd[i-1]+c) % modulus
    return rnd