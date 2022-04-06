import numpy as np
import matplotlib.pyplot as plt

def set_plot_parameters(size: int=16) -> None:
    """
    A helper function for setting plot parameters.
    """
    plt.style.use('seaborn-bright')
    plt.rcParams['mathtext.fontset'] = 'cm'
    font = {'family' : 'serif', 
            'size': size}
    plt.rc('font', **font)
    plt.rc('lines', lw=2)

def middle_square(seed: int, N: int) -> np.ndarray:
    """
    This method uses the middle square method to generate random numbers.
    The seed must be 4 digits long. It will inevitably repeat itself after
    10**4 + 1 numbers (probably quicker). Never use it!
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
        rnd[i] = int(str(int((prev**2))).zfill(8)[2:6]) # Keeps the center four digits
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

def test_middle_square():
    """
    Tests two seeds for the middle square method and plots them.
    Should return a plot where the sequence dies off and a plot
    where the sequence repeats.
    """

    # Initialize parameters
    seed = 1234 # 1234 dies off due to 4003*2 = 16024009 -> 240**2 and the problems begin
    N_grid = 10
    N = N_grid**2

    # Generate, normalize and reshape in a grid
    rnd = middle_square(seed,N)
    rnd = rnd/np.amax(rnd)
    rnd = rnd.reshape(N_grid,N_grid)

    # Plot
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].matshow(rnd)
    axs[0].set_title(f"Seed: {seed}")

    # Repeat
    seed = 1111
    N_grid = 15
    N = N_grid**2

    rnd = middle_square(seed,N)
    rnd = rnd/np.amax(rnd)
    rnd = rnd.reshape(N_grid,N_grid)

    axs[1].matshow(rnd)
    axs[1].set_title(f"Seed: {seed}")
    fig.tight_layout()
    fig.savefig("../plots/middle_square.png", dpi=300)
    plt.show()
# set_plot_parameters()
# test_middle_square()

def test_linear_congruential_generator():
    """
    Tests two sets of parameters for the linear congruential method.
    It should return one 2D plot of seemingly random noise, one ordered
    3D plot and one seemingly unordered 3D plot.
    """
    # RANDU parameters
    modulus = 2**31
    a = 65539
    c = 0
    seed = 571

    # Plots the sequence as a matrix
    N_grid = 50
    N = N_grid**2
    rnd = linear_congruential_generator(modulus, a, c, seed, N)
    rnd = rnd/np.amax(rnd)
    rnd = rnd.reshape(N_grid,N_grid)
    plt.matshow(rnd) # 2D
    plt.title(f"{N_grid**2} numbers (RANDU)")
    plt.tight_layout()
    plt.savefig("../plots/RANDU_2D.png", dpi=300)
    plt.show()

    # Plots the sequence in 3D
    N_grid = 50
    N = N_grid**2
    rnd = linear_congruential_generator(modulus, a, c, seed, N)
    rnd = rnd/np.amax(rnd)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rnd[:-2],rnd[1:-1], rnd[2:], marker='o')
    plt.title(f"{N_grid**2} numbers (RANDU)")
    plt.tight_layout()
    plt.savefig("../plots/RANDU_3D.png", dpi=300)
    plt.show()

    # Numerical recipes parameters
    modulus = 2**32
    a = 1664525
    c = 1013904223
    seed = 571

    # Plots the sequence in 3D
    N_grid = 100
    N = N_grid**2
    rnd = linear_congruential_generator(modulus, a, c, seed, N)
    rnd = rnd/np.amax(rnd)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rnd[:-2],rnd[1:-1], rnd[2:], marker='o')
    plt.title(f"{N_grid**2} numbers (Numerical Recipes)")
    plt.tight_layout()
    plt.savefig("../plots/NR_3D.png", dpi=300)
    plt.show()

if __name__ == "__main__":    
    set_plot_parameters()
    test_middle_square()
    test_linear_congruential_generator()