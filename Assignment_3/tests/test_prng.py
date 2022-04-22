import numpy as np
import matplotlib.pyplot as plt
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.prng import middle_square, linear_congruential_generator
from src.plot_params import set_plot_parameters

def test_middle_square(save=False):
    """
    Tests two seeds for the middle square method and plots them.
    Should return a plot where the sequence dies off and a plot
    where the sequence repeats. Does not save the figure by default.
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
    if save:
        fig.savefig("../plots/middle_square.png", dpi=300)
    plt.show()

def test_linear_congruential_generator(save=False):
    """
    Tests two sets of parameters for the linear congruential method.
    It should return one 2D plot of seemingly random noise, one ordered
    3D plot and one seemingly unordered 3D plot.Does not save the figure 
    by default.
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
    if save:
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
    if save:
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
    if save:
        plt.savefig("../plots/NR_3D.png", dpi=300)
    plt.show()

if __name__ == "__main__":    
    set_plot_parameters()
    test_middle_square()
    test_linear_congruential_generator()