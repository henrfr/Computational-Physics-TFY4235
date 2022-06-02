from euler import *
from physics import *
from simulate import *
import time
import numpy as np
import matplotlib.pyplot as plt

"""
This file will predict the mean velocity for a particle with radius 3 times that of particle_1
"""
particle_1 = {
    "r": 12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}
particle_2 = {
    "r": 3*12e-9,
    "L": 20e-6,
    "eta": 1e-3,
    "kbT": 4.1656e-21,
    "delta_U": 1.2817e-17,
    "alpha": 0.2
}

def test_prediction():
    """
    This function will also be left untidy, but runs as is.
    It compared predicted values to simulated values.
    """

    # The original data for particle_1
    data = np.load("../data/mean_velocity_0_5.npy", allow_pickle=True)    
    avg_lst = data[0]
    taus = data[1]

    # Makes prediction based on the origianl data
    particle_2_vel_pred = np.array(avg_lst)/3
    particle_2_taus_pred = np.array(taus)*3

    # Sets initial parameters
    set_plot_parameters(16)
    N_particles = 200
    sim_time = 40
    taus = np.linspace(0.1, 15, 10)
    start = time.time()
    avg_lst = []

    # Finds average velocity for different taus
    for tau in taus:
        positions = simulate_particle_position_flashing_potential_less_memory(particle_2,N_particles,sim_time,0.2,tau, 0.9)
        avg_lst.append(np.mean(positions)*1e6*particle_2["L"]/sim_time) # Calculates the average velocity
    print(f"Sim time is {time.time()-start:.2f} seconds")

    np.save("../data/mean_velocity_0_5_particle2.npy", (avg_lst, taus))

    plt.plot(particle_2_taus_pred, particle_2_vel_pred)
    plt.plot(taus, avg_lst)
    plt.show()

if __name__ == "__main__":
    test_prediction()
