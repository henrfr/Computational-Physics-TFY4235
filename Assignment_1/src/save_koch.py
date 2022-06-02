import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    l = -1
    while not l >= 0:
        l = int(input("Generation (0,->): "))
    n = -1
    while not n >= 0:
        n = int(input("Number of points between each corner (0,->): "))
    valid_ans = ["Y", "y", "n", "N"]
    compact = ""
    while compact not in valid_ans:
        compact = input("Do you want to save compact version (if already generated)? (Y/n) ")     
    if compact in ["Y", "y"]:
        for i in range(int(l)+1):
            data = np.load(f"../boundary_data/generation_{i}_pts_{n}_compact.npy")
            plt.style.use('seaborn-bright')
            plt.plot(data[0],data[1])
            plt.savefig(f"../plots/generation_{i}_pts_{n}_compact.jpg", bbox_inches='tight')
            plt.clf()
    else:
        for i in range(int(l)+1):
            data = np.load(f"../boundary_data/generation_{i}_pts_{n}.npy")
            plt.style.use('seaborn-bright')
            plt.plot(data[0],data[1])
            plt.savefig(f"../plots/generation_{i}_pts_{n}.jpg", bbox_inches='tight')
            plt.clf()