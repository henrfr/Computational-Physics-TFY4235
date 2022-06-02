import matplotlib as plt
from solve import *

def print_keys():
    solutions = load_solutions()
    print(solutions.keys())

def print_eigenvalues(l, pts, k):
    solutions = load_solutions()
    evals = np.sqrt(solutions[f"{l}_{pts}_{k}"]["evals"])
    evals = evals[:10]
    omega_0 = np.sqrt(2)*np.pi/((pts+1)*4**l)
    print(omega_0)
    evals = evals/omega_0
    print(evals)

def plot_eigenvectors(l, pts, k):
    solutions = load_solutions()
    data = np.load(f"../boundary_data/generation_{l}_pts_{pts}_compact.npy")
    evecs = solutions[f"{l}_{pts}_{k}"]["evecs"]
    nrows = 5
    ncols = 2
    MAX_X = evecs[0].shape[0]
    fig = plt.figure(figsize=(6,15))
    plt.style.use('seaborn-bright')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    for i in range(nrows):
        for j in range(ncols):
            x = np.arange(0,MAX_X,1)
            x,y =np.meshgrid(x,x)
            ax = fig.add_subplot(nrows, ncols, i*ncols+j+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.contourf(x,y,evecs[i*ncols+j], cmap="viridis",zorder=2)
            ax.plot(data[0],data[1], c='red', zorder=3, alpha=0.8)
    plt.tight_layout()
    #plt.savefig(f"eigenmodes_{l}_{pts}_{k}_new.png", dpi=300)
    plt.show()

def plot_degeneracy(l, pts, k):
    solutions = load_solutions()
    data = np.load(f"../boundary_data/generation_{l}_pts_{pts}_compact.npy")
    evecs = solutions[f"{l}_{pts}_{k}"]["evecs"]
    superpos = evecs[1] + evecs[2]  - evecs[3]
    MAX_X = evecs[0].shape[0]
    x = np.arange(0,MAX_X,1)
    x,y =np.meshgrid(x,x)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.style.use('seaborn-bright')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x,y, superpos, cmap='viridis', zorder=2)
    ax.plot(data[0],data[1], c='red', zorder=2, alpha=0.8)
    plt.show()

#def make_lin_comb(l, pts, k):

plot_degeneracy(4,3,10)
#plot_eigenvectors(4,3,10)
#print_eigenvalues(4,3,10)
#print_keys()
