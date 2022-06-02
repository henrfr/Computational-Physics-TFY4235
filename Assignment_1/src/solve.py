import is_inside
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time
from numba import njit
import os
import numba

@njit(parallel=True)
def get_inside_symmetry(points: np.ndarray, polygon: np.ndarray, MAX_X: int) -> np.ndarray:
    """ This functions does exactly the same as get_inside, but runs in parallel. See get_inside()
        for further documentation. It speeds the code up significantly."""
    ln = len(points)
    bool_arr = np.zeros(shape=(MAX_X, MAX_X), dtype=numba.boolean)
    for i in numba.prange(ln):
        bool_arr[points[i][0], points[i][1]] = is_inside.is_inside_ray(points[i],polygon)
    bool_arr_2 = np.rot90(bool_arr)
    bool_arr_3 = np.rot90(bool_arr_2)
    bool_arr_4 = np.rot90(bool_arr_3)
    left = np.concatenate((bool_arr_4[:-1], bool_arr_3), axis=0)
    right = np.concatenate((bool_arr[:-1], bool_arr_2), axis=0)
    res = np.concatenate((right[:,:-1], left), axis=1)
    return res  

@njit(cache=True)
def get_inside(points: np.ndarray, polygon: np.ndarray, MAX_X: int) -> np.ndarray:
    """ Calculates and stores whether a point is inside or not in a boolean matrix
    
    Parameters
    ----------------------------------------------
    points : np.ndarray
        An array containing all possible points on the form [[x1,y1], [x2, y1] ... [x_i, y_j]]
    polygon : np.ndarray 
        An array containing the boundary of the fractal. [[x1,y1], [x2,y2], ... ]
    MAX_X : int
        The number of rows and columns
    
    Returns
    ----------------------------------------------
    bool_arr :  np.ndarray
        The coordinates of a point is represented by the row and column number. Whether it
        is inside or ouside is represented by 0 and 1.
    """
    bool_arr = np.empty(shape=(MAX_X,MAX_X), dtype=np.int8)
    for i in range(len(points)):
        bool_arr[points[i][0]][points[i][1]] = is_inside.is_inside_ray(points[i],polygon)
    return bool_arr

@njit(parallel=True)
def get_inside_parallel(points: np.ndarray, polygon: np.ndarray, MAX_X: int) -> np.ndarray:
    """ This functions does exactly the same as get_inside, but runs in parallel. See get_inside()
        for further documentation. It speeds the code up significantly."""
    ln = len(points)
    bool_arr = np.empty(shape=(MAX_X, MAX_X), dtype=numba.boolean)
    for i in numba.prange(ln):
        bool_arr[points[i][0], points[i][1]] = is_inside.is_inside_ray(points[i],polygon)
    return bool_arr   

def get_laplacian(bool_arr: np.ndarray) -> tuple:
    """ Checks and stores whether a point is inside or not in a boolean matrix and assigns
        corresponding value according to a seconds order finite difference method. 
        The check for the y-points in the stencil is currently done by slicing the 
        boolean matrix and counting non-zero elements. This could probably be implemented
        faster and smarter.
    
    Parameters
    ----------------------------------------------
    bool_arr : np.ndarray
        The coordinates of a point is represented by the row and column number. Whether it
        is inside or ouside is represented by 0 and 1.

    Returns
    ----------------------------------------------
    A :  sp.lil_matrix
        A sparse matrix embodying the finite difference approximation of the laplacian.
        It includes only inside points. Each grid points "is on the diagonal".
    pos : np.ndarray
        An array containing the position of all inside points, calculated from the bool-array
    """
    n = np.count_nonzero(bool_arr)
    A = sp.lil_matrix((n,n))
    pos = np.argwhere(bool_arr==1) # Creates an array of inside points to avoid looping through all points
    for i in range(n):
        A[i,i] = 4
        x,y = pos[i]
        if bool_arr[x][y-1]: # Left. If left point is inside, update the sparse matrix
            A[i,i-1] = -1
        if bool_arr[x][y+1]: # Right
            A[i,i+1] = -1
        if bool_arr[x-1][y]: # Up
            left_x = np.count_nonzero(bool_arr[x,:y])  # Counts points to the left of current point
            right_x_up = np.count_nonzero(bool_arr[x-1][::-1][:-y]) # Counts points to the right of upper point
            A[i,i-(left_x+right_x_up)] = -1
        if bool_arr[x+1][y]: # Down
            left_x_down = np.count_nonzero(bool_arr[x+1,:y])
            right_x = np.count_nonzero(bool_arr[x][::-1][:-y])
            A[i,i+(left_x_down+right_x)] = -1
    return A, pos

def get_laplacian_order_4(bool_arr: np.ndarray) -> tuple:
    """ Checks and stores whether a point is inside or not in a boolean matrix and assigns
        corresponding value according to a fourth order finite difference method. 
        The check for the y-points in the stencil is currently done by slicing the 
        boolean matrix and counting non-zero elements. This could probably be implemented
        faster and smarter.
    
    Parameters
    ----------------------------------------------
    bool_arr : np.ndarray
        The coordinates of a point is represented by the row and column number. Whether it
        is inside or ouside is represented by 0 and 1.

    Returns
    ----------------------------------------------
    A :  sp.lil_matrix
        A sparse matrix embodying the finite difference approximation of the laplacian.
        It includes only inside points. Each grid points "is on the diagonal".
    pos : np.ndarray
        An array containing the position of all inside points, calculated from the bool-array
    """
    n = np.count_nonzero(bool_arr)
    edge = len(bool_arr[0])
    A = sp.lil_matrix((n,n))
    pos = np.argwhere(bool_arr==1) # Creates an array of inside points to avoid looping through all points
    for i in range(n):
        A[i,i] = 60 # Value of x_i, y_i
        x,y = pos[i]
        if y > 1: # Checking that bool_arr[x, y-2] does not map to opposite end of row
            if bool_arr[x, y-2]: # If (x_i-2, y_i) is inside
                A[i, i-2] = 1
        if bool_arr[x][y-1]: # If (x_i-1, y_i) is inside
            A[i,i-1] = -16
        if bool_arr[x][y+1]: # If (x_i+2, y_i) is inside
            A[i,i+1] = -16
        if y < edge-2: # Checking that bool_arr[x,y+2] does not exceed index of matrix
            if bool_arr[x, y+2]:
                A[i,i+2] = 1
        if x > 1:
            if bool_arr[x-2][y]: # If (x_i, y_i-2) is inside
                left_x = np.count_nonzero(bool_arr[x,:y])  # Counts points to the left of current point
                middle_x_up = np.count_nonzero(bool_arr[x-1]) # Counts points inside in middle row
                right_x_2_up = np.count_nonzero(bool_arr[x-2][::-1][:-y]) # Counts points to the right of upper point
                A[i, i - (left_x + middle_x_up + right_x_2_up)] = 1
        if bool_arr[x-1][y]: 
            left_x = np.count_nonzero(bool_arr[x,:y])  # Counts points to the left of current point
            right_x_up = np.count_nonzero(bool_arr[x-1][::-1][:-y]) # Counts points to the right of upper point
            A[i,i-(left_x+right_x_up)] = -16
        if bool_arr[x+1][y]: # If (x_i, y_i+1) is inside
            left_x_down = np.count_nonzero(bool_arr[x+1,:y])
            right_x = np.count_nonzero(bool_arr[x][::-1][:-y])
            A[i,i+(left_x_down+right_x)] = -16
        if x < edge-2:
            if bool_arr[x+2][y]: 
                left_x_2_down = np.count_nonzero(bool_arr[x+2,:y])
                middle_x_down = np.count_nonzero(bool_arr[x+1])
                right_x = np.count_nonzero(bool_arr[x][::-1][:-y])
                A[i, i + (left_x_2_down + middle_x_down + right_x)] = 1
    A=A/12
    return A, pos

def solve(A: sp.bsr_matrix, k=10) -> tuple:
    """ Solves the eigenvalue problem Ax = cx
    
    Parameters
    ----------------------------------------------
    A : sp.bsr_matrix
        A sparse matrix embodying the finite difference approximation of the laplacian
    k : int
        Determines how many eigenvalues to return
    
    Returns
    ----------------------------------------------
    eval :  np.array
            The k smallest eigenvalues of the system, shape = (1,k)
    evec : np.array
           The k smallest eigenvectors of the system, shape = (M, k), where M is size of the matrix
    """
    eval, evec = spl.eigsh(A, which="LM", k=k, sigma=0) # Solving using Shift-Invert method
    return eval, evec

def make_evec(evec: np.ndarray, pos: np.ndarray, n: int, MAX_X: int) -> np.ndarray:
    """ A helper function that constructs an (MAX_X,MAX_X) eigenvector from 
    the (M, k) eigenvector from solve(). All outside or boundary points
    are set to 0.
    
    Parameters
    ----------------------------------------------
    evec : np.ndarray
        Contains the k smallest eigenvectors. Each eigenvector is stored in a column
    pos : np.ndarray
        An array containing the position of all inside points, calculated in the bool-array
    n : int
        Specifies which eigenvector to construct
    MAX_X : int
        The number of rows and columns
    
    Returns
    ----------------------------------------------
    E :  np.array
        The n'th eigenvector of the fractal. shape = (MAX_X, MAX_X)

    """
    E = np.zeros((MAX_X, MAX_X))
    temp = evec[:,n]
    for i, p in enumerate(pos):
        E[p[0],p[1]] = temp[i] # Sets all the inside points to their corresponding value
    return E[::-1] # The mirroring is necessary due to the way the laplacian was set up. 

def shrink(data, x, y, operators):
    """
    A helper function that reduces the amount of points for classification.
    """
    if operators[0] == ">":
        a = data[:,0] > x
    else:
        a = data[:,0] > x
    if operators[1] == ">":
        b = data[:,1] > y
    else:
        b = data[:,1] > y
    return data[a | b]

def get_laplacian_old(data, inside):
    """ The old version of the laplacian. It is slower, yields a bigger matrix and is arguably 
    unstable since the border and outside points are not set to 0. It, however, yields equivalent
    results compared with the new one."""
    MAX_X = int(np.amax(data[0])) + 1
    n = MAX_X**2 # Size of diagonal
    diag = np.ones(n)
    diag0 = 4 * diag
    diagx = -diag[:-1]
    diagy = -diag[:-(MAX_X)]
    A = sp.diags(
        [diagy, diagx, diag0, diagx, diagy], [-MAX_X, -1, 0, 1, MAX_X], format="lil")

    for i in range(MAX_X):
        #print(i/MAX_X)
        for j in range(MAX_X):
            d = i + MAX_X*j # The index of the diagonal
            if not inside[d]: # Was != 1 before making a change
                d1 = (d + 1) % (MAX_X ** 2) # The element to the right of diagonal 
                d2 = d - 1 # The element to the left of diagonal
                d3 = (d + MAX_X) % (MAX_X ** 2) # The element to the far right of diagonal, corresponding to point above/below
                d4 = d - MAX_X # The element to the far left of diagonal

                # If element not inside, set surrounding stencil to zero.
                #A[d,d]=4 Not neccesary at the moment
                A[d,d1]=0
                A[d,d2]=0
                A[d,d3]=0
                A[d,d4]=0
                # Since the point is not inside, it will not contribute to others as well.
                A[d1, d] = 0
                A[d2, d] = 0
                A[d3, d] = 0
                A[d4, d] = 0
    return A

def visualize():
    """
    This makes the visualization of the matrix, as shown in the report
    """
    data = np.load("../boundary_data/generation_1_pts_1_compact.npy")
    MAX_X = int(np.amax(data[0])) + 1
    polygon = np.transpose(data)
    x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_X,1)) # Lager meshgrid fra (0,0) til (MAX_X, MAX_Y)
    points = np.array(list(zip(x.flatten(),y.flatten())))
    bool_arr = get_inside(points, polygon, MAX_X)
    inside = np.empty(len(points))
    for i in range(len(points)):
        inside[i] = is_inside.is_inside_ray(points[i],polygon)
    A, pos = get_laplacian(bool_arr)
    B = get_laplacian_old(data, inside)
    font = {'family' : 'STIXGeneral',
        'size'   : 26}
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.rc('font', **font)
    plt.style.use('seaborn-bright')
    fig, axs=plt.subplots(2,2,figsize=(15,15))
    axs[0][0].spy(B)
    axs[0][0].set_xticks([])
    axs[0][0].set_title("Top-down")
    axs[1][0].spy(A)
    patch = plt.Polygon(polygon, zorder=0, fill=False, lw=4)
    axs[1][1].add_patch(patch)
    axs[1][1].scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    axs[1][0].set_title("Bottom-up")
    axs[1][0].set_xticks([])
    patch1 = plt.Polygon(polygon, zorder=0, fill=False, lw=4)
    axs[0][1].add_patch(patch1)
    axs[0][1].scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    axs[0][1].set_title("Generation: (1,1)")
    fig.tight_layout()
    fig.savefig("Matrix.png", dpi=300)

def main(l,n,k):
    path = f"solutions.npy"

    if os.path.exists(path):
        solutions = np.load(path, allow_pickle=True).item()
    else:
        solutions = {}

    # Sets up the initial data. "reduced" are made to comply with symmetry classification
    start = time.time()
    data = np.load(f"../boundary_data/generation_{l}_pts_{n}_compact.npy")
    MAX_X = int(np.amax(data[0])) + 1 
    polygon = np.transpose(data)
    MAX_X_reduced = int((MAX_X+1)/2)
    x_reduced, y_reduced = np.meshgrid(np.arange(1,MAX_X_reduced,1),np.arange(1,MAX_X_reduced,1)) # Makes meshgrid from (0,0) to (MAX_X_reduced, MAX_Y_reduced)
    points_reduced =  np.array(list(zip(x_reduced.flatten(),y_reduced.flatten())))
    points_reduced = shrink(points_reduced, int(MAX_X/6),int(MAX_X/6),[">",">"])
    print(f"Preamble took {time.time()-start:.2f} seconds")

    # Makes a boolean ndarray, classifying the points 
    start = time.time() 
    bool_arr = get_inside_symmetry(points_reduced, polygon, MAX_X_reduced)
    print(f"Classification 4-fold took {time.time() - start:.2f} seconds")

    # Creates the laplacian
    start = time.time()
    A, pos = get_laplacian(bool_arr)
    sp.save_npz(f"../data/laplacian_{l}_{n}_{k}.npz", A.tobsr())
    #np.save(f"../data/pos_{l}_{n}_{k}.npy", pos)
    print(f"The size of the laplacian is {(A.data.nbytes)/10**6:.2f} MB.")
    print(f"Laplace done in {time.time()-start:.2f} seconds")

    # Solves the eigensystem
    start = time.time()
    eval, evec = solve(A.tobsr(), k=k) # NOTES: bsr-format is much faster than lil-format
    print(f"Solve done in {time.time() - start:.2f} seconds")

    # Recreates eigenvectors for the entire grid
    start = time.time()
    eigvecs = []
    for i in range(10):
        eigvecs.append(make_evec(evec, pos, i, MAX_X))
    print(f"Recreation done in {time.time() - start:.2f} seconds")

    solutions[f"{l}_{n}_{k}"] = {
        "evals": eval,
        "evecs": eigvecs[:10]
    }

    np.save(path, solutions)

def load_solutions():
    sol = np.load("solutions.npy", allow_pickle=True).item()
    return sol

if __name__ == "__main__":
    l = -1
    while not l >= 0:
        l = int(input("Generation (0,->): "))
    n = -1
    while not n >= 0:
        n = int(input("Number of points between each corner (0,->): "))
    k = -1
    while not k >= 0:
        k = int(input("Number of smallest eigenvalues (1,->): "))
        
    main(l,n,k)
