import is_inside
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time
from numba import njit
import sys

data = np.load("../boundary_data/generation_3_pts_2_compact.npy")

MAX_X = int(np.amax(data[0])) + 1
x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_X,1)) # Lager meshgrid fra (0,0) til (X_MAX, Y_MAX)
points = np.array(list(zip(x.flatten(),y.flatten())))  
polygon = np.transpose(data)
inside = np.empty(len(points))
# start = time.time()
# # for i in range(len(points)):
# #     inside[i] = is_inside.is_inside_sm(points[i],polygon)
# print(f"Classification done in {time.time() - start} seconds")
# print(f"{np.count_nonzero(inside)} points inside")
# inside = is_inside.is_inside_sm_parallel(points,polygon)
# print(f"Classification done in {time.time() - start} seconds")
# print(f"{np.count_nonzero(inside)} points inside")

# This will make a bool matrix
@njit(cache=True)
def get_inside(points, polygon):
    bool_arr = np.empty(shape=(MAX_X,MAX_X), dtype=np.int8)
    for i in range(len(points)):
        bool_arr[points[i][0]][points[i][1]] = is_inside.is_inside_sm(points[i],polygon)
    return bool_arr
start = time.time()
bool_arr = get_inside(points, polygon)
print(f"Classification done in {time.time() - start} seconds")
def get_laplacian_2():
    n = np.count_nonzero(bool_arr)
    A = sp.lil_matrix((n,n))
    pos = np.argwhere(bool_arr==1)
    for i in range(n):
        A[i,i] = 4
        x,y = pos[i]
        if bool_arr[x][y-1]: # Left
            A[i,i-1] = -1
        if bool_arr[x][y+1]: # Right
            A[i,i+1] = -1
        if bool_arr[x-1][y]: # Up
            left_x = np.count_nonzero(bool_arr[x,:y])  # Counts points to the left of current
            right_x_up = np.count_nonzero(bool_arr[x-1][::-1][:-y]) # Counts points to the right of upper
            A[i,i-(left_x+right_x_up)] = -1
        if bool_arr[x+1][y]: # Down
            left_x_down = np.count_nonzero(bool_arr[x+1,:y])
            right_x = np.count_nonzero(bool_arr[x][::-1][:-y])
            A[i,i+(left_x_down+right_x)] = -1

    # this will make a plot to illustrate the logic behind the laplacian
    fig, ax = plt.subplots(1,2)
    ax[0].spy(A)
    ax[1].plot(data[1],data[0], c='red', zorder=3)
    ax[1].matshow(bool_arr)
    ax[1].scatter(pos[:,1], pos[:,0])
    plt.show()
    return A, pos

def make_e(e, pos, n):
    E = np.zeros((MAX_X, MAX_X))
    temp = e[:,n]
    for i, p in enumerate(pos):
        E[p[0],p[1]] = temp[i]
    return E[::-1]

def solve(U, k=10):
    e, v = spl.eigsh(U, which="SM", k=k)
    return e, v

start = time.time()
A, pos = get_laplacian_2()
print(f"Laplace done in {time.time()-start} seconds")
start = time.time()
eval, evec = solve(A.tobsr())
print(f"Solve done in {time.time() - start} seconds")

plt.style.use('seaborn-bright')
#I = evec[:,0].reshape(MAX_X, MAX_X)
start = time.time()
I2 = make_e(evec, pos, 4)
print(f"Make e took {time.time() - start} seconds")
fig,ax = plt.subplots(subplot_kw={"projection" : "3d"})
x = np.arange(0,MAX_X,1)
x,y =np.meshgrid(x,x)
ax.plot_surface(x,y,I2, cmap="viridis",zorder=2)
ax.plot(data[0],data[1], c='red', zorder=3)
plt.show()