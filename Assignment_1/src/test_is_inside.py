import is_inside
import numpy as np
import matplotlib.pyplot as plt
import sys
import time as time
from numba import jit, njit
import numba

def test_and_plot_mpl(data: np.array, l: int, n: int) -> None:
    start = time.time()
    MAX_X = np.amax(data[0]) + 1
    MAX_Y = np.amax(data[1]) + 1
    x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_Y,1)) # Makes a meshgrid from (0,0) to (X_MAX, Y_MAX)
    points = list(zip(x.flatten(),y.flatten()))
    polygon = np.transpose(data) # Formats the data to [1:N]
    inside = []
    for i in range(len(points)):
      inside.append(is_inside.mpl(polygon, points[i]))
    inside = np.array(inside)
    fig, ax=plt.subplots()
    patch = plt.Polygon(polygon, zorder=0, fill=False, lw=2) # Plots the boundary
    ax.add_patch(patch)
    ax.scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    plt.title(f'Check inside using mpl for l = {l} and n = {n}. Time: {time.time() - start} seconds')
    plt.show()

def test_and_plot_shapely(data: np.array, l: int , n: int) -> None:
    start = time.time()
    MAX_X = np.amax(data[0]) + 1
    MAX_Y = np.amax(data[1]) + 1
    x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_Y,1)) # Makes a meshgrid from (0,0) to (X_MAX, Y_MAX)
    points = list(zip(x.flatten(),y.flatten()))
    inside = []
    for i in range(len(points)):
      inside.append(is_inside.shapely(data, points[i]))
    inside = np.array(inside)
    fig, ax=plt.subplots()
    polygon = np.transpose(data)
    patch = plt.Polygon(polygon, zorder=0, fill=False, lw=2) # Plots the boundary
    ax.add_patch(patch)
    ax.scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    plt.title(f'Check inside using shapely for l = {l} and n = {n}. Time: {time.time() - start} seconds')
    plt.show()

def test_and_plot_complex(data: np.array, l: int , n: int) -> None:
    start = time.time()
    MAX_X = np.amax(data[0]) + 1
    MAX_Y = np.amax(data[1]) + 1
    x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_Y,1)) # Lager meshgrid fra (0,0) til (X_MAX, Y_MAX)
    points = list(zip(x.flatten(),y.flatten()))   
    polygon = np.transpose(data)
    inside = np.empty(len(points))
    for i in range(len(points)):
        inside[i] = is_inside.complex_is_inside(points[i],polygon)
    fig, ax=plt.subplots()
    plt.style.use('seaborn-bright')
    plt.title(f'Check inside using complex formula for l = {l} and n = {n}. Time: {time.time() - start} seconds')
    patch = plt.Polygon(polygon, zorder=0, fill=False, lw=2)
    ax.add_patch(patch)
    ax.scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    plt.show()

def test_and_plot_ray(data: np.array, l: int , n: int) -> None:
    start = time.time()
    MAX_X = np.amax(data[0]) + 1
    MAX_Y = np.amax(data[1]) + 1
    x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_Y,1)) # Lager meshgrid fra (0,0) til (X_MAX, Y_MAX)
    points = list(zip(x.flatten(),y.flatten()))   
    polygon = np.transpose(data)
    inside = np.empty(len(points))
    for i in range(len(points)):
        inside[i] = is_inside.is_inside_ray(points[i],polygon)
    fig, ax=plt.subplots()
    plt.style.use('seaborn-bright')
    plt.title(f'Check inside using sm for l = {l} and n = {n}. Time: {time.time() - start} seconds')
    patch = plt.Polygon(polygon, zorder=0, fill=False, lw=2)
    ax.add_patch(patch)
    ax.scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    plt.show()

def test_and_plot_ray_p(data: np.array, l: int , n: int) -> None:
    start = time.time()
    MAX_X = np.amax(data[0]) + 1
    MAX_Y = np.amax(data[1]) + 1
    x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_Y,1)) # Lager meshgrid fra (0,0) til (X_MAX, Y_MAX)
    points = list(zip(x.flatten(),y.flatten()))   
    polygon = np.transpose(data)
    inside = is_inside.is_inside_ray_parallel(points, polygon)
    fig, ax=plt.subplots()
    plt.style.use('seaborn-bright')
    plt.title(f'Check inside using smp for l = {l} and n = {n}. Time: {time.time() - start} seconds')
    patch = plt.Polygon(polygon, zorder=0, fill=False, lw=2)
    ax.add_patch(patch)
    ax.scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    plt.show()

def compare():
    font = {'family' : 'STIXGeneral',
        'size'   : 18}
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.rc('font', **font)
    plt.style.use('seaborn-bright')
    data = np.load(f"../boundary_data/generation_{1}_pts_{2}_compact.npy")
    MAX_X = np.amax(data[0]) + 1
    MAX_Y = np.amax(data[1]) + 1
    x,y = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_Y,1)) # Lager meshgrid fra (0,0) til (X_MAX, Y_MAX)
    points = list(zip(x.flatten(),y.flatten()))   
    polygon = np.transpose(data)
    inside = np.empty(len(points))
    for i in range(len(points)):
        inside[i] = is_inside.is_inside_ray(points[i],polygon)
    inside2 = []
    for i in range(len(points)):
      inside2.append(is_inside.mpl(polygon, points[i], radius=0))
    inside2 = np.array(inside2)
    data = np.load(f"../boundary_data/generation_{3}_pts_{3}_compact.npy")
    MAX_X = np.amax(data[0]) + 1
    MAX_Y = np.amax(data[1]) + 1
    x1,y1 = np.meshgrid(np.arange(0,MAX_X,1),np.arange(0,MAX_Y,1)) # Lager meshgrid fra (0,0) til (X_MAX, Y_MAX)
    points = list(zip(x1.flatten(),y1.flatten()))   
    polygon2 = np.transpose(data)
    inside3 = np.empty(len(points))
    for i in range(len(points)):
        inside3[i] = is_inside.is_inside_ray(points[i],polygon2)
    fig, axs=plt.subplots(1,3,figsize=(12,4))
    patch = plt.Polygon(polygon, zorder=0, fill=False, lw=4)
    patch2 = plt.Polygon(polygon, zorder=0, fill=False, lw=4)
    patch3 = plt.Polygon(polygon2, zorder=0, fill=False, lw=4)

    axs[0].add_patch(patch)
    axs[0].scatter(x.flatten(),y.flatten(), c=inside2.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    axs[0].set_title("Matplotlib, (l,n)=(1,1)")
    axs[1].add_patch(patch2)
    axs[1].scatter(x.flatten(),y.flatten(), c=inside.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    axs[1].set_title("Ray casting, (l,n)=(1,1)")
    axs[2].add_patch(patch3)
    axs[2].scatter(x1.flatten(),y1.flatten(), c=inside3.astype(float),cmap="RdYlGn", vmin=-.1,vmax=1.2)
    axs[2].set_title("Raycasting, (l,n)=(3,2)")
    fig.tight_layout()
    fig.savefig("Classifications.png", dpi=300)
if __name__ == "__main__":
    l = sys.argv[1]
    n = sys.argv[2]
    data = np.load(f"../boundary_data/generation_{l}_pts_{n}_compact.npy")
    compare()