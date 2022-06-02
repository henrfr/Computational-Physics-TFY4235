from numba import jit, njit
import numba
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from time import time

@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean) 
    for i in numba.prange(0, len(D)):   #<-- Fixed here, must start from zero
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D  

@jit(nopython=True)
def ray_tracing_numpy_numba(points,poly):
    x,y = points[:,0], points[:,1]
    n = len(poly)
    inside = np.zeros(len(x),np.bool_)
    p2x = 0.0
    p2y = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        idx = np.nonzero((y > min(p1y,p2y)) & (y <= max(p1y,p2y)) & (x <= max(p1x,p2x)))[0]
        if len(idx):    # <-- Fixed here. If idx is null skip comparisons below.
            if p1y != p2y:
                xints = (y[idx]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
            if p1x == p2x:
                inside[idx] = ~inside[idx]
            else:
                idxx = idx[x[idx] <= xints]
                inside[idxx] = ~inside[idxx]    

        p1x,p1y = p2x,p2y
    return inside 

@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections & 1  


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon,points[i])
    return D  

@jit(nopython=True)
def is_inside_postgis(polygon, point):
    length = len(polygon)
    intersections = 0

    dx2 = point[0] - polygon[0][0]
    dy2 = point[1] - polygon[0][1]
    ii = 0
    jj = 1

    while jj<length:
        dx  = dx2
        dy  = dy2
        dx2 = point[0] - polygon[jj][0]
        dy2 = point[1] - polygon[jj][1]

        F =(dx-dx2)*dy - dx*(dy-dy2)
        if 0.0==F and dx*dx2<=0 and dy*dy2<=0:
            return 2

        if (dy>=0 and dy2<0) or (dy2>=0 and dy<0):
            if F > 0:
                intersections += 1
            elif F < 0:
                intersections -= 1

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections != 0  


@njit(parallel=True)
def is_inside_postgis_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_postgis(polygon,points[i])
    return D

np.random.seed(2)

time_parallelpointinpolygon=[]
time_mpltPath=[]
time_ray_tracing_numpy_numba=[]
time_is_inside_sm_parallel=[]
time_is_inside_postgis_parallel=[]
n_points=[]

data = np.load("boundary_data/generation_4_seg_1.npy")
polygon = np.transpose(data)
X_MAX = np.amax(data[0])
for i in range(1, 50002, 10000): 
    n_points.append(i)
    N = i
    points = np.random.uniform(0, X_MAX, size=(N, 2))
    
    
    #Method 1
    start_time = time()
    inside1=parallelpointinpolygon(points, polygon)
    time_parallelpointinpolygon.append(time()-start_time)

    # Method 2
    # start_time = time()
    # path = mpltPath.Path(polygon,closed=True)
    # inside2 = path.contains_points(points)
    # time_mpltPath.append(time()-start_time)

    # Method 3
    # start_time = time()
    # inside3=ray_tracing_numpy_numba(points,polygon)
    # time_ray_tracing_numpy_numba.append(time()-start_time)

    # Method 4
    start_time = time()
    inside4=is_inside_sm_parallel(points,polygon)
    time_is_inside_sm_parallel.append(time()-start_time)

    # Method 5
    start_time = time()
    inside5=is_inside_postgis_parallel(points,polygon)
    time_is_inside_postgis_parallel.append(time()-start_time)

    print(1)


plt.style.use('seaborn-bright')
plt.plot(n_points,time_parallelpointinpolygon,label='parallelpointinpolygon')
# plt.plot(n_points,time_mpltPath,label='mpltPath')
# plt.plot(n_points,time_ray_tracing_numpy_numba,label='ray_tracing_numpy_numba')
plt.plot(n_points,time_is_inside_sm_parallel,label='is_inside_sm_parallel')
plt.plot(n_points,time_is_inside_postgis_parallel,label='is_inside_postgis_parallel')
plt.xlabel("N points")
plt.ylabel("time (sec)")
plt.legend(loc = 'best')
plt.show()
