import numpy as np
import matplotlib.path as mplPath
import cmath
from shapely.geometry import Point, Polygon
from numba import jit, njit
import numba

""" This files contains multiple methods for determining whether a point is inside a closed curve or not.
Depending on the method, it might be necessary to store a list of boundary points in order to correctly
classify them. 
"""

def mpl(boundary: np.array, point: tuple, radius: float = 1e-9) -> int:
    """ Computes whether a point is inside a closed Path or not.
    
    Parameters
    ----------------------------------------------
    boundary : np.array
        Must be closed and on the form "[[x,y] [x,y] [x,y] ... ]
    point : tuple
        The point of interest. Can be (x,y) or [x,y]
    radius : float
        radius > 0 will exclude the boundary and radius < 0 will include it.
        radius == 0 will yield unstable results.
    
    Returns
    ----------------------------------------------
    int : 0 means outside curve, 1 means inside curve. 2 is on the boundary
    """
    # if any(np.equal(boundary,point).all(1)): # Can be used to classify the boundary in together with is_on_segment
    #     return 2
    poly_path = mplPath.Path(boundary, closed=True) # Makes polygon
    return poly_path.contains_point(point, radius = radius)

def shapely(data: np.array, point: tuple) -> int:
    """ Computes whether a shapely Point is inside a Polygon or not.
    
    Parameters
    ----------------------------------------------
    data : np.array
        Must be closed and on the form "[[x1,x2,x3,x4],
                                         [y1,y2,y3,y4]]
    point : tuple
        The point of interest. Can be (x,y) or [x,y]
    
    Returns
    ----------------------------------------------
    int : 0 means outside curve, 1 means inside curve. 2 is on the boundary
    """
    # if any(np.equal(np.transpose(data),point).all(1)): # This can help classify the boundary
    #     return 2
    boundary = zip(data[0], data[1]) # Stores the boundary as [(x,y) (x,y) ...]
    poly = Polygon([*boundary])
    return poly.contains(Point(tuple(point)))

def is_on_segment(P, P0,P1):
    """Helper function to determine whether a point is on a segment"""
    p0 = P0[0]- P[0], P0[1]- P[1]
    p1 = P1[0]- P[0], P1[1]- P[1]
    det = (p0[0]*p1[1] - p1[0]*p0[1])
    prod = (p0[0]*p1[0] + p0[1]*p1[1])
    # Returns True if they are paralell and the dot product is less than 0,
    # meaning  that it is between the points. Also returns True if the point coincides
    # with the borders
    return (det == 0 and prod < 0) or (p0[0] == 0 and p0[1] == 0) or (p1[0] == 0 and p1[1] == 0)
def complex_is_inside(P: tuple, polygon: np.array, validBorder=False) -> int:
    """ Computes whether a point is inside a polygon or not using a complex formula.
    
    Parameters
    ----------------------------------------------
    polygon : np.array
        Must be closed and on the form "[[x1,y1], [x2, y2], ... ]
    P : tuple
        The point of interest. Can be (x,y) or [x,y]
    
    Returns
    ----------------------------------------------
    int : 0 means outside curve, larger than zero (2*pi*i mathematically) means inside, 2 is on boundary
    """

    sum_ = complex(0,0)

    for i in range(1, len(polygon) + 1):
        v0, v1 = polygon[i-1] , polygon[i%len(polygon)]
        if is_on_segment(P,v0,v1): # Is taken care of outside function
            return validBorder
        sum_ += cmath.log( (complex(*v1) - complex(*P)) / (complex(*v0) - complex(*P)) )
    return abs(sum_) > 1 # Checks wheter the sum is 2*pi. 1 is chosen for numerical precision.

@njit(cache=True)
def is_inside_ray(point: tuple, polygon: list) -> int:
  length = len(polygon)-1
  dy2 = point[1] - polygon[0][1] # Current difference in height
  intersections = 0
  i = 0 # Current index
  j = 1 # Next index

  while i<length:
    dy  = dy2
    dy2 = point[1] - polygon[j][1]

    # consider only lines which are not completely above/bellow/right from the point
    if dy*dy2 <= 0.0 and (point[0] >= polygon[i][0] or point[0] >= polygon[j][0]):
        
      # non-horizontal line
      if dy<0 or dy2<0:
        F = dy*(polygon[j][0] - polygon[i][0])/(dy-dy2) + polygon[i][0]

        if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
          intersections += 1
        elif point[0] == F: # point on line
          return 0 # Border point, want 0 at the moment

      # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
      elif dy2==0 and (point[0]==polygon[j][0] or (dy==0 and (point[0]-polygon[i][0])*(point[0]-polygon[j][0])<=0)):
        return 0 # Border point, want 0 at the moment

      # there is another posibility: (dy=0 and dy2>0) or (dy>0 and dy2=0). It is skipped 
      # deliberately to prevent break-points intersections to be counted twice.
    
    i = j
    j += 1
            
  return intersections & 1  # This tests if number of intersections is even or odd. & is bitwise AND, meaning that odd & 1 is 1 and even & 1 is 0

@njit(parallel=True)
def is_inside_ray_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_ray(points[i],polygon)
    return D   