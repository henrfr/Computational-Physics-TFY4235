import numpy as np
import sys
import time
from numba import njit

""" This script will recursively generate a quadratic Koch curve. It can be called from the terminal using:
    'python make_koch.py'
    - l is the generation
    - n is the number of points between each corner
    - up_to will generate all curves up to and including given generation.
    - compact will not store the points between the corners
    
    The transformation applied to each segment (between corners) is the following: 
                                     ____
                                    |   |
    ________________    ---->   ____|   |     ____
                                        |    |
                                        |____|
    
    Runtimes for compact version. n = 1:
        Generation 0-4: ~0 seconds.
        Generation 5: 0.35 seconds.
        Generation 6: 2.67 seconds.
        Generation 7: 22.76 seconds.

    For speedup: Numba could be used, but it will not accept global, so the code would have to be rewritten.
    .npy is also not the fastest nor the most storage efficient format. Generation 8 and onwards would,
    however, be very memory consuming.

    NB: There is a known bug in both generators. When generating the first generation (l=0), the code will 
    sometimes run properly. This is not a problem, since l = 0 is not of interest and can be constructed
    manually if necessary.

    TODO: get the exact scaling of the empty array
    """

def Koch(l: int, array: np.array, direction: int, n: int) -> None:
    """
    Generates the quadratic Koch curve

    Parameters
    ----------------------------------------------
    l : int
        The generation of the Koch curve
    array : np.array
        The boundary is stored in the array
    direction : int
        Keeps track of the orientation
    n : int
        The number of points between each corner.
    
    Returns
    ----------------------------------------------
    None - The array is modified in place
    """
    global i 
    if l == 0:
        if direction == 0: # Right
            for j in range(n): # Also stores non-corner points
                array[:,i+1] = [array[:,i][0]+1, array[:,i][1]] # Adds one to the previous value in right direction
                i += 1
        elif direction == 1: # Down
            for j in range(n):
                array[:,i+1] = [array[:,i][0], array[:,i][1]-1]
                i += 1
        elif direction == 2: # Left
            for j in range(n):
                array[:,i+1] = [array[:,i][0]-1, array[:,i][1]]
                i += 1
        elif direction == 3: # Up
            for j in range(n):
                array[:,i+1] = [array[:,i][0], array[:,i][1]+1]
                i += 1
        return None
    else:
        Koch(l-1, array, direction, n)
        direction = (direction -1)%4 # The change of direction corresponds to the transformation shape
        Koch(l-1, array, direction, n)
        direction = (direction + 1)%4
        Koch(l-1, array, direction, n)
        direction = (direction + 1)%4
        Koch(l-1, array, direction, n)
        Koch(l-1, array, direction, n)
        direction = (direction -1)%4
        Koch(l-1, array, direction, n)
        direction = (direction -1)%4
        Koch(l-1, array, direction, n)
        direction = (direction + 1)%4
        Koch(l-1, array, direction, n)

def Koch_compact(l: int, array: np.array, direction: int, n: int) -> None:
    """
    Generates the quadratic Koch curve, but do not store points between segments.
    """
    global i
    if l == 0:
        if direction == 0: # Right
            array[:,i+1] = [array[:,i][0]+n, array[:,i][1]] # Adds one to the previous value in right direction
        elif direction == 1: # Down
            array[:,i+1] = [array[:,i][0], array[:,i][1]-n]
        elif direction == 2: # Left
            array[:,i+1] = [array[:,i][0]-n, array[:,i][1]]
        elif direction == 3: # Up
            array[:,i+1] = [array[:,i][0], array[:,i][1]+n]
        i += 1
        return None
    else:
        Koch_compact(l-1, array, direction, n) # _ The first line 
        direction = (direction -1)%4 # The change of direction corresponds to the transformation shape
        Koch_compact(l-1, array, direction, n) # _| The second line
        direction = (direction + 1)%4                           #__
        Koch_compact(l-1, array, direction, n) # _|  The third line, etc.
        direction = (direction + 1)%4
        Koch_compact(l-1, array, direction, n)
        Koch_compact(l-1, array, direction, n)
        direction = (direction -1)%4
        Koch_compact(l-1, array, direction, n)
        direction = (direction -1)%4
        Koch_compact(l-1, array, direction, n)
        direction = (direction + 1)%4
        Koch_compact(l-1, array, direction, n)

def save_file(array, l, n):
    np.save(f"../boundary_data/generation_{l}_pts_{n}.npy", array)
def save_file_compact(array, l, n):
    np.save(f"../boundary_data/generation_{l}_pts_{n}_compact.npy", array)

if __name__ == "__main__":
    l = -1
    while not l >= 0:
        l = int(input("Generation (0,->): "))
    n = -1
    while not n >= 0:
        n = int(input("Number of points between each corner (0,->): "))
    valid_ans = ["Y", "y", "n", "N"]
    up_to = ""
    compact = ""
    while up_to not in valid_ans:
        up_to = input("Do you want to genereate all curves up to given entry? (Y/n) ")
    while compact not in valid_ans:
        compact = input("Do you want to store intermediate points between corners? (y/N) ")
    start = time.time()
    if up_to in ["Y", "y"]: # Makes all boundaries up to specified generation
        if compact in ["N", "n"]:
            for gen in range(l + 1):
                i = 0
                direction = 0
                array = np.empty((2,(8**gen+1)*4)) # Initializes empty array
                array[:,0] = [0,0]
            
                # Main loop
                for j in range(4):
                    direction = j
                    Koch_compact(gen, array, direction, n+1)
                
                # Center the boundary to only greater than or equal to one.    
                min_x = np.min(array[0])
                min_y = np.min(array[1])
                array[0] += abs(min_x)
                array[1] += abs(min_y)

                save_file_compact(array, gen, n)
        else:
            for l in range(int(sys.argv[1]) + 1):
                i = 0
                direction = 0
                array = np.empty((2,(8**l+1)*4*(n+1))) # Initializes empty array
                array[:,0] = [0,0]
            
                # Main loop
                for j in range(4):
                    direction = j
                    Koch(l, array, direction, n+1)
                
                # Center the boundary to only greater than or equal to one.    
                min_x = np.min(array[0])
                min_y = np.min(array[1])
                array[0] += abs(min_x)
                array[1] += abs(min_y)

                save_file(array, l, n)
    else:
        if compact in ["N", "n"]:
            i = 0
            direction = 0
            array = np.empty((2,(8**l)*4+1))
            array[:,0] = [0,0]
            
            # Main loop
            for j in range(4):
                direction = j
                Koch_compact(l, array, direction, n+1)
                
            # Center the boundary to only greater than or equal to one.    
            min_x = np.min(array[0])
            min_y = np.min(array[1])
            array[0] += abs(min_x)
            array[1] += abs(min_y)

            save_file_compact(array, l, n)
        else:    
            i = 0
            direction = 0
            array = np.empty((2,(8**l)*4*(n+1)+1))
            array[:,0] = [0,0]
            
            # Main loop
            for j in range(4):
                direction = j
                Koch(l, array, direction, n+1)
                
            # Center the boundary to only greater than or equal to one.    
            min_x = np.min(array[0])
            min_y = np.min(array[1])
            array[0] += abs(min_x)
            array[1] += abs(min_y)

            save_file(array, l, n)
    print(f"The curve(s) were generated in {time.time() - start} seconds.")