import numpy as np
import pkfire.ptl_lib.particle as ptllib

    
def addK(grid, k, amp, x0, y0):
    xs = np.linspace(0, grid.box, grid.getShape()[0])
    X, Y, Z = np.meshgrid(xs, xs, xs)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    grid.grid += amp * np.sin(k * R - x0) + y0
    return

def random(grid, nptls, mass_range = [6, 13], 
        constructor = None, const_args = (), const_kwargs = {}):
    dim = len(grid.getShape())
    if constructor is None:
        constructor = ptllib.Particle
    mr = mass_range
    pos = np.random.rand(nptls, dim) * grid.box
    mass = np.random.rand(nptls) * (mr[1] - mr[0]) + mr[0]
    mass = 10**mass
    ptlList = []
    for i in range(nptls):
        ptl = constructor(pos[i, :], mass[i], 
                *const_args, **const_kwargs)
        ptlList.append(ptl)
    return ptlList

def gaussian(grid, mean = 0, stdev = 1):
    rng_array = np.random.normal(mean, stdev, grid.getShape())
    grid.setDelta(rng_array)
    return