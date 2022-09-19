from os import lseek
from pkfire.grid_lib.grid import Grid
from pkfire.pk_lib.pk import pk, pk2D
import numpy as np

class Pkfire():

    def __init__(self, grid, particles = []):
        # user should make own grid object - can 
        # be empty or they can attune it as they want
        self.particles = particles
        self.grid = grid
        self.pk1D = None
        self.pk2D = None
        self.pk3D = None
        return

    def addPtl(self, ptls):
        if isinstance(ptls, list):
            self.particles.extend(ptls)
        else:
            self.particles.append(ptls)
        return
    
    def lines(self, distbin):
        # find the pairs of particles that are separated
        # within the range given

        return

    def subPkfire(self, attrs = {}):
        # use attrs to find matches in the particles,
        # use the subset that match to make a new Pkfire obj
        # user can use the new object to calculate Pks for subset

        sub_pkfire = Pkfire(Grid(self.grid.getShape(), self.grid.box))
        for ptl in self.particles:
            if ptl.isMatch(attrs):
                sub_pkfire.addPtl(ptl)
        
        return sub_pkfire
    
    def pk(self):
        # get positions and masses of the particles
        dim = len(self.grid.getShape())
        nptls = len(self.particles)
        pos_arr = np.zeros((nptls, dim), dtype = np.float32)
        mass_arr = np.zeros(nptls, dtype = np.float32)

        for ptl in range(nptls):
            pos_arr[ptl, :] = self.particles[ptl].pos
            mass_arr[ptl] = self.particles[ptl].mass

        # add the particles to the grid, if there are any
        self.grid._CICW(pos_arr, mass_arr)
        # calculate power spectrum, the implementation is stored in pklib
        if dim == 2:
            results = pk2D(self.grid.getDelta(), self.grid.box)
            self.pk1D = results
        elif dim == 3:
            results = pk(self.grid.getDelta(), self.grid.box)
            # set the pk results
            self.pk1D = results[0]
            self.pk2D = results[1]
            self.pk3D = results[2]
        else:
            msg = "Pkfire only accepts dimensions of 2 or 3"
            raise NotImplementedError(msg)
        return

    def xi(self):
        # same process as pk but save Xi results
        return

class XPkfire():
    # same thing as pkfire but stores 2 grids and calculates the cross-power spectra
    def __init__(self):
        return