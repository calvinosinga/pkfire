import numpy as np

class Particles():

    def __init__(self, pos, mass = None):
        self.pos = pos
        if mass is None:
            self.mass = np.ones(len(pos))
        else:
            self.mass = mass
        self.nptls = len(self.mass)
        return
    
    def getPos(self, grid = None): # gives flexibility for subclasses
        return self.pos
    
    def getMass(self):
        return self.mass

    def toGrid(self, grid):
        # given a grid, place the particles within the grid.
        grid._CICW(self.pos, self.mass)
        return

    def _plot(self, ax, zmin, zmax, cmap, norm, scatter_kw): # add opt to plot lines
        
        pos_arr = self.getPos()
        mass_arr = self.getMass()
        zmask = (pos_arr[:, 2] >= zmin) & \
            (pos_arr[:, 2] < zmax)

        ax.scatter(pos_arr[zmask, 0], pos_arr[zmask, 1], 
                c = mass_arr[zmask], norm = norm,
                cmap = cmap, **scatter_kw)
        return