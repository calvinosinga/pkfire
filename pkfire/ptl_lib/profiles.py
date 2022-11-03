from pkfire.ptl_lib.particle import Particles
import numpy as np

class Profiles(Particles):

    def __init__(self, centers, pf_args, profile_func):
        # masses are determined in the function, so replace with None
        super().__init__(centers, None)
        self.pfunc = profile_func
        self.pfargs = pf_args
        return
    
    
    def toGrid(self, grid):
        box = grid.getBox()
        shape = grid.getShape()
        x = np.linspace(0, box, shape[0])
        X, Y, Z = np.meshgrid(x, x, x)
        array = grid.grid
        for i in range(self.nptls):
            pos = self.pos[i, :]
            radii = np.zeros(shape)
            radii = \
                (X - pos[0])**2 + \
                (Y - pos[1])**2 + \
                (Z - pos[2])**2
            args_list = [i for i in self.pfargs[i]]
            print(args_list)
            mass = self.pfunc(radii, *args_list)
            array += mass
        return
    
    
    def _plot(self, *args):
        raise NotImplementedError("a particle plot for profiles is not defined")
        
