from multiprocessing.sharedctypes import Value
from pkfire.ptl_lib.particle import Particles

class Profiles(Particles):

    def __init__(self, centers, profile_func, pf_args = [], mass=None, attrs={}, **kwargs):
        super().__init__(centers, mass, attrs, **kwargs)
        self.pfunc = profile_func
        self.pfargs = pf_args
        self.cell_limit = -1
        return
    
    def _setCellLimit(self, cell_limit):
        self.cell_limit = cell_limit
        return
        
    def getPos(self, grid=None):
        if grid is None:
            raise TypeError("Profiles need grid input object")
        
        nnode_dist = grid.getNodeDist()
