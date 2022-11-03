import numpy as np


class Grid():

    def __init__(self, shape, boxsize):
        self.grid = np.zeros(shape, dtype = np.float32)
        self.box = boxsize
        self.is_overdensity = False # used to avoid accidentally normalizing twice
        
        return
    
    def getBox(self):
        return self.box
    
    def getShape(self):
        return self.grid.shape

    def getDelta(self):
        if self.is_overdensity:
            return self.grid
        else:
            return (self.grid / np.mean(self.grid)) - 1

    def clear(self):
        self.grid = np.zeros(self.getShape(), dtype = np.float32)
        return
        

    # METHODS FOR ALTERING THE MASS GRID ############################
    
    def setArray(self, mass_array):
        self.grid = mass_array
        return

    def setDelta(self, odensity_array):
        self.grid = odensity_array
        self.is_overdensity = True
        return

    def setZeros(self):
        self.grid = np.zeros_like(self.grid)
        return

    def _CICW(self, pos, masses):            

        boxsize = self.getBox()
        ptls = len(masses); coord = len(self.getShape()); dims = self.getShape()[0]
        inv_cell_size = dims/boxsize
        
        index_d = np.zeros(coord, dtype=np.int64)
        index_u = np.zeros(coord, dtype=np.int64)
        d = np.zeros(coord)
        u = np.zeros(coord)

        # get every combination of each index...
        uord_dim = [[0, 1] for i in range(coord)]
        uord = np.array(np.meshgrid(*uord_dim)).T.reshape(-1, coord)
        for i in range(ptls):

            for axis in range(coord):
                dist = pos[i,axis] * inv_cell_size
                u[axis] = dist - int(dist)
                d[axis] = 1 - u[axis]
                index_d[axis] = (int(dist))%dims
                index_u[axis] = index_d[axis] + 1
                index_u[axis] = index_u[axis]%dims #seems this is faster
            

            
            for ud in uord: # for each possible combination of 0 or 1
                # ud is list of len(coord) that contains a 0 or 1
                # in each element. A zero means we use index_d[ax] or d[ax]
                # one means index_u[ax] or u[ax]
                idx = np.zeros(coord, np.int32)
                factor = 1
                for ax in range(len(ud)):
                    if ud[ax] == 0:
                        factor *= d[ax]
                        idx[ax] = index_d[ax]
                    else:
                        factor *= u[ax]
                        idx[ax] = index_u[ax]
                self.grid[tuple(idx)] += factor * masses[i]
            

        return

    def _NGP(self):
        return
        
    def _plot(self, ax, zmin, zmax, cmap, norm, imshow_kw):
        edges = np.linspace(0, self.getBox(), self.getShape()[0])
        volume = (edges[zmax] - edges[zmin]) * edges[1]**2
        slc_sum = np.sum(self.grid[:, :, zmin:zmax], axis = 2) / volume
        ax.imshow(slc_sum, norm = norm, cmap = cmap, **imshow_kw)

        return

