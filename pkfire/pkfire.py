from pkfire.grid_lib.grid import Grid
from pkfire.pk_lib.pk import pk, pk2D, xi, xpk, xxi
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# import plotly.graph_objects as go

class Pkfire():

    def __init__(self, grid, particles = None, verbose = True):
        # user should make own grid object - can 
        # be empty or they can attune it as they want
        self.particles = particles

        self.grid = grid
        self.pk1D = None
        self.pk2D = None
        self.pk3D = None

        self.xi3D = None

        self.v = verbose

        # ptl plot params
        self.ptl_cmap = 'viridis'
        self.ptl_norm = 'log'

        # grid plot params
        self.grid_cmap = 'viridis'
        self.grid_norm = 'log'
        return

    def getBox(self):
        return self.grid.getBox()

    def getShape(self):
        return self.grid.getShape()

    def setPtlColormap(self, cmap = '', norm = None):
        if not cmap == '':
            self.ptl_cmap = cmap
        if not norm is None:
            self.ptl_norm = norm
        return
    
    def setGridColormap(self, cmap = '', norm = None):
        if not cmap == '':
            self.ptl_cmap = cmap
        if not norm is None:
            self.ptl_norm = norm
        return
    
    def clearGrid(self):
        self.grid.clear()
        return

    def gridPtls(self):
        self.particles.toGrid(self.grid)
        return

    def pk(self):


        # calculate power spectrum, the implementation is stored in pklib

        results = pk(self.grid.getDelta(), self.grid.box, verbose = self.v)
        # set the pk results
        self.pk1D = results[0]
        self.pk2D = results[1]
        self.pk3D = results[2]

        return

    def xi(self):
        # same process as pk but save Xi results
        results = xi(self.grid.getDelta(), self.grid.box, verbose = self.v)

        self.xi3D = results

        return


    def pkPlot(self, ax, rlim = [], setax = True, plot_kwargs = {}, 
            **other_kwargs):

        nnodes = self.grid.getShape()[0]
        box = self.grid.box
        nyq = nnodes * np.pi / box

        k = self.pk3D[0]
        pk = self.pk3D[1][:, 0] # Pk(k, ell) -> specifies ell = 0

        axkw = dict(
            yscale = 'log',
            xscale = 'log',
            xlim = [np.min(k), nyq])
        
        plot_kwargs.update(other_kwargs)

        if setax:
            ax.set(**axkw)

        if rlim:
            klim = 2*np.pi/np.array(rlim)
            kmin = np.min(klim)
            kmax = np.max(klim)

            self._plotSegment(ax, k, pk, kmin, kmax, plot_kwargs)
        else:
            ax.plot(k, pk, **plot_kwargs)
        return

    def xiPlot(self, ax, rlim = [], setax = True, mask_zeros = False, 
            plot_kwargs = {}, **other_kwargs):
        plot_kwargs.update(other_kwargs)
        r = self.xi3D[0]
        xi = self.xi3D[1][:, 0]

        nnodes = self.grid.getShape()[0]
        box = self.grid.box
        nyq = nnodes * np.pi / box
        if setax:
            ax.set(yscale = 'log', xscale = 'log', 
                    xlim = [2 * np.pi / nyq, np.max(r)])
        if mask_zeros:
            zero_mask = xi > 0
        else:
            zero_mask = np.ones_like(r, dtype = bool)
        if rlim:
            self._plotSegment(ax, r[zero_mask], xi[zero_mask], rlim[0], rlim[1], plot_kwargs)
        else:
            ax.plot(r[zero_mask], xi[zero_mask], **plot_kwargs)
        return

    def pk2DPlot(self):
        #TODO

        return
    

    def ptlPlot(self, ax, cax = None, slc = None, scatter_kw = {}, **other_kw):
        if self.particles is None:
            raise ValueError("particles not given")
    
        # since we are representing a 3D grid in 2D, we need to decide
        # what slice we are going to display
        nmodes = self.grid.getShape()[0]
        if slc is None:
            mid = nmodes // 2
            slc = slice(mid - 1, mid + 1)
        
        zvals = np.linspace(0, self.grid.box, nmodes)
        zmin = zvals[slc.start]; zmax = zvals[slc.stop]

        # now setting axis properties
        box = self.getBox()
        ax.set(xlim = [0, box], ylim = [0, box])

        # now setting colorbar properties
        if self.ptl_norm == 'log':
            norm = mpl.colors.LogNorm(1, np.max(self.particles.getMass()))
        else:
            raise ValueError("ptl plot norm not defined...")

        smap = mpl.cm.ScalarMappable(norm = norm, cmap = self.ptl_cmap)
        if cax is not None and not cax == 'skip':
            plt.colorbar(cax = cax, mappable=smap)
        elif not cax == 'skip':
            plt.colorbar(ax = ax, mappable = smap)
        
        # now let subclass make the plot
        scatter_kw.update(other_kw)
        self.particles._plot(ax, zmin, zmax, self.ptl_cmap, norm, scatter_kw)
        return
    
    def gridPlot(self, ax, cax = None, slc = None,
                plot_kwargs = {}, **other_kwargs):
        

        kwargs = {
            'aspect':'auto',
            'origin':'lower',
            'extent': (0, self.getBox(), 0, self.getBox())
        }
        plot_kwargs.update(other_kwargs)
        kwargs.update(plot_kwargs)

        if self.grid_norm is 'log':
            norm = mpl.colors.LogNorm(np.min(self.grid.grid), np.max(self.grid.grid))
        else:
            raise ValueError("norm for grid plot is not defined")
        smap = mpl.cm.ScalarMappable(norm = norm, cmap = self.grid_cmap)
        
        
        if cax is not None:
            plt.colorbar(cax = cax, mappable=smap)
        else:
            plt.colorbar(ax = ax, mappable = smap)
        # get zmin and zmax
        if slc is None:
            nodes = self.getShape()[0]
            zmin = int(nodes // 2 - nodes * 0.1)
            zmax = int(nodes // 2 + nodes * 0.1)
        else:
            zmin = slc.start; zmax = slc.stop
        self.grid._plot(ax, zmin, zmax, self.grid_cmap, norm, kwargs)
        return

    
        
        
class XPkfire():
    # same thing as pkfire but stores 2 grids and calculates the cross-power spectra
    def __init__(self, pkf1, pkf2, verbose = True):
        self.pkf1 = pkf1
        self.pkf2 = pkf2

        self.pk1D = None
        self.pk2D = None
        self.pk3D = None

        self.xi3D = None

        self.v = verbose

        self.obs_bias = None
        self.th_bias = None
        self.corr_coef = None
        return
    
    def gridPtls(self):
        self.pkf1.gridPtls()
        self.pkf2.gridPtls()
        return
    

    def pk(self):
        
        # retreive each grid

        grid1 = self.pkf1.grid
        grid2 = self.pkf2.grid

        # check that the boxes for each grid are equal
        if not grid1.box == grid2.box:
            raise ValueError("grid boxes are not the same size!")
        
        # get overdensities for xpk
        deltas = [grid1.getDelta(), grid2.getDelta()]

        # calculate xpk and unpack values
        out = xpk(deltas, grid1.getBox(), verbose = self.v)
        self.pk1D, self.pk2D, self.pk3D = out[0], out[1], out[2]
        self.pk3D = [self.pk3D[0], self.pk3D[1], self.pk3D[2]]
        auto1 = out[2][3][:, 0, 0]
        auto2 = out[2][3][:, 0, 1]
        xpk_out = out[2][1][:, 0, 0]
        self.obs_bias = np.sqrt(auto1/auto2)
        self.th_bias = [xpk_out/auto1, xpk_out/auto2]
        self.corr_coef = xpk_out / (np.sqrt(auto1 * auto2))
        return
    
    def xi(self):
        # retreive each grid

        grid1 = self.pkf1.grid
        grid2 = self.pkf2.grid

        # check that the boxes for each grid are equal
        if not grid1.box == grid2.box:
            raise ValueError("grid boxes are not the same size!")

        self.xi3D = xxi(
            grid1.getDelta(),
            grid2.getDelta(),
            grid1.getBox(),
            verbose = self.v)
        return
    
    def pkPlot(self, ax, rlim = [], setax = True, plot_kwargs = {}, 
            **other_kwargs):
        plot_kwargs.update(other_kwargs)
        
        k = self.pk3D[0]
        xpk = self.pk3D[1][:, 0, 0] # xpk(k, ell, field)

        nnodes = self.pkf1.getShape()[0]
        box = self.pkf1.getBox()
        nyq = nnodes * np.pi / box

        if setax:
            xlim = [np.min(k), nyq]
            ax.set(yscale = 'log', xscale = 'log', xlim = xlim)
        
        if rlim:
            klim = 2*np.pi/np.array(rlim)
            kmin = np.min(klim)
            kmax = np.max(klim)
            self._plotSegment(ax, k, xpk, kmin, kmax, plot_kwargs)
        else:
            ax.plot(k, xpk, **plot_kwargs)
        
        return
    
    def xiPlot(self, ax, rlim = [], setax = True, plot_kwargs = {},
            **other_kwargs):
        plot_kwargs.update(other_kwargs)
        r = self.xi3D[0]
        xxi = self.xi3D[1][:, 0]

        nnodes = self.pkf1.getShape()[0]
        box = self.pkf1.getBox()
        nyq = nnodes * np.pi / box
        if setax:
            ax.set(yscale = 'log', xscale = 'log', 
                    xlim = [2 * np.pi / nyq, np.max(r)])

        if rlim:
            self._plotSegment(ax, r, xxi, rlim[0], rlim[1], plot_kwargs)
        else:
            ax.plot(r, xxi, **plot_kwargs)
        return
    
    def obsbiasPlot(self, ax, setax = True, **plot_kw):
        k = self.pk3D[0]
        b = self.obs_bias

        nnodes = self.pkf1.getShape()[0]
        box = self.pkf1.getBox()
        nyq = nnodes * np.pi / box

        if setax:
            xlim = [np.min(k), nyq]
            ax.set(yscale = 'log', xscale = 'log', xlim = xlim)
        
        ax.plot(k, b, **plot_kw)
        return
    
    def corrcoefPlot(self, ax, setax = True, **plot_kw):
        k = self.pk3D[0]
        cc = self.corr_coef
        
        nnodes = self.pkf1.getShape()[0]
        box = self.pkf1.getBox()
        nyq = nnodes * np.pi / box

        if setax:
            xlim = [np.min(k), nyq]
            ax.set(xscale = 'log', xlim = xlim, ylim = [0, 1])
        
        ax.plot(k, cc, **plot_kw)
        return
        
    def xpk2DPlot(self):
        return
    
    def ptlPlot(self, ax, cax = None, cmap = 'viridis', norm = None, slc = None, rlim = [],
                scatter_kwargs = {}, rline_kwargs = {}, **other_kwargs):
        self.pkf1.pltPlot(ax, cax, cmap, norm, slc, rlim, scatter_kwargs,
                rline_kwargs, **other_kwargs)
        return
    



    # def _getRadiiPairs(self, rlim, zmin = -1, zmax = -1):
    #     pairs = []
    #     nptls = self.particles.nptls
    #     for i in range(nptls):
    #         for j in range(nptls):
    #             ithpos = self.particles.pos[i, :]
    #             jthpos = self.particles.pos[j, :]

    #             r = np.sqrt(np.sum((ithpos - jthpos)**2))
    #             if zmin == -1 and zmax == -1:
    #                 # expects output pos in same dimensions as grid
    #                 # (works for both 2D and 3D)
    #                 if r >= rlim[0] and r < rlim[1]:
    #                     pairs.append([ithpos, jthpos])
    #             else:
    #                 # is taking a slice of the 3D grid - 
    #                 # wants projected 2D pos
    #                 iz = ithpos[2]
    #                 jz = jthpos[2]
    #                 i_in_slc = (iz >= zmin) and (iz < zmax)
    #                 j_in_slc = (jz >= zmin) and (jz < zmax)

    #                 r_in_range = (r>= rlim[0]) and (r < rlim[1])

    #                 if i_in_slc and j_in_slc and r_in_range:
    #                     pairs.append([ithpos[:2], jthpos[:2]])

    #     return pairs