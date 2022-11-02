from pkfire.grid_lib.grid import Grid
from pkfire.pk_lib.pk import pk, pk2D, xi, xpk, xxi
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go

class Pkfire():

    def __init__(self, grid, particles = None, verbose = True):
        # user should make own grid object - can 
        # be empty or they can attune it as they want
        self.particles = particles

        self.added_ptls = False

        self.grid = grid
        self.pk1D = None
        self.pk2D = None
        self.pk3D = None

        self.xi3D = None

        self.v = verbose
        return

    def getBox(self):
        return self.grid.box

    def getShape(self):
        return self.grid.getShape()

    def _getPosMassArr(self):
        pos = self.particles.getPos(self.grid)
        mass = self.particles.getMass()
        return pos, mass
    
    def addPtl(self, ptls):
        if isinstance(ptls, list):
            self.particles.extend(ptls)
        else:
            self.particles.append(ptls)
        return

    def clearGrid(self):
        self.grid.clear()
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

    def gridPtls(self):
        if not self.added_ptls:
            pos_arr, mass_arr = self._getPosMassArr()
            self.grid._CICW(pos_arr, mass_arr)
        return

    def pk(self):
        # get positions and masses of the particles
        dim = len(self.grid.getShape())

        # calculate power spectrum, the implementation is stored in pklib
        if dim == 2:
            results = pk2D(self.grid.getDelta(), self.grid.box, verbose = self.v)
            self.pk1D = results
        elif dim == 3:
            results = pk(self.grid.getDelta(), self.grid.box, verbose = self.v)
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
        dim = len(self.grid.getShape())

        if dim == 2:
            raise NotImplementedError('xi not defined for 2D')
        elif dim == 3:
            results = xi(self.grid.getDelta(), self.grid.box, verbose = self.v)

            self.xi3D = results
        else:
            msg = "Pkfire only accepts dimensions of 2 or 3"
            raise NotImplementedError(msg)
        return

    def _plotSegment(self, ax, x, y, xmin, xmax, plot_kwargs):
        xmask = (x >= xmin) & (x < xmax)
        ax.plot(x[xmask], y[xmask], **plot_kwargs)
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
    
    def gridPlot(self, ax, cax = None, cmap = 'viridis', slc = None,
                plot_kwargs = {}, **other_kwargs):
        plot_kwargs.update(other_kwargs)
        grid = self.grid
        array = grid.grid
        dim = len(self.grid.getShape())
        nnodes = self.grid.getShape()[0]
        kwargs = {
            'aspect':'auto',
            'origin':'lower',
            'extent': (0, grid.box, 0, grid.box)
        }
        kwargs.update(plot_kwargs)
        if grid.is_overdensity:
            cax.set_ylabel('Overdensity')
            norm = mpl.colors.Normalize(np.min(array), np.max(array))
        else:
            cax.set_ylabel('Mass')
            norm = mpl.colors.LogNorm(1, np.max(array))
        if dim == 2:
            ax.imshow(array, norm = norm, cmap = cmap, **kwargs)
        elif dim == 3:
            if slc is None:
                slc = (slice(None), slice(None), nnodes//2)
                array_slc = array[slc]
            else:
                slc = (slice(None), slice(None), slc)
                array_slc = np.sum(array[slc], axis = 2)
            ax.imshow(array_slc, norm = norm, cmap = cmap, **kwargs)
            
        smap = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
        if cax is not None:
            plt.colorbar(cax = cax, mappable=smap)
        else:
            plt.colorbar(ax = ax, mappable = smap)
        

        cax.yaxis.set_label_position('right')
        return

    def _getRadiiPairs(self, rlim, zmin = -1, zmax = -1):
        pairs = []
        nptls = self.particles.nptls
        for i in range(nptls):
            for j in range(nptls):
                ithpos = self.particles.pos[i, :]
                jthpos = self.particles.pos[j, :]

                r = np.sqrt(np.sum((ithpos - jthpos)**2))
                if zmin == -1 and zmax == -1:
                    # expects output pos in same dimensions as grid
                    # (works for both 2D and 3D)
                    if r >= rlim[0] and r < rlim[1]:
                        pairs.append([ithpos, jthpos])
                else:
                    # is taking a slice of the 3D grid - 
                    # wants projected 2D pos
                    iz = ithpos[2]
                    jz = jthpos[2]
                    i_in_slc = (iz >= zmin) and (iz < zmax)
                    j_in_slc = (jz >= zmin) and (jz < zmax)

                    r_in_range = (r>= rlim[0]) and (r < rlim[1])

                    if i_in_slc and j_in_slc and r_in_range:
                        pairs.append([ithpos[:2], jthpos[:2]])

        return pairs

    def ptlPlot(self, ax, cax = None, cmap = 'viridis', norm = None, slc = None, rlim = [],
                scatter_kwargs = {}, rline_kwargs = {}, **other_kwargs): # add opt to plot lines
        
        if self.particles is None:
            raise ValueError("particles not given")
        
        # if 3D, get slice width
        dim = len(self.grid.getShape())
        nmodes = self.grid.getShape()[0]
        if slc is None and dim == 3:
            mid = nmodes // 2
            slc = slice(mid - 1, mid + 1)
        elif not slc is None and dim == 2:
            raise ValueError("slc input not defined for 2D grid")
        zvals = np.linspace(0, self.grid.box, nmodes)
        zmin = zvals[slc.start]; zmax = zvals[slc.stop]

        scatter_kwargs.update(other_kwargs)
        box = self.grid.box
        pos_arr, mass_arr = self._getPosMassArr()
        if norm is None:
            norm = mpl.colors.LogNorm(1, np.max(mass_arr))

        ax.set(xlim = [0, box], ylim = [0, box])
        smap = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
        if cax is not None and not cax == 'skip':
            plt.colorbar(cax = cax, mappable=smap)
        elif not cax == 'skip':
            plt.colorbar(ax = ax, mappable = smap)
        
        if rlim:
            if dim == 2:
                pairs = self._getRadiiPairs(rlim)
            elif dim == 3:
                pairs = self._getRadiiPairs(rlim, zmin, zmax)
            lines = []
            for p in pairs:
                segment = [p[0], p[1]]
                lines.append(segment)
            kwargs = {'color':'gray', 'linestyle':'dashed'}
            kwargs.update(rline_kwargs)
            lc = mpl.collections.LineCollection(lines, **kwargs)
            ax.add_collection(lc)
        else:
            if dim == 2:
                ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c = mass_arr,
                    norm = norm, cmap = cmap, **scatter_kwargs)
            if dim == 3:

                zmask = (pos_arr[:, 2] >= zmin) & \
                    (pos_arr[:, 2] < zmax)

                ax.scatter(pos_arr[zmask, 0], pos_arr[zmask, 1], 
                        c = mass_arr[zmask], norm = norm,
                        cmap = cmap, **scatter_kwargs)
        return

    def ptlPlot3D(self, marker_args = {}, rlim = [], rline_args = {}):
        dim = len(self.grid.getShape())
        if dim < 3:
            raise ValueError("3D plot not defined for 2D grid")
        
        box = self.grid.box
        side = [0, box]
        
        margs = {}
        margs['colorbar'] = {'thickness':10, 'len':0.5}
        margs['size'] = 2
        layout = go.Layout(scene = {
            'xaxis':{'range':side},
            'yaxis':{'range':side},
            'zaxis':{'range':side}
            })
        
        if rlim:
            pairs = self._getRadiiPairs(rlim)
            residual_ptls = np.ones(len(self.particles))
            xs = []; ys = []; zs = []; ms = []
            for p in pairs:
                ptl1 = self.particles[p[0]].pos
                ptl2 = self.particles[p[1]].pos
                residual_ptls[p[0]] = 0
                residual_ptls[p[1]] = 0
                x = [ptl1[0], ptl2[0]]
                y = [ptl1[1], ptl2[1]]
                z = [ptl1[2], ptl2[2]]
                mass = [np.log10(self.particles[p[0]].mass), 
                        np.log10(self.particles[p[1]].mass)]
                xs.extend(x); ys.extend(y); zs.extend(z); ms.extend(mass)
                xs.append(None); ys.append(None) 
                zs.append(None); ms.append(0)

            for rp in range(len(residual_ptls)):
                if residual_ptls[rp]:
                    ptl = self.particles[rp]
                    xs.extend([ptl.pos[0], None])
                    ys.extend([ptl.pos[1], None])
                    zs.extend([ptl.pos[2], None])
                    ms.extend([np.log10(ptl.mass), 0])
            kwargs = {'color':'gray'}
            kwargs.update(rline_args)

            margs['color'] = ms
            margs.update(marker_args)
            lines = go.Scatter3d(x = xs, y = ys, z = zs, 
                    mode = 'lines+markers', line=kwargs,
                    marker = margs, showlegend=False)
        
            fig = go.Figure(data = lines, layout = layout)
        else:
            pos, mass = self._getPosMassArr()
            margs['color'] = mass
            margs.update(marker_args)
            scatter = go.Scatter3d(
                x = pos[:, 0],
                y = pos[:, 1],
                z = pos[:, 2],
                marker = margs,
                mode='markers')
            fig = go.Figure(data = scatter, layout = layout)
        # scatter = go.Scatter3d(x = x, y = y, z = z, 
        #         mode = 'markers', marker = marker_args)

        # fig.add_trace(scatter)
        


        # fig = px.scatter_3d(x = x, y = y, z = z, color = mass_arr,
        #         range_x=side, range_y=side, range_z=side)
        return fig
    
        
        
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
    