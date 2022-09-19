import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class PKFPlot():
    def __init__(self, pkfire):
        self.pkf = pkfire
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['font.family'] = 'serif'
        return

    
    def pkPlot(self, ax, unit = '', plot_kwargs = {}, **other_kwargs):
        plot_kwargs.update(other_kwargs)
        k = self.pkf.pk1D[0]
        pk = self.pkf.pk1D[1]
        ax.plot(k, pk, **plot_kwargs)
        if not unit == '':
            xlab = 'k (' + unit + ')'
            ylab = 'P(k) (' + unit + '$^2$'
        else:
            xlab = 'k'
            ylab = 'P(k)'
        nnodes = self.pkf.grid.getShape()[0]
        box = self.pkf.grid.box
        nyq = nnodes * np.pi / box
        ax.set(yscale = 'log', xscale = 'log', xlabel = xlab,
                ylabel = ylab, xlim = [np.min(k), nyq])

        return
    
    def pk2DPlot(self):


        return
    
    def gridPlot(self, ax, cax = None, cmap = 'viridis',
                plot_kwargs = {}, **other_kwargs):
        plot_kwargs.update(other_kwargs)
        grid = self.pkf.grid
        array = grid.grid
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

        ax.imshow(array, norm = norm, cmap = cmap, **kwargs)
        smap = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
        if cax is not None:
            plt.colorbar(cax = cax, mappable=smap)
        else:
            plt.colorbar(ax = ax, mappable = smap)
        

        cax.yaxis.set_label_position('right')
        return
    
    def ptlPlot(self, ax, cax = None, cmap = 'viridis', 
                scatter_kwargs = {}, **other_kwargs): # add opt to plot lines
        scatter_kwargs.update(other_kwargs)
        ptl_list = self.pkf.particles
        nptls = len(ptl_list)
        dim = len(self.pkf.grid.getShape())
        box = self.pkf.grid.box
        pos_arr = np.zeros((nptls, dim), dtype = np.float32)
        mass_arr = np.zeros(nptls, dtype = np.float32)

        for ptl in range(nptls):
            pos_arr[ptl, :] = ptl_list[ptl].pos
            mass_arr[ptl] = ptl_list[ptl].mass
        
        norm = mpl.colors.LogNorm(1, np.max(mass_arr))

        ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c = mass_arr,
                norm = norm, cmap = cmap, **scatter_kwargs)
        ax.set(xlabel = 'x', ylabel = 'y', xlim = [0, box], ylim = [0, box])
        smap = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
        if cax is not None:
            plt.colorbar(cax = cax, mappable=smap)
            
        else:
            plt.colorbar(ax = ax, mappable = smap)
        return
    

    
    