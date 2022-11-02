from pkfire.movie_lib.movie_super import Movie
from pkfire.pkfire import Pkfire
from pkfire.grid_lib.grid import Grid
from pkfire.ptl_lib.particle import Particles
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class SequenceMovie(Movie):

    def __init__(self, pkfire, duration, outpath = '', darkmode = False):
        super().__init__(pkfire, duration)
        self.dm = darkmode
        self.out = outpath
        self.props = dict(
            fontsize = 12,
            pkylab = 'P (k) (h / cMpc)$^3$',
            pkxlab = 'k (h / cMpc)',
            ptlylab = 'y (cMpc / h)',
            ptlxlab = 'x (cMpc / h)',
            xixlab = 'r (cMpc / h)',
            xiylab = '$\\xi$ (r) (cMpc / h)$^3$'
        )
        return
    
    def setProps(self, new_attr):
        self.props.update(new_attr)
        return
    
    def run(self, positions = None, masses = None, idx_list = None, xi_panel = False):
        postemp, masstemp = self.pkf._getPosMassArr()

        if positions is None:
            positions = postemp
        else:
            del postemp
        
        if masses is None:
            masses = masstemp
        else:
            del masstemp
        
        if idx_list is None:
            idx_list = np.arange(1, len(masses))
        
        pps = self.props
        for i, end in enumerate(idx_list):
            # make new grid and pkfire
            igrid = Grid(self.pkf.getShape(), self.pkf.getBox())
            subptllist = Particles(positions[:end, :], masses[:end])
            ipkf = Pkfire(igrid, subptllist)
            ipkf.gridPtls()

            # make pk and xi
            ipkf.pk(); ipkf.xi()

            # make figure and axes
            pkidx = 0
            ptlidx = 1
            cbar_idx = 2
            ncols = 3
            if xi_panel:
                ncols += 1; pkidx += 1; cbar_idx += 1; ptlidx += 1
                xi_idx = 0
                
            wrs = np.ones(ncols)
            wrs[cbar_idx] = 0.15

            fig, axes = self._makeFig(ncols, width_ratios=wrs)

            if self.dm:
                self._darkmode(fig, axes)
                        
            # make pk plot
            ax = axes[pkidx]
            ipkf.pkPlot(ax)
            ax.tick_params(which = 'both', direction = 'in')
            ax.set_ylabel(pps['pkylab'], fontsize = pps['fontsize'])
            ax.set_xlabel(pps['pkxlab'], fontsize = pps['fontsize'])
            
            # make ptl plot
            ax = axes[ptlidx]
            ipkf.ptlPlot(ax, cax = axes[cbar_idx])
            ax.set_ylabel(pps['ptlylab'], fontsize = pps['fontsize'])
            ax.set_xlabel(pps['ptlxlab'], fontsize = pps['fontsize'])
            ax.tick_params(which = 'both', direction = 'in')
            
            axes[cbar_idx].set_ylabel('M$_\\star$ (M$_\\odot$)')
            # make xi plot
            if xi_panel:
                ax = axes[xi_idx]
                ipkf.xiPlot(ax)
                ax.tick_params(which = 'both', direction = 'in')
                ax.set_ylabel(pps['xiylab'], fontsize = pps['fontsize'])
                ax.set_xlabel(pps['xixlab'], fontsize = pps['fontsize'])
            # save frame
            fig.savefig(self.out + '_frame%04d'%i, bbox_inches = 'tight')
            plt.clf()

        self._makeMovie(self.out + '_frame*.png', self.out + '_movie.gif')

            
        return
            
