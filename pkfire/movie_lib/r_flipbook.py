from pkfire.movie_lib.movie_super import Movie
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class RFlipbook(Movie):
    def __init__(self, pkfire, duration, outpath = '', darkmode = False):
        super().__init__(pkfire, duration)
        self.cmap = 'plasma'
        self.norm = 'lin'
        self.dm = darkmode
        self.out = outpath
        
        # labels
        self.ptllabs = dict(
            xlabel = 'x (cMpc / h)',
            ylabel = 'y (cMpc / h)')
        
        self.xilabs = dict(
            xlabel = 'r (cMpc / h)',
            ylabel = '$\\xi$ (r) (cMpc / h)$^3$'
        )

        self.pklabs = dict(
            xlabel = 'k (h / cMpc)',
            ylabel = 'P (k) (h / cMpc)$^3$'
        )

        self.rcbarlab = 'r (cMpc / h)'
        self.mcbarlab = 'M (M$_\\odot$)'
        self.label_props = dict(
            fontsize = 12
        )
        return
    
    def setCmap(self, cmap = '', norm = ''):
        # sets the colormap for the different r values
        if not cmap == '':
            self.cmap = cmap
        if not norm == '':
            self.norm = norm
        return
    
    def _runPkfire(self):
        self.pkf.gridPtls()
        self.pkf.pk()
        self.pkf.xi()
        return
    
    
    def _makeNorm(self):
        r = self.pkf.xi3D[0]
        rmin = np.min(r); rmax = np.max(r)
        if self.norm == 'lin':
            norm = mpl.colors.Normalize(rmin, rmax)
        elif self.norm == 'log':
            norm = mpl.colors.LogNorm(rmin, rmax)
        return norm
    
    def run(self, rbins = [], del_frames = True, filename = 'movie.gif',
                fig_kwargs = {}):

        def _axProps(ax, labs):
            ax.set_xlabel(labs['xlabel'], **self.label_props)
            ax.set_ylabel(labs['ylabel'], **self.label_props)
            ax.tick_params(which = 'both', direction = 'in')
            return
            
        def _cbarProps(cax, lab):
            cax.yaxis.set_label_position('left')
            cax.set_ylabel(lab, **self.label_props)
            cax.tick_params(which = 'both', left = True, right = False,
                    labelleft = True, labelright = False)
            return
        # calculate pk and xi
        self._runPkfire()

        nnodes = self.pkf.getShape()[0]
        # make cmap
        if isinstance(self.norm, str):
            norm = self._makeNorm()
        else:
            norm = self.norm
        if isinstance(self.cmap, str):
            cmap = mpl.colormaps[self.cmap]
        else:
            cmap = self.cmap
        

        # figure out which radii bins to use
        if not rbins or isinstance(rbins, int):
            if isinstance(rbins, int):
                step = rbins
            else:
                step = 1
            rbins = self.pkf.xi3D[0][::step]
        
        # for each frame of the movie
        nframes = len(rbins) - 1
        prog_marker = 0.1
        pkidx = 0; xiidx = 1; rcbar = 2; mcbar = 3; ptlidx = 4
        for i in range(nframes):
            if i / nframes > prog_marker:
                print("%.1f percent done"%(i/nframes * 100))
                prog_marker += 0.1
            # get rlim
            rmin = rbins[i]
            rmax = rbins[i+1]
            
            # make the figure, axes
            cbar_width = 0.15
            cbar_mask = [rcbar, mcbar]
            wrs = np.ones(5)
            wrs[cbar_mask] = cbar_width
            
            # make xi, pk, ptl, cbar 1 and 2 panels
            fig, axes = self._makeFig(5, width_ratios=wrs)

            



            # plot entire xi, pk lines in dim gray in background
            self.pkf.pkPlot(axes[pkidx], color = 'gray', linestyle = '--')
            _axProps(axes[pkidx], self.pklabs)
            self.pkf.xiPlot(axes[xiidx], color = 'gray', linestyle = '--')
            _axProps(axes[xiidx], self.xilabs)

            # plot initial ptl scatter plot
            mid = int(nnodes / 2)
            zdist = int(nnodes * 0.1)
            slc = slice(mid - zdist, mid + zdist)
            self.pkf.ptlPlot(axes[ptlidx], axes[mcbar], slc = slc)
            _axProps(axes[ptlidx], self.ptllabs)
            _cbarProps(axes[mcbar], self.mcbarlab)
            
            # get the color for this rlim
            rcolor = cmap(norm((rmin + rmax)/2))

            # plot the rlim segment for xi, pk
            linekw = dict(
                linestyle = '-',
                linewidth = 2,
                color = rcolor
            )
            self.pkf.pkPlot(axes[pkidx], rlim = [rmin, rmax], **linekw)
            self.pkf.xiPlot(axes[xiidx], rlim = [rmin, rmax], **linekw)

            # plot rlim pairs for ptl scatter
            self.pkf.ptlPlot(axes[ptlidx], cax = 'skip', rlim = [rmin, rmax], 
                    rline_kwargs = {'color':rcolor}, slc = slc)
            
            # make cbar for xi, pk line segments
            plt.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax = axes[rcbar])
            _cbarProps(axes[rcbar], self.rcbarlab)
            # save frame
            fig.savefig(self.out + "_frame%04d.png"%i)
            plt.clf()
            

        # use frames to make movie
        self._makeMovie(self.out + "_frame*.png", self.out + filename,
                del_frames=del_frames)



        return