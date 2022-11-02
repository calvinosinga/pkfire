from PIL import Image
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from pkfire.pkfire import Pkfire
from pkfire.grid_lib.grid import Grid

class Movie():

    def __init__(self, pkfire, duration, darkmode = False):
        self.pkf = pkfire
        self.duration = duration
        self.dm = darkmode
        return
    
    def _makeMovie(self, inpath, outpath, loop = 0, del_frames = True):
        imgs = (Image.open(f) for f in sorted(glob.glob(inpath)))
        img = next(imgs)
        img.save(fp=outpath, format='GIF', append_images=imgs,
                save_all=True, duration=self.duration * 1e3, loop=loop)

        # delete frames, if desired
        if del_frames:
            for i in glob.glob(inpath):
                os.remove(i)
        return
    
    @classmethod
    def _makeFig(self, ncols, panel_size = 3, width_ratios = None, gspec_kw = {}):
        # make default
        if width_ratios is None:
            width_ratios = np.ones(ncols)
        # xbwidth = 0.15
        # ybheight = 0.15
        gspec_kwargs = {}
        gspec_kwargs['width_ratios'] = width_ratios
        gspec_kwargs['wspace'] = 0.5
        gspec_kwargs['left'] = 0
        gspec_kwargs['right'] = 1
        gspec_kwargs['top'] = 1
        gspec_kwargs['bottom'] = 0
        gspec_kwargs.update(gspec_kw)
        figwidth = np.sum(panel_size * width_ratios) + \
            gspec_kwargs['wspace']*(ncols-1)*np.mean(width_ratios) * panel_size + \
            gspec_kwargs['left'] + (1 - gspec_kwargs['right'])
        figheight = panel_size + (1 - gspec_kwargs['top']) + \
            gspec_kwargs['bottom']
        
        fig, axes = plt.subplots(1, ncols,
                gridspec_kw=gspec_kwargs, figsize = (figwidth, figheight))
        return fig, axes
    
    def run(self, pkf_list, plot_func, outpath = ''):
        for i, pkf in enumerate(pkf_list):
            print("making frame %d"%i)
            pkf.gridPtls()
            pkf.pk(); pkf.xi()
            fig, axes = plot_func(pkf)
            if self.dm:
                self._darkmode(fig, axes)
            fig.savefig(outpath + '_frame%04d.png'%i)
        return


    @classmethod
    def _darkmode(cls, fig, axes):
        fig.set(facecolor = 'black')
        axes = np.ravel(axes)
        for ax in axes:
            # change spine colors to white
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')

            # change facecolors to black
            ax.set(facecolor = 'black')

            # change tick colors to white
            ax.tick_params(which = 'both', colors = 'white')

            # change label/title colors to white
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')

        return
    

            

