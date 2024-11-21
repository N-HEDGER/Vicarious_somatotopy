import os

import cortex
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

backend = matplotlib.get_backend()


def basic_plot(dat, vmax, subject='hcp_999999', vmin=0, rois=False, colorbar=False, cmap='plasma', ax=None, labels=True):
    """ basic_plot
    Plots 1D data using pycortex

    """

    dat = np.array(dat)

    light = cortex.Vertex(dat, subject=subject,
                          vmin=vmin, vmax=vmax, cmap=cmap)
    mfig = cortex.quickshow(light, with_curvature=True, with_rois=rois,
                            with_colorbar=colorbar, with_labels=labels, fig=ax)
    return mfig


def alpha_plot(dat, dat2, vmin, vmax, vmin2, vmax2, subject='hcp_999999', rois=False, labels=False, colorbar=False, cmap='nipy_spectral_alpha', ax=None):
    """ alpha_plot
    Plots 2D data using pycortex

    """

    light = cortex.Vertex2D(dat, dat2, subject=subject, vmin=vmin,
                            vmax=vmax, vmin2=vmin2, vmax2=vmax2, cmap=cmap)
    mfig = cortex.quickshow(light, with_curvature=True, with_rois=rois,
                            with_colorbar=colorbar, fig=ax, with_labels=labels)
    return mfig


class Plot(object):
    def __init__(self, **kwargs):
        """
        Initialize the Vis class.

        Args:
            **kwargs: Additional keyword arguments to be stored as attributes.

        Attributes:
            roifields (list): List of variable names in the `add_rois` function of the `cortex.quickflat.composite` module.
            roidict (dict): Dictionary containing the attributes from `kwargs` that are present in `roifields` or `linewidth`.
            flatfields (list): List of variable names in the `make_figure` function of the `cortex.cortex.quickflat` module.
            flatdict (dict): Dictionary containing the attributes from `kwargs` that are present in `flatfields`.
            col_list (list): List of color values for each region of interest (ROIs). Default is white for each ROI.
            labcol_list (list): List of label color values for each ROI. Default is white for each ROI.
            linewidth_list (list): List of line widths for each ROI. Default is 10 for each ROI.
            dashes_list (list): List of dash patterns for each ROI. Default is no dashes for each ROI.
            zoomrects (list): List of zoom rectangles for each ROI. Only present if `zoomrect_list` attribute is provided.

        """
        self.__dict__.update(kwargs)

        self.roifields = cortex.quickflat.composite.add_rois.__code__.co_varnames
        self.roidict = {k: v for k, v in self.__dict__.items()
                        if k in self.roifields or k in ['linewidth']}

        self.flatfields = cortex.cortex.quickflat.make_figure.__code__.co_varnames
        self.flatdict = {k: v for k,
                         v in self.__dict__.items() if k in self.flatfields}

        if not hasattr(self, 'col_list') and hasattr(self, 'ROIS'):
            self.col_list = [[1, 1, 1]]*len(self.ROIS)

        if not hasattr(self, 'labcol_list') and hasattr(self, 'ROIS'):
            self.labcol_list = [[1, 1, 1]]*len(self.ROIS)

        if not hasattr(self, 'linewidth_list') and hasattr(self, 'ROIS'):
            self.linewidth_list = [10]*len(self.ROIS)

        if not hasattr(self, 'dashes_list') and hasattr(self, 'ROIS'):
            self.dashes_list = [[0, 0]]*len(self.ROIS)

        if hasattr(self, 'zoomrect_list') and hasattr(self, 'ROIS'):
            self.zoomrects = self.zoomrect_list

    def uber_plot(self, **kwargs):

        self.__dict__.update(kwargs)

        """ uber_plot
        Flexible plotter of data with pycortex.
        
        """

        figure = plt.figure(figsize=(self.x, self.y), dpi=self.dpi)

        if 'rdat' in kwargs and 'gdat' in kwargs and 'bdat' in kwargs and 'dat2' in kwargs:
            vx = cortex.VertexRGB(
                self.rdat, self.gdat, self.bdat, alpha=self.dat2, subject=self.subject)

        elif 'dat2' not in kwargs:
            alpha = False
            dat2 = np.ones_like(self.dat)
            vx = cortex.Vertex2D(self.dat, dat2, subject=self.subject,
                                 vmin=self.vmin, vmax=self.vmax, vmin2=0, vmax2=1, cmap=self.cmap)
        else:
            alpha = True
            vx = cortex.Vertex2D(self.dat, self.dat2, subject=self.subject, vmin=self.vmin,
                                 vmax=self.vmax, vmin2=self.vmin2, vmax2=self.vmax2, cmap=self.cmap)

        self.fig = cortex.quickflat.make_figure(vx, **self.flatdict)

        if 'ROIS' in self.__dict__:

            for c, v in enumerate(self.ROIS):
                print("adding rois")
            # Highlight face- and body-selective ROIs:
                _ = cortex.quickflat.composite.add_rois(self.fig, vx,
                                                        roi_list=[v], linecolor=self.col_list[c], dashes=self.dashes_list[c], labelcolor=self.labcol_list[c], linewidth=self.linewidth_list[c], labelsize=self.labelsize, **self.roidict)

        if 'zoomrect' in self.__dict__:
            plt.axis(self.zoomrect)

        return self.fig

    def saveout(self):

        self.fig.savefig(os.path.join(
            self.outpath, self.outname), dpi=self.dpi)
