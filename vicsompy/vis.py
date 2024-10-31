

from .surface import CiftiHandler
import os

import cortex
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pkg_resources
import yaml
from matplotlib import gridspec
from matplotlib.colors import Normalize


def normalize_array(arr):
    """
    Normalizes a NumPy array into the 0-1 range.

    Parameters:
    arr (numpy array): The array to be normalized.

    Returns:
    numpy array: The normalized array with values between 0 and 1.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def clip_and_normalize(data, vmin, vmax):
    """
    Clips the data between vmin and vmax, then normalizes it between 0 and 1.

    Parameters:
    - data: numpy array or array-like, the input data.
    - vmin: float, the minimum value for clipping.
    - vmax: float, the maximum value for clipping.

    Returns:
    - normalized_data: numpy array, the clipped and normalized data.
    """
    # Ensure data is a NumPy array
    data = np.array(data)

    # Clip the data between vmin and vmax
    clipped_data = np.clip(data, vmin, vmax)

    # Handle the case where vmax == vmin to avoid division by zero
    if vmax == vmin:
        # If vmax and vmin are equal, return an array of zeros
        normalized_data = np.zeros_like(clipped_data)
    else:
        # Normalize the clipped data to be between 0 and 1
        normalized_data = (clipped_data - vmin) / (vmax - vmin)

    return normalized_data


backend = matplotlib.get_backend()


base_path = os.path.dirname(os.path.dirname(
    pkg_resources.resource_filename("Sensorium", 'config')))

# Define the path to the config file for the package.
pkg_yaml = os.path.join(base_path, 'config', 'config.yml')


def set_param(handle, param_name, param_value):
    handle.set(param_name, param_value)


def zoom_to_roi(subject, roi, hem, ax, margin=15):
    roi_verts = cortex.get_roi_verts(subject, roi)[roi]
    roi_map = cortex.Vertex.empty(subject)
    roi_map.data[roi_verts] = 1

    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat",
                                                                nudge=True)
    sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
    roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0], :2]
    print(roi_pts.min(0))
    xmin, ymin = roi_pts.min(0) - margin
    xmax, ymax = roi_pts.max(0) + margin

    ax.axis([xmin, xmax, ymin, ymax])
    print([xmin, xmax, ymin, ymax])
    return


def NormalizeData(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def zoom_to_rect(myrect):
    plt.axis(myrect)


def zoomed_plot(dat, vmin, vmax, ROI, hem, subject='hcp_999999', rois=False, colorbar=False, cmap='plasma', ax=None, labels=True, alpha=False):

    basic_plot(dat, vmax, subject, vmin, rois, colorbar, cmap, ax, labels)

    zoom_to_roi(subject, ROI, hem, ax)


def zoomed_alpha_plot(dat, dat2, vmin, vmax, vmin2, vmax2, ROI, hem, subject='fsaverage', rois=False, colorbar=False, cmap='plasma', ax=None, labels=True, alpha=False):
    alpha_plot(dat, dat2, vmin, vmax, vmin2, vmax2,
               cmap=cmap, rois=rois, labels=labels)
    zoom_to_roi(subject, ROI, hem)


def zoomed_plot2(dat, vmin, vmax, subject='hcp_999999', rect=[-229.33542, -121.50809, -117.665405, 28.478895], rois=False, colorbar=False, cmap='plasma', ax=None, labels=True):
    basic_plot(dat, vmax, subject, vmin, rois, colorbar, cmap, ax, labels)
    zoom_to_rect(rect)


def zoomed_alpha_plot2(dat, dat2, vmin, vmax, vmin2, vmax2, subject='hcp_999999', rect=[-229.33542, -121.50809, -117.665405, 28.478895], rois=False, colorbar=False, cmap='plasma', ax=None, labels=True):
    alpha_plot(dat, dat2, vmin, vmax, vmin2, vmax2,
               cmap=cmap, rois=rois, labels=labels)
    zoom_to_rect(rect)


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


def zoom_to_roi(subject, roi, hem, margin=15.0):
    """ zoom_to_roi
    zooms an roi specified in svg file for the subject.

    """

    roi_verts = cortex.get_roi_verts(subject, roi)[roi]
    roi_map = cortex.Vertex.empty(subject)
    roi_map.data[roi_verts] = 1

    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat",
                                                                nudge=True)
    sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
    roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0], :2]

    xmin, ymin = roi_pts.min(0) - margin
    xmax, ymax = roi_pts.max(0) + margin

    plt.axis([xmin, xmax, ymin, ymax])
    print([xmin, xmax, ymin, ymax])
    return


class Plot(object):
    def __init__(self, **kwargs):
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

    def make_zoomplots(self, **kwargs):

        nrow = 1
        ncol = 2

        self.zoomfig = plt.figure(figsize=(6, 3))

        gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1],
                               wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)

        self.__dict__.update(kwargs)
        self.plotlist = []
        self.canvases = []
        for c, zrect in enumerate(self.zoomrects):
            print(zrect)
            self.zoomrect = zrect
            self.plotlist.append(self.uber_plot())
            self.plotlist[c].canvas.draw()
            self.canvases.append(
                np.array(self.plotlist[c].canvas.buffer_rgba()))

        a = np.hstack(self.canvases)
        matplotlib.use(backend)

        for i in range(1):
            for j in range(2):
                ax = plt.subplot(gs[i, j])
                ax.imshow(self.canvases[j])
                ax.axis('off')

        return self.zoomfig

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

    def saveout_zoomfig(self):

        self.zoomfig.savefig(os.path.join(self.outpath, self.outname))

    def saveout(self):

        self.fig.savefig(os.path.join(
            self.outpath, self.outname), dpi=self.dpi)


class webplotter():

    def __init__(self, data, lims, cmaps, labels, subject, outpath='/Users/nicholashedger/Documents/tmp', port=2245, pause=1, alpha=True, overlays='/tank/hedger/DATA/nsd_pycortex_db/hcp_999999_draw_NH/custom_overlays.svg', alphadat=[], lims2=[], **kwargs):

        self.port = port
        self.data = data
        self.lims = lims
        self.lims2 = lims2
        self.cmaps = cmaps
        self.labels = labels
        self.subject = subject
        self.curv = cortex.db.get_surfinfo(subject)
        self.curv.vmin = np.nanmin(self.curv.data)
        self.curv.vmax = np.nanmax(self.curv.data)
        self.outpath = outpath
        self.pause = pause
        self.alpha = alpha
        self.alphadat = alphadat
        self.overlays = overlays

        self.alphalabs = ['alpha_' + var for var in self.labels]

        if self.alpha == True:

            self.prep_curv()

            if len(self.lims2) == 1:
                self.lims2 = self.lims2*len(self.data)
                self.alphadat = self.alphadat*len(self.data)

    def prep_premade_RGB(self):

        self.vx_data = []
        for i in range(len(self.data)):
            vx = self.data[i]

            vx_rgb = np.vstack(
                [vx.raw.red.data*255, vx.raw.green.data*255, vx.raw.blue.data*255])
            norm2 = Normalize(self.lims2[i][0], self.lims2[i][1])
            alpha = np.clip(norm2(self.alphadat[i]), 0, 1)

            display_data = self.curv_rgb * (1-alpha) + vx_rgb * alpha
            print(np.max(display_data))

            self.vx_data.append(cortex.VertexRGB(*display_data, self.subject))

        self.data_dict = dict(zip(self.labels, self.vx_data))

    def prep_curv(self):

        curv = cortex.db.get_surfinfo(self.subject)

        curv = curv.data
        normalized_curv = normalize_array(curv)-.5

        curv_im = normalized_curv

        curvT = (curv_im > 0).astype(float)
        curvT[np.isnan(curv_im)] = np.nan
        curv_im = curvT

        # vmin, vmax = -curvature_lims, curvature_lims
        # norm = Normalize(vmin=vmin, vmax=vmax)
        # curv_im = norm(curv)

        curv_im = (np.nan_to_num(curv_im) > 0.5).astype(float)
        curv_im[np.isnan(curv)] = np.nan

        # Scale and shift curvature image
        curv_im = (curv_im - 0.5) * .25 + .5

        curv = cortex.Vertex(curv_im, self.subject,
                             vmin=0, vmax=1, cmap='gray')
        self.curv_rgb = np.vstack(
            [curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    def prep_alpha_data(self, i):
        print('hello')

        if 'covar' in self.cmaps[i]:
            vx = cortex.Vertex2D(np.array(self.data[i]), self.alphadat[i], vmin=self.lims[i][0], vmax=self.lims[i]
                                 [1], vmin2=self.lims2[i][0], vmax2=self.lims2[i][1], cmap=self.cmaps[i], subject=self.subject)
            self.alphadat[i] = np.mean(
                [self.data[i], self.alphadat[i]], axis=0)

        else:
            vx = cortex.Vertex(np.array(self.data[i]), vmin=self.lims[i][0], vmax=self.lims[i][1], cmap=self.cmaps[i].split(
                '_alpha')[0], subject=self.subject)

        # Map to RGB
        vx_rgb = np.vstack(
            [vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
        norm2 = Normalize(self.lims2[i][0], self.lims2[i][1])
        # norm = Normalize(0, 1)
        # Pick an arbitrary region to mask out
        # (in your case you could use np.isnan on your data in similar fashion)

        self.alpha = np.clip(norm2(self.alphadat[i]), 0, 1)

        # self.alpha = clip_and_normalize(
        # self.alphadat[i], self.lims2[i][0], self.lims2[i][1])

        # normalized_alpha = (self.alphadat[i] - np.nanmin(self.alphadat[i])) / (
        #    np.nanmax(self.alphadat[i]) - np.nanmin(self.alphadat[i]))
        # self.alpha = np.clip(norm2(self.alphadat[i]), 0, 1)
        # self.alpha = np.clip(normalized_alpha,)
        # self.alpha = np.clip(norm2(self.alphadat[i]), 0, 1)

        # Alpha mask
        display_data = self.curv_rgb.astype('uint8') * (1-self.alpha.astype(
            'uint8')) + vx_rgb.astype('uint8') * self.alpha.astype('uint8')
        # display_data = vx_rgb

        vdat = cortex.VertexRGB(*display_data, self.subject)

        # vdat.raw.alpha = self.alpha[i]

        # Create vertex RGB object out of R, G, B channels
        return vdat

    def prep_data(self, i):

        if 'covar' in self.cmaps[i]:
            vx = cortex.Vertex2D(np.array(self.data[i]), self.alphadat[i], vmin=self.lims[i][0], vmax=self.lims[i]
                                 [1], vmin2=self.lims2[i][0], vmax2=self.lims2[i][1], cmap=self.cmaps[i], subject=self.subject)
            vx_rgb = np.vstack(
                [vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
            return cortex.VertexRGB(*vx_rgb, self.subject)

        else:

            vx = cortex.Vertex(np.array(self.data[i]), vmin=self.lims[i][0], vmax=self.lims[i][1], cmap=self.cmaps[i].split(
                '_alpha')[0], subject=self.subject)

        return vx

    def prep_all_data(self):
        print('hello')
        self.vx_data = []
        for i in range(len(self.data)):
            self.vx_data.append(self.prep_data(i))

        if self.alpha == True:

            for i in range(len(self.data)):

                self.vx_data.append(self.prep_alpha_data(i))
            self.labels = self.labels+self.alphalabs
        self.data_dict = dict(zip(self.labels, self.vx_data))

    def show(self):

        # self.prep_all_data()
        self.handle = cortex.webshow(
            self.data_dict, port=self.port, labels_visible=('display'), overlays_visible=('rois', 'display'), recache=True, overlay_file=self.overlays)

    def internalize_plot_yaml(self, myml):
        """internalize_config_yaml

        """
        self.yaml = myml
        with open(self.yaml, 'r') as f:
            self.y = yaml.safe_load(f)

        self.camera_dict = self.y['camera']

        self.animation_dict = self.y['animation']

        self.dlist = self.y['datatoplot']

        self.camera_dicts = [self.camera_dict[key]
                             for key in self.camera_dict.keys()]
        self.viewnames = [key for key in self.camera_dict.keys()]

        self.size_dict = self.y['size']
        self.with_labels = self.y['with_labels']

        for key in self.size_dict.keys():
            setattr(self, key, self.size_dict[key])

    def make_animation_dicts(self):
        self.animseq = []

        for key in self.animation_dict.keys():
            vals = np.array([np.linspace(self.animation_dict[key]['start_dict'][skey], self.animation_dict[key]['end_dict']
                            [skey], self.animation_dict[key]['nframes']) for skey in self.animation_dict[key]['start_dict'].keys()]).T

            self.animseq.append(
                [dict(zip(self.animation_dict[key]['start_dict'].keys(), i)) for i in vals])

    def make_anim_snaps(self):

        for idx, cam in enumerate(self.animation_dict.keys()):

            for idx, cdict in enumerate(self.animseq[idx]):

                cdict['camera.azimuth'] = int(cdict['camera.azimuth'])
                cdict['camera.altitude'] = int(cdict['camera.altitude'])
                cdict['camera.radius'] = int(cdict['camera.radius'])
                print(cdict)
                self.handle._set_view(**cdict)

                self.handle.draw()
                time.sleep(self.pause)
                figpath = os.path.join(
                    self.outpath, 'Anim_'+cam+"_{0:03}.png".format(idx))
                self.handle.getImage(figpath, size=(self.sizex, self.sizey))

    def make_snaps(self, label, camera_dict, c):

        for param_name, param_value in camera_dict.items():
            print(param_name)
            print(param_value)
            time.sleep(1)

            try:
                set_param(self.handle.ui, param_name, param_value)

            except:
                print('didnt set')
                time.sleep(5)
                set_param(self.handle.ui, param_name, param_value)

            # self.handle.ui.set(param_name, param_value)

        # self.handle._set_view(**camera_dict)
        # self.handle.draw()
        # time.sleep(self.pause)
        outstring = json.dumps(camera_dict)
        figpath = os.path.join(self.outpath, label+'_' +
                               self.viewnames[c]+'.png')

        self.handle.getImage(figpath, size=(self.sizex, self.sizey))

        while not os.path.exists(figpath):
            pass
        time.sleep(1)

    def make_data_snaps(self, label):

        self.handle.setData([label])

        try:
            if self.with_labels[self.dlist.index(label)] == True and self.labelstate == 0:
                print('showing labels')
                self.handle.ui.hide_labels()
                self.labelstate = 1
            if self.with_labels[self.dlist.index(label)] == False and self.labelstate == 1:
                print('hiding labels')
                self.handle.ui.hide_labels()
                self.labelstate = 0
        except:
            print('didnt set labels')
            time.sleep(5)
            if self.with_labels[self.dlist.index(label)] == True and self.labelstate == 0:
                print('showing labels')
                self.handle.ui.hide_labels()
                self.labelstate = 1
            if self.with_labels[self.dlist.index(label)] == False and self.labelstate == 1:
                print('hiding labels')
                self.handle.ui.hide_labels()
                self.labelstate = 0

        for c, camera_dict in enumerate(self.camera_dicts):
            self.make_snaps(label, camera_dict, c)

    def make_all_snaps(self):
        self.labelstate = 0
        self.names = []
        for lab in self.dlist:
            self.make_data_snaps(lab)

    def make_static(self):

        cortex.webgl.make_static(outpath=self.outpath,
                                 data=self.data_dict, recache=True)
