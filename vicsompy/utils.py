import os
import pickle

import nibabel as nib
import numpy as np
import pkg_resources
import scipy.stats as stats
import yaml
from sklearn.utils.validation import check_random_state

# Define the base path for the package
base_path = os.path.dirname(os.path.dirname(
    pkg_resources.resource_filename("vicsompy", 'config')))

# Define the path to the config file
pkg_yaml = os.path.join(base_path, 'config', 'config.yml')
plot_pkg_yaml = os.path.join(base_path, 'config', 'plot_config.yml')

# Define the paths to the subdirectories
p_paths = dict()

# Loop through the subdirectories and define the paths, then add them to the dict
for el in ['data', 'docs', 'results', 'scripts', 'tests']:
    p_paths['{el}_dir'.format(el=el)] = os.path.join(base_path, el)


def load_pkg_yaml(pkg_yaml=pkg_yaml, **kwargs):
    """_summary_

    Args:
        pkg_yaml (_type_, optional): The path to the package config yaml. Defaults to pkg_yaml.

    Returns:
        A dictionry containing the information in the package config file.
    """

    with open(pkg_yaml, 'r') as f:
        y = yaml.safe_load(f)

        # Add the paths to the config dict
        y['p_paths'] = p_paths

    if 'subdict' in kwargs.keys():
        return y[kwargs['subdict']]
    else:
        return y


def load_plot_pkg_yaml(pkg_yaml=plot_pkg_yaml, **kwargs):
    """_summary_

    Args:
        pkg_yaml (_type_, optional): The path to the package config yaml. Defaults to pkg_yaml.

    Returns:
        A dictionry containing the information in the package config file.
    """

    y = load_pkg_yaml(pkg_yaml)
    return y


def factory(module, name):

    class Lie:
        def __init__(self, *args, **kwargs):
            print(args, kwargs)
            self.__d = dict()

        def __setitem__(self, key, value):
            self.__d[key] = value

        def __getitem__(self, item):
            return self.__d[item]

        def __call__(self, *args, **kwargs):
            print(args, kwargs)

        def __repr__(self):
            return f"<class '{module}.{name}'>"

    return Lie


class Unpick(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except:
            return factory(module, name)


def generate_leave_one_run_out(n_samples, run_onsets, random_state=None,
                               n_runs_out=1):
    """
    https://github.com/gallantlab/voxelwise_tutorials
    (Note that this is borrowed directly from the gallantlab 'voxelwise tutorials' package so as not to install
    an entire package just for one function.)



    Generate a leave-one-run-out split for cross-validation.

    Generates as many splits as there are runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the training set.
    run_onsets : array of int of shape (n_runs, )
        Indices of the run onsets.
    random_state : None | int | instance of RandomState
        Random state for the shuffling operation.
    n_runs_out : int
        Number of runs to leave out in the validation set. Default to one.

    Yields
    ------
    train : array of int of shape (n_samples_train, )
        Training set indices.
    val : array of int of shape (n_samples_val, )
        Validation set indices.
    """
    random_state = check_random_state(random_state)

    n_runs = len(run_onsets)
    # With permutations, we are sure that all runs are used as validation runs.
    # However here for n_runs_out > 1, a run can be chosen twice as validation
    # in the same split.
    all_val_runs = np.array(
        [random_state.permutation(n_runs) for _ in range(n_runs_out)])

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs have no samples. Check that run_onsets "
                         "does not include any repeated index, nor the last "
                         "index.")

    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack([runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val


def pol2cart(rho, phi):
    """pol2cart
    convert polar to cartesian coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def cart2pol(x, y):
    """cart2pol
    convert cartesian to polar coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def gauss1D_cart(x, mu=0.0, sigma=1.0):
    """gauss1D_cart

    gauss1D_cart takes a 1D array x, a mean and standard deviation,
    and produces a gaussian with given parameters, with a peak of height 1.

    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the gauss
    mu : float, optional
        mean/mode of gaussian (the default is 0.0)
    sigma : float, optional
        standard deviation of gaussian (the default is 1.0)

    Returns
    -------
    numpy.ndarray
        gaussian values at x
    """

    return np.exp(-((x-mu)**2)/(2*sigma**2))


MMPloc = '/tank/hedger/scripts/natpac_CF/data/HCP-MMP_59k'


class MmpMasker:
    """
    Class for creating masks based on the MMP parcellation.

    Attributes
    ----------
    MMPloc : str
        Path to the MMP parcellation.

    Methods
    -------
    load()
        Loads the atlas and gets the ROI labels.
    decode_list(inlist)
        Decodes a list of strings from bytes to Unicode.
    get_roi_index(label, hem='L')
        Get the index of the named ROI.
    get_roi_verts(label)
        Get the vertices of an ROI.
    make_roi_mask(label, boolean=True)
        Make a mask of the named ROI.
    make_composite_mask(labels)
        Make a mask that is a composite of a list of ROIs.
    """

    def __init__(self, MMPloc=MMPloc):
        """
        Initialise the MMP_masker class.

        Parameters
        ----------
        MMPloc : str, optional
            Path to the MMP parcellation.
        """

        self.MMPloc = MMPloc
        self.load()

    def load(self):
        """load
        Loads the atlas and gets the ROI labels.
        """

        self.annotfile_L = os.path.join(self.MMPloc, 'lh.HCP-MMP1.annot')
        self.annotfile_R = os.path.join(self.MMPloc, 'rh.HCP-MMP1.annot')

        self.lh_labels, self.lh_ctab, self.lh_names = nib.freesurfer.io.read_annot(
            self.annotfile_L)
        self.rh_labels, self.rh_ctab, self.rh_names = nib.freesurfer.io.read_annot(
            self.annotfile_R)
        self.lh_names = self.decode_list(self.lh_names)
        self.rh_names = self.decode_list(self.rh_names)

    def decode_list(self, inlist):
        """
        Decode a list of strings from bytes to Unicode.

        Parameters
        ----------
        inlist : list
            The list of strings to decode.

        Returns
        -------
        list
            The decoded list of strings.
        """

        outlist = [x.decode() for x in inlist]
        return outlist

    def get_roi_index(self, label, hem='L'):
        """
        Get the index of the named ROI.

        Parameters
        ----------
        label : str
            The name of the ROI.
        hem : str, optional
            The hemisphere of the ROI ('L' for left, 'R' for right).

        Returns
        -------
        int
            The index of the named ROI.
        """

        if hem == 'L':
            n2search = self.lh_names
        elif hem == 'R':
            n2search = self.rh_names

        idx = n2search.index('{hem}_{label}_ROI'.format(label=label, hem=hem))

        return idx

    def get_roi_verts(self, label):
        """
        Get the vertices of an ROI.

        Parameters
        ----------
        label : str
            The name of the ROI.

        Returns
        -------
        tuple
            A tuple containing the vertices of the ROI for the left hemisphere and right hemisphere, respectively.
        """

        Lverts, Rverts = np.where(self.lh_labels == self.get_roi_index(label))[
            0], np.where(self.rh_labels == self.get_roi_index(label, hem='R'))[0]
        return Lverts, Rverts

    def make_roi_mask(self, label, boolean=True):
        """
        Make a mask of the named ROI.

        Parameters
        ----------
        label : str
            The name of the ROI.
        boolean : bool, optional
            Whether to return the mask as boolean values (True/False) or as integers (1/0).

        Returns
        -------
        tuple
            A tuple containing the mask for the left hemisphere, mask for the right hemisphere, and combined mask.
        """

        L_empty, R_empty = np.zeros(
            len(self.lh_labels)), np.zeros(len(self.rh_labels))
        Lverts, Rverts = self.get_roi_verts(label)
        L_empty[Lverts] = 1
        R_empty[Rverts] = 1

        combined_mask = np.concatenate([L_empty, R_empty])

        if boolean == True:
            L_empty, R_empty, combined_mask = L_empty.astype(
                bool), R_empty.astype(bool), combined_mask.astype(bool)

        return L_empty, R_empty, combined_mask

    def make_composite_mask(self, labels):
        """
        Make a mask that is a composite of a list of ROIs.

        Parameters
        ----------
        labels : list
            A list of ROI names.

        Returns
        -------
        ndarray
            The composite mask.
        """

        roimasks = np.sum([self.make_roi_mask(label)
                          for label in labels], axis=0)
        return roimasks
