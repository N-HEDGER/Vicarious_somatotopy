# Imports
import os
from glob import glob

import nibabel as nib
import numpy as np
import pkg_resources
from scipy import stats
from tqdm import tqdm

from .surface import CiftiHandler
from .utils import *

base_path = os.path.dirname(os.path.dirname(
    pkg_resources.resource_filename("vicsompy", 'config')))

# Define the path to the config file for the package.
pkg_yaml = os.path.join(base_path, 'config', 'config.yml')


class HcpSubject():

    """ HcpSubject
    Class for interacting with and loading the HCP data.

    """

    def __init__(self, subject: str, experiment_id: str, yaml: str = pkg_yaml):
        """_summary_

        Args:
            subject (_type_): The string identifier for the subject.
            experiment_id (str): The string identifier for the experiment.
            yaml (str, optional): Path to the yaml file. Defaults to pkg_yaml.
        """

        self.subject = subject
        self.yaml = yaml
        self.experiment_id = experiment_id
        self.startup()

        self.internalize_config(self.y, 'aggregate_subjects')
        self.internalize_config(self.y, 'analysis')

        self.check_is_agg_sub()

        if 'movsplit' in self.subject:
            self.experiment_dict['runs'] = self.movsplit_runs[self.subject]

    def startup(self):
        """startup
        Reads in yaml file, internalises subject information.
        """

        self._internalize_config_yaml()
        self.internalize_config(self.y, 'paths')
        self._setup_paths_and_commands()

    def _internalize_config_yaml(self):
        """internalize_config_yaml
        Needs to have subject and experiment set up before running.
        Parameters
        ----------
        yaml_file : [string]
            path to yaml file containing config info
        """
        with open(self.yaml, 'r') as f:
            self.y = yaml.safe_load(f)

        # first, determine which host are we working on
        self.experiment_dict = self.y['experiments'][self.experiment_id]

    def _setup_paths_and_commands(self):
        """setup_paths
        sets up all the paths and commands necessary given an experiment id and subject
        """

        # set base directory paths dependent on experiment and subject
        self.experiment_base_dir = self.in_base.format(
            experiment=self.experiment_id)
        self.subject_base_dir = os.path.join(
            self.experiment_base_dir, 'subjects', self.subject)

    def check_is_agg_sub(self):
        """check_is_agg_sub
        If the subject is an aggregate subject, we move the subject base dir away from the HCP root folder
        to where the aggregate subjects have been created.
        """

        # Point to the location of the aggregate subjects.
        if self.subject in self.agg_subs:
            self.subject_base_dir = os.path.join(self.agg_path, self.subject)

    def internalize_config(self, y: dict, subdict: str):
        """Internalises a sub-dictionary from the yaml file.
        Args:
            y (dict): A dictionary containing analysis paramters.
            subdict (str): The sub-dictionary to internalise.
        """

        subdict = y[subdict]

        for key in subdict.keys():
            setattr(self, key, subdict[key])

    def get_data_path(self, run: int):
        """gets the data path for a given run.

        Args:
            run (int): The run to get the data path for.

        Returns:
            dpath (str): The data path.
        """

        wildcard = os.path.join(self.subject_base_dir, self.experiment_dict['data_file_wildcard'].format(
            experiment_id=self.experiment_dict['wc_exp'], run=self.experiment_dict['runs'][run]))
        dpath = glob(wildcard)[0]
        return dpath

    def get_data_paths(self):
        """get_data_paths
            Gets all the data paths for a given subject.
        """
        dpaths = []
        for run in range(len(self.experiment_dict['runs'])):
            dpaths.append(self.get_data_path(run))
        self.dpaths = dpaths

    def prep_data(self):
        """prep_data
            Prepares the data for analysis.
        """

        self.get_data_paths()

    def read_full_run_data(self, dat: str) -> np.ndarray:
        """Reads in the full run data.

        Args:
            dat (str): path to the data.

        Returns:
            np.ndarray: a numpy array of the data.
        """
        dataobj = nib.load(dat)
        data = dataobj.get_fdata().T
        return data

    def read_all_data(self):
        """
        Reads in all the data for a given subject.
        """

        print('Reading in data')
        concat_data = []

        for run in tqdm(range(len(self.experiment_dict['runs']))):
            concat_data.append(self.read_full_run_data(self.dpaths[run]))

        self.all_data = concat_data

        self.split_test_sequence()

        self.rundurs = [run.shape[-1] for run in self.all_data]
        self.run_onsets = np.concatenate(
            [[0], np.cumsum(np.array(self.rundurs))])[:-1]

        self.brainmodel = CiftiHandler(self.dpaths[0])
        self.brainmodel.get_data()

    def split_test_sequence(self):
        """split the test sequence from the rest of the data.
        """

        self.test_data = []

        if self.experiment_id == 'movie' or self.experiment_id == 'rs':
            for c, v in enumerate(self.all_data):
                self.test_data.append(stats.zscore(
                    v[:, -self.experiment_dict['test_duration']:], axis=1))
                self.all_data[c] = stats.zscore(
                    self.all_data[c][:, :-self.experiment_dict['test_duration']], axis=1)

        self.testarr = np.array(self.test_data)
        self.testarr = np.moveaxis(self.testarr, 1, -1)

        self.averaged_test_sequence = np.nanmean(self.testarr, axis=0)
        self.concatenated_test_sequence = np.concatenate(
            self.test_data, axis=1)

    def import_data(self):
        """import the data for a given subject.
        """

        self.read_all_data()
        self.all_data = np.hstack(self.all_data)

    def prepare_out_dirs(self, analysis_name: str):
        """Prepares the output directories for a given analysis.

        Args:
            analysis_name (str): The name of the analysis.
        """

        self.analysis_name = analysis_name

        self.out_csv = os.path.join(
            self.out_base, self.analysis_name, 'csvs', self.subject)

        self.out_flat = os.path.join(
            self.out_base, self.analysis_name, 'flat', self.subject)

        if not os.path.isdir(self.out_flat):
            os.makedirs(self.out_flat, exist_ok=True)

        if not os.path.isdir(self.out_csv):

            os.makedirs(self.out_csv, exist_ok=True)
