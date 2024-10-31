import os

import numpy as np
import pandas as pd
import pkg_resources
import yaml

from .modeling import MssCf
from .subject import HcpSubject
from .utils import cart2pol, pol2cart

base_path = os.path.dirname(os.path.dirname(
    pkg_resources.resource_filename("vicsompy", 'config')))

# Define the path to the config file for the package.
pkg_yaml = os.path.join(base_path, 'config', 'config.yml')


class Aggregator():

    def __init__(self, expt_id, analysis_name, agg_name, yaml=pkg_yaml):
        """
        Initializes an instance of Aggregator.

        Args:
            expt_id (str): The experiment ID.
            analysis_name (str): The name of the analysis.
            agg_name (str): The name of the aggregate.
            yaml (str, optional): The YAML file path. Defaults to pkg_yaml.
        """
        base_sub = HcpSubject(subject='999999', experiment_id='movie')

        self.expt_id = expt_id
        self.analysis_name = analysis_name
        self.agg_name = agg_name

        self.yaml = pkg_yaml
        self.load_yaml()

        self.internalize_config(self.y, 'agg')
        self.internalize_config(self.y, 'paths')

        self.analysis_path = os.path.join(self.out_base, self.analysis_name)
        self.out_dir = os.path.join(self.analysis_path, 'aggregates')

        self.out_csv = os.path.join(self.out_dir, 'csvs', self.agg_name)
        self.out_flat = os.path.join(self.out_dir, 'flatmaps', self.agg_name)

        self.csvname = os.path.join(
            self.out_csv, self.out_csv_wildcard.format(name=self.agg_name))

        self.out_webGL = os.path.join(self.out_dir, 'webGL', self.agg_name)

        os.makedirs(self.out_csv, exist_ok=True)
        os.makedirs(self.out_flat, exist_ok=True)
        os.makedirs(self.out_webGL, exist_ok=True)

        self.all_subjects = base_sub.full_data_subjects

    def load_yaml(self):
        """ load_yaml
        Loads the yaml file into memory.
        """

        with open(self.yaml, 'r') as f:
            self.y = yaml.safe_load(f)

    def internalize_config(self, y, subdict):
        """ internalize_config
        Internalises a subdictionary of the yaml file.
        """

        subdict = y[subdict]

        for key in subdict.keys():
            setattr(self, key, subdict[key])

    def load_sub_frame(self, csub):
        """ load_sub_frame
        loads the output for a given subject
        """

        my_sub = HcpSubject(subject=csub, experiment_id=self.expt_id)
        my_sub.prepare_out_dirs(self.analysis_name)
        nm = MssCf(my_sub, self.analysis_name)
        frame = pd.read_csv(os.path.join(
            nm.subject.out_csv, nm.out_csv_wildcard))

        if 'spliced_params_eccentricity_visual' in frame.columns:
            frame['x'], frame['y'] = pol2cart(frame['spliced_params_eccentricity_visual'], np.radians(
                frame['spliced_params_angle_visual']))
        else:

            frame = frame.assign(spliced_params_eccentricity_visual=0)
            frame = frame.assign(spliced_params_angle_visual=0)

            frame['x'], frame['y'] = pol2cart(frame['spliced_params_eccentricity_visual'], np.radians(
                frame['spliced_params_angle_visual']))

        return frame

    def load_all_sub_frames(self, subjects):
        """ load_all_sub_frames
        loads the outputs for a set of subjects, returns into a list
        """

        self.subjects = subjects

        self.frames = [self.load_sub_frame(str(sub)) for sub in self.subjects]

        self.lframe = pd.concat(self.frames)

    def weighted_mean(self, df, values, weights, groupby):
        df = df.copy()
        grouped = df.groupby(groupby)
        df['weighted_average'] = df[values] / \
            grouped[weights].transform('sum') * df[weights]
        # min_count is required for Grouper objects
        return grouped['weighted_average'].sum(min_count=1)

    def summarise_fits(self, lframe, vars2av, vars2wav, weightvar):
        """ Summarise_fits

        Performs averaging, or weighted averaging to make a summary dataframe across folds.

        ----------

        """
        lframe = lframe.drop(['grouper'], axis=1, errors='ignore')

        lframe['grouper'] = lframe.index

        xval4weights = np.copy(lframe[weightvar])
        xval4weights[xval4weights < 0] = 0.00001
        lframe['weights'] = xval4weights

        vars2av = vars2av
        vars2wav = vars2wav

        wavs = pd.DataFrame(np.array(
            [self.weighted_mean(lframe, var, 'weights', 'grouper') for var in vars2wav]).T)
        wavs.columns = vars2wav

        # Regular averaging of variance explained.
        avs = pd.concat([lframe.groupby('grouper', as_index=False)[
                        var].mean() for var in vars2av], axis=1)

        av_frame = pd.concat([wavs, avs], axis=1)

        return av_frame

    def make_average(self):
        """ make_long_grand_average
        Average the long data across subjects.

        """

        self.av_frames = [self.summarise_fits(
            self.lframe, self.vars2avagg[c], self.vars2wavagg[c], self.weightvar[c]) for c, v in enumerate(self.vars2avagg)]

        self.av_frames[0]['spliced_params_eccentricity_visual'], self.av_frames[0]['spliced_params_angle_visual'] = cart2pol(
            self.av_frames[0]['x'], self.av_frames[0]['y'])

        self.av_frame = pd.concat(self.av_frames, axis=1)
        self.av_frame['test_scores_combined'] = self.av_frame['test_scores_somato_score'] + \
            self.av_frame['test_scores_visual_score']

    def saveout_frame(self):

        self.av_frame.to_csv(self.csvname)
