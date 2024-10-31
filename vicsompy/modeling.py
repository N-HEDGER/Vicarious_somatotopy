import os
import pickle
from copy import copy, deepcopy

import pandas as pd
import pkg_resources
import torch
import yaml
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import r2_score_split
from sklearn import set_config
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .surface import *
from .utils import Unpick, factory, generate_leave_one_run_out

base_path = os.path.dirname(os.path.dirname(
    pkg_resources.resource_filename("vicsompy", 'config')))

# Define the path to the config file
pkg_yaml = os.path.join(base_path, 'config', 'config.yml')


class MssCf():

    """ MssCf
    Multi-source spectral CF model. This class is used to fit a multi-source CF model to surface defined data.

    """

    def __init__(self, subject, analysis_name, yaml: str = pkg_yaml):
        """ initialze the sensory_dm class.

        Args:
            subject (object): A subject object must have the following attributes:

            # Essential
            subject: The subject identifier (str).
            analysis_name: A name for the analysis (str).
            yaml (str): A yaml file that defines the following subdictionaries:
                dm: Defines design matrix parameters.
                source_regions: Defines locations of subsurfaces, lookup tables, and masks.
                output: Defines output parameters.
                modeling: Defines modeling parameters.
                splicing: Defines splicing parameters.
                patches: Defines the parameters for decomposing the subsurface
                output: Defines parameters for saving out.

            An example yaml is bundled with the package and annotated. Defaults to pkg_yaml.
        """

        self.subject = deepcopy(subject)
        self.analysis_name = analysis_name
        self.yaml = yaml
        self.load_yaml()

        self.internalize_config(self.y, 'dm')
        self.internalize_config(self.y, 'output')
        self.internalize_config(self.y, 'source_regions')
        self.internalize_config(self.y, 'patches')
        self.internalize_config(self.y, 'output')

    def load_yaml(self):
        """_summary_
        loads a yaml file into memory
        """

        with open(self.yaml, 'r') as f:
            self.y = yaml.safe_load(f)

    def internalize_config(self, y: dict, subdict: str):
        """Internalises a sub-dictionary from the yaml file.
        Args:
            y (dict): A dictionary containing analysis paramters.
            subdict (str): The sub-dictionary to internalise.
        """

        subdict = y[subdict]

        for key in subdict.keys():
            setattr(self, key, subdict[key])

    def define_modality_indices(self):
        """Defines the indices for each modality. """

        self.comps_per_hem = [self.y['patches'][v]['laplacians']
                              for c, v in enumerate(self.modalities)]
        vals = np.insert(np.cumsum(self.comps_per_hem)*2, 0, 0)

        self.modality_idxs = []
        for v in range(len(vals)-1):
            self.modality_idxs.append(np.array(range(vals[v], vals[v+1])))

    def make_subsurface_from_mask(self, modality: str, laplacians: int):
        """ Makes a subsurface from a binary mask defined in a pandas dataframe.

        Args:
            modality (str): The name of the modality to make the subsurface for.
            laplacians (int): The number of laplacian components to make.

        Returns:
            surf: A pycortex subsurface object.
        """
        maskdatL = pd.read_csv(os.path.join(self.source_region_dir,
                                            self.maskdir, self.maskwcard.format(hem='L', modality=modality)))
        maskdatR = pd.read_csv(os.path.join(self.source_region_dir,
                                            self.maskdir, self.maskwcard.format(hem='R', modality=modality)))

        maskL, maskR = np.array(maskdatL['mask']), np.array(maskdatR['mask'])

        surf = Subsurface(self.pcx_sub, [maskL, maskR], surftype=self.surftype)
        surf.create()
        print('Making {ncomponents} laplacian components'.format(
            ncomponents=laplacians))
        surf.make_laplacians(laplacians)
        return surf

    def load_lookups(self):
        """Loads the lookup tables for each modality.
        """

        self.lookups = [pd.read_csv(os.path.join(self.source_region_dir, self.lookupdir, self.lookupwcard.format(
            modality=v))) for c, v in enumerate(self.modalities)]

        # Subset to the specific variables we want to splice.
        self.lookups = [self.lookups[c][self.y['patches'][v]['vars2splice']]
                        for c, v in enumerate(self.modalities)]

    def make_subsurfaces(self, force_new=False):
        """_summary_
        Makes the subsurfaces for each modality.

        Args:
            force_new (bool, optional): whether to make subsurfaces from scratch or to preload surfaces. Defaults to False.
        """
        self.define_modality_indices()

        if force_new:  # Make a new subsurface

            self.subsurfaces = [self.make_subsurface_from_mask(
                v, self.y['patches'][v]['laplacians']) for c, v in enumerate(self.modalities)]

        else:  #  Load a pre-made subsurface.
            self.subsurfaces = []

            for c, v in enumerate(self.modalities):

                with open(os.path.join(self.source_region_dir, self.surfdir, self.surfwcard.format(modality=v)), 'rb') as handle:
                    self.subsurfaces.append(Unpick(handle).load())

    def make_roi_data(self):
        """Gets the data within each subsurface and hemisphere.
        """
        self.roidat = []

        for c, v in enumerate(self.subsurfaces):
            self.roidat.append(self.data[getattr(v, 'subsurface_verts_L'), :])
            self.roidat.append(self.data[getattr(v, 'subsurface_verts_R'), :])

    def make_eigs(self):
        """Gets the eigenvectors for each subsurface and hemisphere.
        """

        self.eig = []
        for c, v in enumerate(self.subsurfaces):
            self.eig.append(getattr(v, 'L_eigenvectors'))
            self.eig.append(getattr(v, 'R_eigenvectors'))

    def make_dm(self, data: np.ndarray):
        """_summary_

        Args:
            data (np.ndarray): The data to make the design matrix from.
        """

        self.data = data
        self.make_roi_data()
        self.make_eigs()

        self.dms = []
        for c, v in enumerate(self.roidat):
            self.dms.append(np.dot(v.T, self.eig[c].real))
        self.dm = np.hstack(self.dms)

    def set_backend(self):
        """sets the backend for the Himalaya package.
        """
        self.backend = set_backend(self.backend_engine, on_error="warn")

    def make_cv(self, run_durations=None, ksplits=None):
        """Make the cross-validation scheme.

        Args:
            run_durations (np.ndarrray, optional): The onset of each run. Defaults to None.
            ksplits (int, optional): If kfold, the number of ksplits. Defaults to None.
        """
        if type(run_durations) == np.ndarray:
            # if run_durations!=None:
            self.n_samples_train = self.dm.shape[0]
            self.cv = generate_leave_one_run_out(
                self.n_samples_train, run_durations)
            self.cv = check_cv(self.cv)

        if ksplits != None:
            self.cv = int(ksplits)

    def make_preproc(self):
        """Make the preprocessing pipeline. Includes standardization and kernelization.
        """

        set_config(display='diagram')  # requires scikit-learn 0.23

        self.preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=self.with_mean, with_std=self.with_std),
            Kernelizer(kernel="linear")
        )

    def kernelize(self):
        """ Kernelize the design matrix by defiining the feature spaces in the design matrix.
        """

        start_and_end = np.concatenate(
            [[0], np.cumsum(np.array(self.comps_per_hem)*2)])
        slices = [
            slice(start, end)
            for start, end in zip(start_and_end[:-1], start_and_end[1:])]

        kernelizers_tuples = [(name, self.preprocess_pipeline, slice_)
                              for name, slice_ in zip(self.modalities, slices)]
        self.column_kernelizer = ColumnKernelizer(kernelizers_tuples)

    def setup_model(self):
        """Sets up the banded ridge model based on parameters defined in the yaml
        """

        self.alphas = np.logspace(
            self.alpha_min, self.alpha_max, self.alpha_vals)

        solver_params = dict(n_iter=self.n_iter, alphas=self.alphas,
                             n_targets_batch=self.n_targets_batch,
                             n_alphas_batch=self.n_alphas_batch,
                             n_targets_batch_refit=self.n_targets_batch_refit)

        self.model = MultipleKernelRidgeCV(kernels="precomputed", solver=self.solver,
                                           solver_params=solver_params, cv=self.cv)

    def complete_pipeline(self):
        """Completes the pipeline by adding the model to the preprocessing pipeline.
        """

        self.pipeline = make_pipeline(
            self.column_kernelizer,
            self.model,)

        # Make a copy of the pipeline prior to fitting.
        self.pipeline2 = deepcopy(self.pipeline)

    def prep_pipeline(self, run_durations=None, ksplits=None):
        """Prepares the pipeline by setting the backend, making the cross-validation scheme, setting up the model, making the preprocessing pipeline, kernelizing the design matrix, and completing the pipeline.

        """

        self.internalize_config(self.y, 'modeling')
        self.set_backend()
        self.make_cv(run_durations=run_durations, ksplits=ksplits)
        self.setup_model()
        self.make_preproc()
        self.kernelize()
        self.complete_pipeline()

    def make_test_dm(self, surf_data: np.ndarray):
        """_summary_
        Makes a test design matrix from surface defined data.

        Args:
            surf_data (np.ndarray): Surface defined data to make the test design matrix from.
        """

        train_roidat = copy(self.roidat)  # Copy the training roidata.
        train_data = copy(self.data)

        self.data = surf_data
        self.make_roi_data()  # Make new ROI based on the test data.

        # Make a new design matrix, ensuring that the pre-defined eigenvectors are used.
        test_dms = []
        for c, v in enumerate(self.roidat):
            test_dms.append(np.dot(v.T, self.eig[c].real))
        self.test_dm = np.hstack(test_dms)

        self.roidat = train_roidat  # Relace the roidat.
        self.data = train_data

    def fit(self, fit_data: np.ndarray):
        """fit the banded ridge model. This is the main function of the class.

        Args:
            fit_data (np.ndarray): The data to fit the model to.
        """

        self.fit_data = fit_data
        self.pipeline.fit(self.dm, self.fit_data)

    def get_params(self):

        self.xfit = self.column_kernelizer.get_X_fit()
        # Get the beta weights.
        self.betas = self.be_to_npy(
            self.pipeline[-1].get_primal_coef(self.xfit))

        # Get the split predictions (training data)
        self.Y_hat_split = self.be_to_npy(
            self.pipeline.predict(self.dm, split=True))
        # Get the split R2 (training data).
        self.split_scores = self.be_to_npy(
            r2_score_split(self.fit_data, self.Y_hat_split))
        self.best_alphas = self.be_to_npy(self.pipeline[-1].best_alphas_)

    def test_xval(self, surf_data: np.ndarray, data: np.ndarray):
        """
        Perform cross-validation on the test data using the trained model.

        Args:
            surf_data (np.ndarray): The surface-defined data from the test datset.
            data (np.ndarray): The target data for test datset.

        Returns:
            None
        """
        train_roidat = copy(self.roidat)  # Copy the training roidata.
        train_data = copy(self.data)  # Copy the training data.

        self.data = surf_data
        self.make_roi_data()  # Make new ROI data

        # Make a new design matrix, ensuring that the current eigenvectors are used.
        test_dms = []
        for c, v in enumerate(self.roidat):
            test_dms.append(np.dot(v.T, self.eig[c].real))
        test_dm = np.hstack(test_dms)

        self.xval_score = self.be_to_npy(self.pipeline.score(test_dm, data))

        self.Y_hat_split_test = self.be_to_npy(
            self.pipeline.predict(test_dm, split=True))
        self.test_split_scores = self.be_to_npy(
            r2_score_split(data, self.Y_hat_split_test))

        self.roidat = train_roidat  # Replace the roidat.
        self.data = train_data

    def test_null_model(self, surf_data: np.ndarray, data: np.ndarray):
        """_summary_

        Args:
            surf_data (np.ndarray): The surface defined data to make the null predictions from.
            data (np.ndarray): The data on which to evaluate the null predictions.
        """

        null_regressors = [np.nanmean(
            surf_data[v.subsurface_verts, :], axis=0) for c, v in enumerate(self.subsurfaces)]

        self.null_rsq = []
        for c, v in enumerate(null_regressors):
            dm = v.reshape(-1, 1)
            dm = np.column_stack((dm, np.ones(dm.shape[0])))
            betas = np.linalg.lstsq(dm, data.T)[0]
            yhat = np.dot(dm, betas).T
            self.null_rsq.append(1-(data-yhat).var(-1)/data.var(-1))
            self.frame['null_score_'+self.modalities[c]] = self.null_rsq[c]

    def reconstruct_profiles(self):
        """ recomnstructs the modality profiles by taking the dot product of the eigenvectors and the betas. 
        """
        self.modality_profiles = [np.vstack([np.dot(self.subsurfaces[c].L_eigenvectors.real, self.betas[c][:self.comps_per_hem[c]]), np.dot(
            self.subsurfaces[c].R_eigenvectors.real, self.betas[c][self.comps_per_hem[c]:])]) for c, v in enumerate(self.modalities)]

    def save_outcomes(self):
        """ save_outcomes
        Saves out beta, alpha, split scores for test and train
        """

        self.beta_names = []
        self.score_names = []
        self.Yhat_names = []

        for e, v in enumerate(self.modalities):
            self.beta_names.append(['{modality}_{x}_beta'.format(
                modality=v, x=str(x).zfill(3)) for x in range(len(self.betas[e]))])

            self.Yhat_names.append(['{modality}_{x}_Yhat'.format(modality=v, x=str(
                x).zfill(3)) for x in range(len(self.Y_hat_split_test[e]))])

            self.score_names.append('{modality}_score'.format(modality=v))
            self.saveout(self.betas[e], 'betas_{modality}'.format(
                modality=v), self.beta_names[e])
            if self.save_yhat:
                self.saveout(self.Y_hat_split_test[e], 'Yhat_{modality}'.format(
                    modality=v), self.Yhat_names[e])

        self.saveout(self.split_scores, self.train_scorename, self.score_names)
        self.saveout(self.test_split_scores,
                     self.test_scorename, self.score_names)
        self.saveout([self.best_alphas], self.alphaname, [self.alphaname])

    def be_to_npy(self, var):
        """ be_to_npy
        Converts the backend format to numpy - but also handles lists.
        """

        if type(var) == list:
            res = [self.backend.to_numpy(el) for el in var]
        else:
            res = self.backend.to_numpy(var)
        return res

    def make_mean_profiles(self):
        """ make_mean_profiles
        Makes the mean profile across eigenvectors.
        """

        self.mean_modality_profiles = [np.concatenate([np.mean(self.subsurfaces[c].L_eigenvectors.real, axis=1), np.mean(
            self.subsurfaces[c].R_eigenvectors.real, axis=1)]) for c, v in enumerate(self.modalities)]

    def regress_out(self, profiles: np.ndarray, mean_profile: np.ndarray):
        """_summary_

        Args:
            profiles (np.ndarray): The profiles for a given modality
            mean_profile (np.ndarray): The mean profile for a given modality

        Returns:
            resid (np.ndarray): The profiles after regressing out the mean profile
        """

        # Make design matrix with intercept.
        dm = np.vstack([np.ones(profiles.shape[-1]), mean_profile]).T
        # Get betas.
        betas = np.linalg.lstsq(dm, profiles.T)[0]

        # Get prediction
        yhat = np.dot(dm, betas).T

        # Get residuals
        resid = profiles-yhat

        # Add back intercept
        resid += np.mean(profiles, axis=-1)[:, np.newaxis]

        return resid

    def regress_out_mean_profiles(self):
        """Regress out the mean profiles from the modality profiles.
        """

        if not hasattr(self, 'mean_modality_profiles'):
            self.make_mean_profiles()

        self.modality_profiles = [self.regress_out(
            self.modality_profiles[c].T, self.mean_modality_profiles[c]).T for c, v in enumerate(self.modalities)]

    def splice_lookup(self, lookup_wb: np.ndarray, subsurface, modality_profiles: np.ndarray, label: str, dot_product=False, pos_only=True):
        """ Splices the lookup table with the modality profiles for a given modality.

        Args:
            lookup_wb (np.ndarray): The lookup table for the whole cortex
            subsurface (_type_): The subssurface for the given modality
            modality_profiles (np.ndarray): The modality profiles for the given modality
            label (str): The label for the given modality
            dot_product (bool, optional): Whether to splice using the peak of the profile or via dot product. Defaults to False.
            pos_only (bool, optional): Consider only positively weighted vertices in dot product. Defaults to True.

        Returns:
            _type_: _description_
        """

        lookup = lookup_wb.loc[np.concatenate(
            [subsurface.subsurface_verts_L, subsurface.subsurface_verts_R])]

        colnames = lookup.columns

        if dot_product == True:

            weights2 = np.copy(modality_profiles)  # Copy the modality profiles

            if pos_only == True:
                weights2[weights2 < 0] = 0  # Set all negative weights to zero.

            else:  # Else, raise them all so that they are >0.
                weights2 = weights2 + \
                    np.abs(np.min(weights2, axis=0))[np.newaxis, :]

            summed_weights = np.sum(weights2, axis=0)
            dp = np.dot(np.array(lookup.T), weights2)
            wav = dp/summed_weights
            spliced_lookup = pd.DataFrame(
                wav.T, columns=[s + '_'+label for s in colnames])

        else:
            peaks = np.nanargmax(modality_profiles, axis=0)
            spliced_lookup = lookup.iloc[np.array(peaks), :]
            spliced_lookup.columns = [s + '_'+label for s in colnames]

        return spliced_lookup

    def splice_lookups(self):
        """Splices all modality profiles with all lookup tables.
        """

        self.internalize_config(self.y, 'splicing')

        if self.regress_out_mean == True:
            self.regress_out_mean_profiles()

        self.spliced_lookups = [self.splice_lookup(self.lookups[c], self.subsurfaces[c], self.modality_profiles[c],
                                                   v, dot_product=self.dot_product[c], pos_only=self.pos_only[c]) for c, v in enumerate(self.modalities)]

    def save_spliced_lookups(self):
        """ save_spliced_lookups
        Saves out the spliced lookup tables.
        """

        joint_names = [self.spliced_lookups[c].columns.tolist()
                       for c, v in enumerate(self.spliced_lookups)]
        joint_names = [val for sublist in joint_names for val in sublist]
        joint_array = np.concatenate(self.spliced_lookups, axis=1).T

        self.saveout(joint_array, self.spliced_paramname, joint_names)

    def saveout(self, dlist, pname, nlist=None):
        """ saveout
        Saves out parameters to a cifti file, or a numpy file
        """

        fname = os.path.join(self.subject.out_csv,
                             '{param}.npy'.format(param=pname))
        np.save(fname, dlist)

    def load_param(self, pname: str, split=False):
        """_summary_

        Args:
            pname (str): The parameter name to load.
            split (bool, optional): Whether or not to split into surface and subcortex. Defaults to False.
        Returns:
            data (np.ndarray): The parameter data.
        """

        if self.npy == True:
            fname = os.path.join(self.subject.out_csv,
                                 self.out_npy_wildcard.format(param=pname))
            return np.load(fname)

        else:
            fname = os.path.join(self.subject.out_csv,
                                 self.out_cifti_wildcard.format(param=pname))
            bm = CiftiHandler(fname)
            bm.get_data()
            if split == True:
                split_dat = bm.decompose_data(bm.data)
                return split_dat
            else:
                return bm.data

    def load_precomputed_betas(self):
        """ load_precomputed_betas
        Loads precomputed betas into memory.
        Useful for when we want to splice a lookup table after the fitting has been performed.
        """

        self.betas = [self.load_param('betas_{modality}'.format(
            modality=v)) for e, v in enumerate(self.modalities)]

    def prepare_frame(self, with_spliced=False):
        """Prepares a pandas dataframe for saving out all outcomes.

        Args:
            with_spliced (bool, optional): Whether or not to include spliced outcomes. Defaults to False.
        """

        data = []
        labels = []
        for param in self.varsinframe:

            dat = self.load_param(param)

            if 'scores' in param:
                nlist = ['{var}_{mod}_score'.format(
                    var=param, mod=mod) for mod in self.modalities]
                for n in nlist:
                    labels.append(n)
            for d in range(dat.shape[0]):
                #    Need to log alphas
                if param == 'best_alphas':
                    data.append(np.log10(dat[d, :]))
                    labels.append(param+'_'+param)
                else:
                    data.append(dat[d, :])

        frame = pd.DataFrame(np.array(data).T, columns=labels)
        self.frame = frame.assign(
            analysis_name=self.analysis_name, subject=self.subject.subject)

        if with_spliced == True:
            joint_names = [self.spliced_lookups[c].columns.tolist()
                           for c, v in enumerate(self.spliced_lookups)]
            joint_names = ['spliced_params_' +
                           val for sublist in joint_names for val in sublist]
            joint_array = np.concatenate(self.spliced_lookups, axis=1).T

            self.frame = pd.concat([self.frame, pd.DataFrame(
                joint_array.T, columns=joint_names)], axis=1)

    def save_frame(self, with_spliced=False):
        """_summary_

        Args:
            with_spliced (bool, optional): Whether or not to save spliced outcomes. Defaults to False.
        """

        if not hasattr(self, 'frame'):
            self.prepare_frame(with_spliced)

        self.frame.to_csv(os.path.join(
            self.subject.out_csv, self.out_csv_wildcard))

    def load_frame(self):
        """Loads in the subjects data frame.
        """

        self.frame = pd.read_csv(os.path.join(
            self.subject.out_csv, self.out_csv_wildcard))
