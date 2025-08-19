import pickle

import cortex
import nibabel as nb
import numpy as np
import scipy as sp
from scipy import stats


class Subsurface(object):

    """subsurface
        This is a utility that uses pycortex for making sub-surfaces for CF fitting.
    """

    def __init__(self, cx_sub, boolmasks, surftype='fiducial'):
        """__init__
        Parameters
        ----------
        cx_sub : The name of the cx subject (string). This is used to get surfaces from the pycx database.
        boolmasks: A list of boolean arrays that define the vertices that correspond to the ROI one wants to make a subsurface from [left hem, right hem].
        surftype: The surface (default = fiducial).
        """

        self.cx_sub = cx_sub
        self.surftype = surftype
        self.boolmasks = boolmasks
        # Put the mask into int format for plotting.
        self.mask = np.concatenate(
            [self.boolmasks[0], self.boolmasks[1]]).astype(int)

    def create(self):
        """get_surfaces
        Function that creates the subsurfaces.
        """

        self.get_surfaces()
        self.generate()
        self.get_geometry()
        self.pad_distance_matrices()

    def get_surfaces(self):
        """get_surfaces
        Accesses the pycortex database to return the subject surfaces (left and right).
        Returns
        -------
        subsurface_L, subsurface_R: A pycortex subsurfaces classes for each hemisphere (These are later deleted by 'get_geometry', but can be re-created with a call to this function).
        self.subsurface_verts_L,self.subsurface_verts_R : The whole brain indices of each vertex in the subsurfaces.
        """

        self.surfaces = [cortex.polyutils.Surface(*d)
                         for d in cortex.db.get_surf(self.cx_sub, self.surftype)]

    def generate(self):
        """generate
        Use the masks defined in boolmasks to define subsurfaces.
        """

        print('Generating subsurfaces')
        # Create sub-surface, left hem.
        self.subsurface_L = self.surfaces[0].create_subsurface(
            vertex_mask=self.boolmasks[0])
        # Create sub-surface, right hem.
        self.subsurface_R = self.surfaces[1].create_subsurface(
            vertex_mask=self.boolmasks[1])

        # Get the whole-brain indices for those vertices contained in the subsurface.
        self.subsurface_verts_L = np.where(self.subsurface_L.subsurface_vertex_map != stats.mode(
            self.subsurface_L.subsurface_vertex_map)[0])[0]
        self.subsurface_verts_R = np.where(self.subsurface_R.subsurface_vertex_map != stats.mode(
            self.subsurface_R.subsurface_vertex_map)[0])[0]+self.subsurface_L.subsurface_vertex_map.shape[-1]

        self.dangling_vertex_mask_L = self.subsurface_L.subsurface_vertex_mask[
            self.boolmasks[0]]
        self.dangling_vertex_mask_R = self.subsurface_R.subsurface_vertex_mask[
            self.boolmasks[1]]

    def laplacian_decomposition(self, subsurface, n_components):
        """ laplacian_decomposition
        Parameters
        ----------
        subsurface : pycortex subsurface
        the subsurface to perform the decomposition of
        n_components : int
        n of eigenvalues/vectors to find.
        Returns
        -------
        numpy.ndarray, 2D
        The eigenvectors of the laplacian decomposition of the subsurface
        """
        B, D, W, V = subsurface.laplace_operator
        npt = W.shape[0]
        Dinv = sp.sparse.dia_matrix(
            (D**-1, [0]), (npt, npt)).tocsr()  # construct Dinv
        L = Dinv.dot((V-W))

        eigenvalues, eigenvectors = sp.sparse.linalg.eigs(
            -L, k=n_components, which="LM", sigma=0)

        return eigenvalues, eigenvectors

    def make_laplacians(self, n_components):
        self.L_eigenvalues, self.L_eigenvectors = self.laplacian_decomposition(
            self.subsurface_L, n_components)
        self.R_eigenvalues, self.R_eigenvectors = self.laplacian_decomposition(
            self.subsurface_R, n_components)

    def get_geometry(self):
        """get_geometry
        Calculates geometric info about the sub-surfaces. Computes geodesic distances from each point of the sub-surface.
        Returns
        -------
        dists_L, dists_R: Matrices of size n vertices x n vertices that describes the distances between all vertices in each hemisphere of the subsurface.
        subsurface_verts: The whole brain indices of each vertex in the subsurface.
        leftlim: The index that indicates the boundary between the left and right hemisphere. 
        """

        # Assign some variables to determine where the boundary between the hemispheres is.
        self.leftlim = np.max(self.subsurface_verts_L)
        self.subsurface_verts = np.concatenate(
            [self.subsurface_verts_L, self.subsurface_verts_R])

        # Make the distance x distance matrix.
        ldists, rdists = [], []

        print('Creating distance by distance matrices')

        for i in range(len(self.subsurface_verts_L)):
            ldists.append(self.subsurface_L.geodesic_distance([i]))
        self.dists_L = np.array(ldists)

        for i in range(len(self.subsurface_verts_R)):
            rdists.append(self.subsurface_R.geodesic_distance([i]))
        self.dists_R = np.array(rdists)

        # Get rid of these as they are harmful for pickling. We no longer need them.
        # self.surfaces, self.subsurface_L, self.subsurface_R = None, None, None

    def remove_surfaces(self):
        """remove_surfaces

        Gets rid of the subsurface objects. These cannot be pickled, so sometimes it is necessary to remove them.
        """

        self.surfaces, self.subsurface_L, self.subsurface_R = None, None, None

    def save(self, filename):
        self.remove_surfaces()
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

    def pad_distance_matrices(self, padval=np.Inf):
        """pad_distance_matrices
        Pads the distance matrices so that distances to the opposite hemisphere are np.inf
        Stack them on top of each other so they will have the same size as the design matrix
        Returns
        -------
        distance_matrix: A matrix of size n vertices x n vertices that describes the distances between all vertices in the subsurface.
        """

        # Pad the right hem with np.inf.
        padL = np.pad(
            self.dists_L, ((0, 0), (0, self.dists_R.shape[-1])), constant_values=np.Inf)
        # pad the left hem with np.inf..
        padR = np.pad(
            self.dists_R, ((0, 0), (self.dists_L.shape[-1], 0)), constant_values=np.Inf)

        self.distance_matrix = np.vstack([padL, padR])  # Now stack.

    def elaborate(self):
        """elaborate
        Prints information about the created subsurfaces.
        """

        print(
            f"Maximum distance across left subsurface: {np.max(self.dists_L)} mm")
        print(
            f"Maximum distance across right subsurface: {np.max(self.dists_R)} mm")
        print(f"Vertices in left hemisphere: {self.dists_L.shape[-1]}")
        print(f"Vertices in right hemisphere: {self.dists_R.shape[-1]}")


class CiftiHandler(object):

    """Ciftihandler
        This is a utility for loading, splitting and saving cifi data.
    """

    def __init__(self, dfile='/tank/hedger/DATA/HCP_temp/movsplit_1/tfMRI_MOVIE1_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries_sg_psc.nii'):
        self.dfile = dfile
        self.load_data()

    def load_data(self):
        """ load_data
        """

        self.img = nb.load(self.dfile)
        self.header = self.img.header
        self.brain_models = [self.header.get_axis(
            i) for i in range(self.img.ndim)][1]

    def get_data(self):
        """ get_data
        Loads the cifti data into memory.
        """

        self.load_data()
        self.data = self.img.get_fdata(dtype=np.float32)

    def surf_data_from_cifti(self, data, axis, surf_name):
        assert isinstance(axis, nb.cifti2.BrainModelAxis)
        for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
            if name == surf_name:                                 # Just looking for a surface
                # Assume brainmodels axis is last, move it to front
                data = data.T[data_indices]
                # Generally 1-N, except medial wall vertices
                vtx_indices = model.vertex
                surf_data = np.zeros(
                    (vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                surf_data[vtx_indices] = data
                return surf_data
        raise ValueError(f"No structure named {surf_name}")

    def volume_from_cifti(self, data, axis):
        assert isinstance(axis, nb.cifti2.BrainModelAxis)
        # Assume brainmodels axis is last, move it to front
        data = data.T[axis.volume_mask]
        # Which indices on this axis are for voxels?
        volmask = axis.volume_mask
        # ([x0, x1, ...], [y0, ...], [z0, ...])
        vox_indices = tuple(axis.voxel[axis.volume_mask].T)
        vol_data = np.zeros(axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
                            dtype=data.dtype)
        # "Fancy indexing"
        vol_data[vox_indices] = data
        return nb.Nifti1Image(vol_data, axis.affine)

    def decompose_cifti(self, data):
        """ decompose_cifti
        Decomposes the data in the cifti_file
        """

        self.subcortex = self.volume_from_cifti(
            data, self.brain_models)
        self.surf_left = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT")
        self.surf_right = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")
        self.surface = np.vstack([self.surf_left, self.surf_right])

        if data.ndim == 1:
            self.surface = np.concatenate([self.surf_left, self.surf_right])
        else:
            self.surface = np.vstack([self.surf_left, self.surf_right])

    def decompose_data(self, data):
        subcortex = self.volume_from_cifti(
            data, self.brain_models)
        surf_left = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT")
        surf_right = self.surf_data_from_cifti(
            data, self.brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")

        if data.ndim == 1:
            surface = np.concatenate([surf_left, surf_right])
        else:
            surface = np.vstack([surf_left, surf_right])

        return surface, subcortex

    def project_surface_into_cifti(self, dat, dtype=int):

        if not hasattr(self, 'data'):
            self.get_data()

        self.decompose_cifti(self.data)

        # ldat=dat[:self.surf_left.shape[-1]]
        # rdat=dat[self.surf_left.shape[-1]:]
        emptsub = np.zeros(self.subcortex.shape[:-1])

        empt = np.zeros(self.data.shape[-1]).astype('float')
        test_dat = np.array(range(self.data.shape[-1])).astype('float')
        split_testdat = self.decompose_data(test_dat)

        subc = emptsub[self.vox_indices]
        empt[self.vox_flat] = subc
        inds = split_testdat[0]
        empt[inds] = dat

        return empt.astype(dtype)

    def recompose_data(self, ldat, rdat, sdat, dtype=int):

        empt = np.zeros(self.data.shape[-1])
        test_dat = np.array(range(self.data.shape[-1]))
        split_testdat = self.decompose_data(test_dat)

        linds = split_testdat[0][:int((split_testdat[0].shape[-1])/2)]
        rinds = split_testdat[0][int((split_testdat[0].shape[-1])/2):]

        subc = sdat[self.vox_indices]
        empt[self.vox_flat] = subc
        empt[linds] = ldat
        empt[rinds] = rdat

        return empt.astype(dtype)

    def save_cii(self, data, filename):
        """ save_cii
        Saves data out to a cifti file, using the brainmodel.


        """
        nb.Cifti2Image(data, header=self.img.header,
                       nifti_header=self.img.nifti_header).to_filename(filename)

    def save_subvol(self, data, filename):
        """ save_subvol
        Saves subcortical data out to a nifti file. 
        """

        nb.save(data, filename)
