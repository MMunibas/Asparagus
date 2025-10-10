
import logging
from typing import Optional, Union, List, Dict, Callable, Any

import numpy as np

import torch

from asparagus import module
from asparagus import layer
from asparagus import settings
from asparagus import utils

from asparagus.layer import layer_painn

__all__ = ['Input_AMP', 'Graph_AMP', 'Output_AMP']

#======================================
# Input Module
#======================================

class Input_AMP(torch.nn.Module):
    """
    AMP input module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    input_n_atombasis: int, optional, default 128
        Number of atomic features (length of the atomic feature vector)
    input_radial_fn: (str, callable), optional, default 'GaussianRBF'
        Type of the radial basis function.
    input_n_radialbasis: int, optional, default 64
        Number of input radial basis centers
    input_cutoff_fn: (str, callable), optional, default 'Poly6'
        Cutoff function type for radial basis function scaling
    input_radial_cutoff: float, optional, default 8.0
        Cutoff distance radial basis function
    input_rbf_center_start: float, optional, default 1.0
        Lowest radial basis center distance.
    input_rbf_center_end: float, optional, default None (input_radial_cutoff)
        Highest radial basis center distance. If None, radial basis cutoff
        distance is used.
    input_rbf_trainable: bool, optional, default True
        If True, radial basis function parameter are optimized during training.
        If False, radial basis function parameter are fixed.
    input_n_maxatom: int, optional, default 94 (Plutonium)
        Highest atom order number to initialize atom feature vector library.

    """

    # Default arguments for input module
    _default_args = {
        'input_n_atombasis':            128,
        'input_radial_fn':              'SinusRBF',
        'input_n_radialbasis':          8,
        'input_cutoff_fn':              'Poly6',
        'input_radial_cutoff':          5.0,
        'input_rbf_center_start':       0.0,
        'input_rbf_center_end':         None,
        'input_rbf_trainable':          True,
        'input_n_maxatom':              94,
        'input_mlmm_radial_fn':         'SinusRBF',
        'input_mlmm_n_radialbasis':     8,
        'input_mlmm_cutoff_fn':         'Poly6',
        'input_mlmm_radial_cutoff':     9.0,
        'input_mlmm_rbf_center_start':  0.0,
        'input_mlmm_rbf_center_end':    None,
        'input_mlmm_rbf_trainable':     True,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'input_n_atombasis':            [utils.is_integer],
        'input_radial_fn':              [utils.is_string, utils.is_callable],
        'input_n_radialbasis':          [utils.is_integer],
        'input_cutoff_fn':              [utils.is_string, utils.is_callable],
        'input_radial_cutoff':          [utils.is_numeric],
        'input_rbf_center_start':       [utils.is_numeric],
        'input_rbf_center_end':         [utils.is_None, utils.is_numeric],
        'input_rbf_trainable':          [utils.is_bool],
        'input_n_maxatom':              [utils.is_integer],
        'input_mlmm_radial_fn':         [utils.is_string, utils.is_callable],
        'input_mlmm_n_radialbasis':     [utils.is_integer],
        'input_mlmm_cutoff_fn':         [utils.is_string, utils.is_callable],
        'input_mlmm_radial_cutoff':     [utils.is_numeric],
        'input_mlmm_rbf_center_start':  [utils.is_numeric],
        'input_mlmm_rbf_center_end':    [utils.is_None, utils.is_numeric],
        'input_mlmm_rbf_trainable':     [utils.is_bool],
        }

    # Input type module
    _input_type = 'AMP'

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        input_n_atombasis: Optional[int] = None,
        input_radial_fn: Optional[Union[str, object]] = None,
        input_n_radialbasis: Optional[int] = None,
        input_cutoff_fn: Optional[Union[str, object]] = None,
        input_radial_cutoff: Optional[float] = None,
        input_rbf_center_start: Optional[float] = None,
        input_rbf_center_end: Optional[float] = None,
        input_rbf_trainable: Optional[bool] = None,
        input_n_maxatom: Optional[int] = None,
        input_mlmm_radial_fn: Optional[Union[str, object]] = None,
        input_mlmm_n_radialbasis: Optional[int] = None,
        input_mlmm_cutoff_fn: Optional[Union[str, object]] = None,
        input_mlmm_radial_cutoff: Optional[float] = None,
        input_mlmm_rbf_center_start: Optional[float] = None,
        input_mlmm_rbf_center_end: Optional[float] = None,
        input_mlmm_rbf_trainable: Optional[bool] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize AMP input model.

        """

        super(Input_AMP, self).__init__()
        input_type = 'AMP'

        ####################################
        # # # Check Module Class Input # # #
        ####################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )
            
        # Update global configuration dictionary
        config.update(
            config_update,
            verbose=verbose)
        
        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)
        
        # Check general model cutoff with radial basis cutoff
        if config.get('model_cutoff') is None:
            raise ValueError(
                "No general model interaction cutoff 'model_cutoff' is yet "
                + "defined for the model calculator!")
        elif config['model_cutoff'] < self.input_radial_cutoff:
            raise ValueError(
                "The model interaction cutoff distance 'model_cutoff' "
                + f"({self.model_cutoff:.2f}) must be larger or equal "
                + "the descriptor range 'input_radial_cutoff' "
                + f"({config.get('input_radial_cutoff'):.2f})!")

        ####################################
        # # # Input Module Class Setup # # #
        ####################################
        
        # Initialize atomic feature vectors
        self.atom_features = torch.nn.Embedding(
            self.input_n_maxatom + 1,
            self.input_n_atombasis,
            padding_idx=0,
            max_norm=float(self.input_n_atombasis),
            device=self.device, 
            dtype=self.dtype)

        # Initialize radial cutoff function
        self.cutoff = layer.get_cutoff_fn(self.input_cutoff_fn)(
            self.input_radial_cutoff,
            device=self.device,
            dtype=self.dtype)
        self.mlmm_cutoff = layer.get_cutoff_fn(self.input_mlmm_cutoff_fn)(
            self.input_mlmm_radial_cutoff,
            device=self.device,
            dtype=self.dtype)
        
        # Get upper RBF center range
        if self.input_rbf_center_end is None:
            self.input_rbf_center_end = self.input_radial_cutoff
        if self.input_mlmm_rbf_center_end is None:
            self.input_mlmm_rbf_center_end = self.input_mlmm_radial_cutoff
        
        # Initialize Radial basis function
        radial_fn = layer.get_radial_fn(self.input_radial_fn)
        self.radial_fn = radial_fn(
            self.input_n_radialbasis,
            self.input_rbf_center_start,
            self.input_rbf_center_end,
            self.input_rbf_trainable, 
            self.device,
            self.dtype)
        mlmm_radial_fn = layer.get_radial_fn(self.input_mlmm_radial_fn)
        self.mlmm_radial_fn = mlmm_radial_fn(
            self.input_mlmm_n_radialbasis,
            self.input_mlmm_rbf_center_start,
            self.input_mlmm_rbf_center_end,
            self.input_mlmm_rbf_trainable, 
            self.device,
            self.dtype)

        return

    def __str__(self):
        return self._input_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'input_type': self._input_type,
            'input_n_atombasis': self.input_n_atombasis,
            'input_radial_fn': str(self.input_radial_fn),
            'input_n_radialbasis': self.input_n_radialbasis,
            'input_radial_cutoff': self.input_radial_cutoff,
            'input_cutoff_fn': str(self.input_cutoff_fn),
            'input_rbf_trainable': self.input_rbf_trainable,
            'input_n_maxatom': self.input_n_maxatom,
            }

    def forward(
        self, 
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the input module.

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors. Required and optional keys are:
            atomic_numbers : torch.Tensor(N_atoms)
                Atomic numbers of the system
            positions : torch.Tensor(N_atoms, 3)
                Atomic positions of the system
            idx_i : torch.Tensor(N_pairs)
                Atom i pair index
            idx_j : torch.Tensor(N_pairs)
                Atom j pair index
            pbc_offset_ij : torch.Tensor(N_pairs, 3), optional, default None
                Position offset from periodic boundary condition
            idx_u : torch.Tensor(N_pairs), optional, default None
                Long-range atom u pair index
            idx_v : torch.Tensor(N_pairs), optional, default None
                Long-range atom v pair index
            pbc_offset_uv : torch.Tensor(N_pairs, 3), optional, default None
                Long-range position offset from periodic boundary condition
            fragment_numbers : torch.Tensor(N_atoms)
                Atomic fragment index number
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary added by module results:
            features: torch.tensor(N_atoms, n_atombasis)
                Atomic feature vectors
            distances: torch.tensor(N_pairs)
                Atom pair distances
            vectors: torch.tensor(N_pairs, 3)
                Atom pair vectors
            cutoffs: torch.tensor(N_pairs)
                Atom pair distance cutoffs
            rbfs: torch.tensor(N_pairs, n_radialbasis)
                Atom pair radial basis functions
            distances_uv: torch.tensor(N_pairs_uv)
                Long-range atom pair distances
            vectors_uv: torch.tensor(N_pairs_uv, 3)
                Long-range atom pair vectors

        """

        # Assign ML-ML input data
        positions = batch['positions']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']

        # ML/MM approach - Select only ML atom inputs
        if 'ml_idx' in batch:
            batch['atomic_numbers'] = batch['atomic_numbers'][batch['ml_idx']]
            batch['sys_i'] = batch['sys_i'][batch['ml_idx']]
            batch['atoms_number'] = torch.bincount(batch['sys_i'])

        # Collect ML atom feature vectors
        features = self.atom_features(batch['atomic_numbers'])

        # Compute ML atom pair connection vectors
        if 'pbc_offset_ij' in batch:
            vectors = (
                positions[idx_j] - positions[idx_i] + batch['pbc_offset_ij'])
        else:
            vectors = positions[idx_j] - positions[idx_i]

        # Compute ML atom pair distances
        distances = torch.norm(vectors, dim=-1)

        # Compute ML atom pair vectors outer product
        outer_product = vectors.unsqueeze(-1)*vectors.unsqueeze(-2)

        # Compute ML atom pair distance cutoff values
        cutoffs = self.cutoff(distances)

        # Compute ML atom pair radial basis functions
        rbfs = self.radial_fn(distances)

        # ML/MM approach - Point ML atom pair indices from full ML/MM system to
        # the ML system indices (ML/MM index number of, e.g., 41 for the first
        # ML atom in the ML/MM system becomes index 0 for the ML system).
        if 'ml_idx_p' in batch:

            batch['idx_i'] = batch['ml_idx_p'][idx_i]
            batch['idx_j'] = batch['ml_idx_p'][idx_j]

            # PBC supercluster approach - Point from ML atom pair index j of
            # atoms in the image cell to the respective primary cell ML
            # atom index.
            if 'ml_idx_jp' in batch:
                batch['idx_j'] = batch['ml_idx_p'][batch['ml_idx_jp']]

        # Compute distances and vectors for long-range ML atom pairs
        if 'idx_u' in batch:

            # Assign long-range ML-ML input data
            idx_u = batch['idx_u']
            idx_v = batch['idx_v']

            # Compute long-range ML atom pair connection vectors
            if 'pbc_offset_uv' in batch:
                vectors_uv = (
                    positions[idx_v] - positions[idx_u]
                    + batch['pbc_offset_uv'])
            else:
                vectors_uv = positions[idx_v] - positions[idx_u]

            # Compute long-range ML atom pair distances
            distances_uv = torch.norm(vectors_uv, dim=-1)
            
            # Compute long-range ML atom pair vectors outer product
            outer_product_uv = (
                vectors_uv.unsqueeze(-1)*vectors_uv.unsqueeze(-2))

            # ML/MM approach - Point ML atom pair indices from full ML/MM
            # system to the ML system indices 
            if 'ml_idx_p' in batch:
                batch['idx_u'] = batch['ml_idx_p'][idx_u]
                batch['idx_v'] = batch['ml_idx_p'][idx_v]
                
                # PBC supercluster approach - Point from ML atom pair index j
                # of atoms in the image cell to the respective primary cell
                # ML atom index.
                if 'ml_idx_vp' in batch:
                    batch['idx_v'] = batch['ml_idx_p'][batch['ml_idx_vp']]

        else:
            
            # Assign ML atom pair results as long-range pair results
            batch['idx_u'] = batch['idx_i']
            batch['idx_v'] = batch['idx_j']
            vectors_uv = vectors
            distances_uv = distances
            outer_product_uv = outer_product

        # Assign ML atom result data
        batch['features'] = features
        batch['distances'] = distances
        batch['vectors'] = vectors
        batch['outer_product'] = outer_product
        batch['cutoffs'] = cutoffs
        batch['rbfs'] = rbfs
        batch['distances_uv'] = distances_uv
        batch['vectors_uv'] = vectors_uv
        batch['outer_product_uv'] = outer_product_uv

        # Assign ML-MM input data for ML charge polarization
        mlmm_idx_i = batch['mlmm_idx_i']
        mlmm_idx_j = batch['mlmm_idx_j']

        # Compute ML-MM atom pair connection vectors
        if 'mlmm_pbc_offset_ij' in batch:
            mlmm_vectors = (
                positions[mlmm_idx_j] - positions[mlmm_idx_i]
                + batch['mlmm_pbc_offset_ij'])
        else:
            mlmm_vectors = positions[mlmm_idx_j] - positions[mlmm_idx_i]

        # Compute ML-MM atom pair distances
        mlmm_distances = torch.norm(mlmm_vectors, dim=-1)

        # Compute ML-MM atom pair vectors outer product
        mlmm_outer_product = (
            mlmm_vectors.unsqueeze(-1)*mlmm_vectors.unsqueeze(-2))

        # Compute ML-MM atom pair distance cutoff values
        mlmm_cutoffs = self.cutoff(mlmm_distances)

        # Compute ML-MM atom pair radial basis functions
        mlmm_rbfs = self.mlmm_radial_fn(mlmm_distances)

        # ML/MM approach - Point ML/MM atom pair indices i (ML atom) from full
        # ML/MM system to the ML system indices and store as additional entry.
        if 'ml_idx_p' in batch:
            batch['mlmm_idx_i'] = batch['ml_idx_p'][mlmm_idx_i]

        # PBC supercluster approach - Point from ML/MM atom pair index j 
        # (MM atom) of atoms in the image cell to the respective primary cell
        # MM atom index, while keeping structure data to image atom.
        if 'mlmm_idx_jp' in batch:
            batch['mlmm_idx_j'] = batch['mlmm_idx_jp']

        # Compute distances and vectors for long-range ML-MM atom pairs
        if 'mlmm_idx_u' in batch:

            # Assign long-range ML-ML input data
            mlmm_idx_u = batch['mlmm_idx_u']
            mlmm_idx_v = batch['mlmm_idx_v']

            # Compute long-range ML atom pair connection vectors
            if 'mlmm_pbc_offset_uv' in batch:
                mlmm_vectors_uv = (
                    positions[mlmm_idx_v] - positions[mlmm_idx_u]
                    + batch['mlmm_pbc_offset_uv'])
            else:
                mlmm_vectors_uv = positions[mlmm_idx_v] - positions[mlmm_idx_u]
            
            # Compute long-range  ML-MM atom pair distances
            mlmm_distances_uv = torch.norm(mlmm_vectors_uv, dim=-1)

            # Compute long-range ML-MM atom pair vectors outer product
            mlmm_outer_product_uv = (
                mlmm_vectors_uv.unsqueeze(-1)*mlmm_vectors_uv.unsqueeze(-2))
            
            # ML/MM approach - Point ML/MM atom pair indices u (ML atom) from
            # full ML/MM system to the ML system indices and store as 
            # additional entry.
            if 'ml_idx_p' in batch:
                batch['mlmm_idx_u'] = batch['ml_idx_p'][mlmm_idx_u]

            # PBC supercluster approach - Point from ML/MM atom pair index v 
            # (MM atom) of atoms in the image cell to the respective primary
            # cell MM atom index, while keeping structure data to image atom.
            if 'mlmm_idx_vp' in batch:
                batch['mlmm_idx_v'] = batch['mlmm_idx_vp']

        else:
            
            # Assign ML atom pair results as long-range pair results
            batch['mlmm_idx_u'] = batch['mlmm_idx_i']
            batch['mlmm_idx_v'] = batch['mlmm_idx_j']
            mlmm_vectors_uv = mlmm_vectors
            mlmm_distances_uv = mlmm_distances
            mlmm_outer_product_uv = mlmm_outer_product

        # Assign ML-MM atom pair result data
        batch['mlmm_distances'] = mlmm_distances
        batch['mlmm_vectors'] = mlmm_vectors
        batch['mlmm_outer_product'] = mlmm_outer_product
        batch['mlmm_cutoffs'] = mlmm_cutoffs
        batch['mlmm_rbfs'] = mlmm_rbfs
        batch['mlmm_distances_uv'] = mlmm_distances_uv
        batch['mlmm_vectors_uv'] = mlmm_vectors_uv
        batch['mlmm_outer_product_uv'] = mlmm_outer_product_uv

        return batch


#======================================
# Graph Module
#======================================

class Graph_AMP(torch.nn.Module): 
    """
    AMP message passing module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    graph_n_blocks: int, optional, default 2
        Number of information processing cycles
    graph_n_messagebasis: int, optional, default 32
        Size of message vector for each degree of atomic multipole
    graph_activation_fn: (str, object), optional, default 'silu'
        Activation function

    """
    
    # Default arguments for graph module
    _default_args = {
        'graph_n_blocks':               2,
        'graph_n_messagebasis':         32,
        'graph_activation_fn':          'silu',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'graph_n_blocks':               [utils.is_integer],
        'graph_n_messagebasis':         [utils.is_integer],
        'graph_activation_fn':          [utils.is_string, utils.is_callable],
        }

    # Graph type module
    _graph_type = 'AMP'

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        graph_n_blocks: Optional[int] = None,
        graph_n_messagebasis: Optional[int] = None,
        graph_activation_fn: Optional[Union[str, object]] = None,
        graph_stability_constant: Optional[float] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize NNP graph model.

        """
        
        super(Graph_AMP, self).__init__()
        graph_type = 'AMP'
        
        ####################################
        # # # Check Module Class Input # # #
        ####################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )

        # Update global configuration dictionary
        config.update(
            config_update,
            verbose=verbose)

        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)

        # Get input to graph module interface parameters 
        self.n_atombasis = config.get('input_n_atombasis')
        self.n_radialbasis = config.get('input_n_radialbasis')

        # Get maximum degree of atomic multipoles
        if config.get('model_properties') is None:
            raise SyntaxError(
                "Model properties must be defined when initializing a Graph "
                + "module!")
        if 'atomic_quadrupoles' in config['model_properties']:
            self.graph_atomic_quadrupoles = torch.tensor(
                True, device=self.device, dtype=torch.bool)
            self.graph_atomic_dipoles = torch.tensor(
                True, device=self.device, dtype=torch.bool)
            self.graph_atomic_charges = torch.tensor(
                True, device=self.device, dtype=torch.bool)
            self.graph_max_multipole_order = 2
            n_features_anisotropic = 11
        elif 'atomic_dipoles' in config['model_properties']:
            self.graph_atomic_quadrupoles = torch.tensor(
                False, device=self.device, dtype=torch.bool)
            self.graph_atomic_dipoles = torch.tensor(
                True, device=self.device, dtype=torch.bool)
            self.graph_atomic_charges = torch.tensor(
                True, device=self.device, dtype=torch.bool)
            self.graph_max_multipole_order = 1
            n_features_anisotropic = 5
        elif 'atomic_charges' in config['model_properties']:
            self.graph_atomic_quadrupoles = torch.tensor(
                False, device=self.device, dtype=torch.bool)
            self.graph_atomic_dipoles = torch.tensor(
                False, device=self.device, dtype=torch.bool)
            self.graph_atomic_charges = torch.tensor(
                True, device=self.device, dtype=torch.bool)
            self.graph_max_multipole_order = 0
            n_features_anisotropic = 2
        else:
            raise SyntaxError(
                "Graph module of AMP require at least on electrostatic atomic "
                "mono- or multipole!")

        ####################################
        # # # Graph Module Class Setup # # #
        ####################################

        # Initialize activation function
        self.activation_fn = layer.get_activation_fn(
            self.graph_activation_fn)

        # Initialize embedding layers
        self.embedding_rbfs = layer.DenseLayer(
            (self.n_radialbasis + 2*self.n_atombasis),
            (self.n_atombasis//4),
            None,
            False,
            self.device,
            self.dtype)
        self.embedding_features = layer.DenseLayer(
            self.n_atombasis,
            self.n_atombasis,
            None,
            False,
            self.device,
            self.dtype)

        # Initialize message passing and feature vector update blocks
        n_input = (
            [self.n_atombasis//4 + 2*self.n_atombasis]
            + (self.graph_n_blocks - 1)
            * [
                (n_features_anisotropic + 1)*(self.n_atombasis//4)
                + 2*self.n_atombasis
            ])
        self.equivariant_message_block = torch.nn.ModuleList([
            torch.nn.Sequential(
                layer.DenseLayer(
                    n_input[ii],
                    self.n_atombasis,
                    self.activation_fn,
                    True,
                    self.device,
                    self.dtype,
                ),
                layer.DenseLayer(
                    self.n_atombasis,
                    self.graph_n_messagebasis*(
                        self.graph_max_multipole_order + 1),
                    None,
                    False,
                    self.device,
                    self.dtype,
                    weight_init=torch.nn.init.xavier_normal_,
                    kwargs_weight_init={'gain': 1.E-2},
                ),
            )
            for ii in range(self.graph_n_blocks)
        ])
        self.invariant_message_block = torch.nn.ModuleList([
            torch.nn.Sequential(
                layer.DenseLayer(
                    (
                        (n_features_anisotropic + 1)*(self.n_atombasis//4)
                        + 2*self.n_atombasis
                    ),
                    self.n_atombasis,
                    self.activation_fn,
                    True,
                    self.device,
                    self.dtype,
                ),
                layer.DenseLayer(
                    self.n_atombasis,
                    self.n_atombasis,
                    self.activation_fn,
                    True,
                    self.device,
                    self.dtype,
                    weight_init=torch.nn.init.xavier_normal_,
                    kwargs_weight_init={'gain': 1.E-2},
                ),
            )
            for _ in range(self.graph_n_blocks)                
        ])
        self.invariant_update_block = torch.nn.ModuleList([
            torch.nn.Sequential(
                layer.DenseLayer(
                    2*self.n_atombasis,
                    self.n_atombasis,
                    self.activation_fn,
                    True,
                    self.device,
                    self.dtype,
                ),
                layer.DenseLayer(
                    self.n_atombasis,
                    self.n_atombasis,
                    self.activation_fn,
                    True,
                    self.device,
                    self.dtype,
                    weight_init=torch.nn.init.xavier_normal_,
                    kwargs_weight_init={'gain': 1.E-2},
                ),
            )
            for _ in range(self.graph_n_blocks)                
        ])

        # Initialize ML atom polarization factors prediction
        self.ml_alpha_block = torch.nn.Sequential(
            layer.DenseLayer(
                self.n_atombasis,
                self.n_atombasis//2,
                self.activation_fn,
                True,
                self.device,
                self.dtype,
            ),
            layer.DenseLayer(
                self.n_atombasis//2,
                self.graph_n_messagebasis*self.graph_max_multipole_order,
                torch.nn.Softplus(),
                False,
                self.device,
                self.dtype,
                weight_init=torch.nn.init.xavier_normal_,
                kwargs_weight_init={'gain': 1.E-2},
            )
        )

        # initialize MM electric field coefficients prediction
        self.mm_b_coefficient_block = torch.nn.Sequential(
            layer.DenseLayer(
                self.graph_max_multipole_order + self.n_radialbasis,
                8,
                self.activation_fn,
                True,
                self.device,
                self.dtype,
            ),
            layer.DenseLayer(
                8,
                self.graph_max_multipole_order,
                None,
                False,
                self.device,
                self.dtype,
                weight_init=torch.nn.init.xavier_normal_,
                kwargs_weight_init={'gain': 1.E-2},
            )
        )

        self.ml_coefficients = torch.nn.Sequential(
            layer.DenseLayer(
                (
                    (n_features_anisotropic + 1)*(self.n_atombasis//4)
                    + 2*self.n_atombasis
                ),
                self.n_atombasis,
                self.activation_fn,
                True,
                self.device,
                self.dtype,
            ),
            layer.DenseLayer(
                self.n_atombasis,
                self.graph_max_multipole_order + 1,
                None,
                False,
                self.device,
                self.dtype,
                weight_init=torch.nn.init.xavier_normal_,
                kwargs_weight_init={'gain': 1.E-2},
            ),
        )

        return

    def __str__(self):
        return self._graph_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'graph_type': self._graph_type,
            'graph_n_blocks': self.graph_n_blocks,
            'graph_n_messagebasis': self.graph_n_messagebasis,
            'graph_activation_fn': str(self.graph_activation_fn),
            }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of the graph model.
        
        Parameter
        ---------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors. Required and optional keys are:
            features: torch.tensor(N_atoms, n_atombasis)
                Atomic feature vectors
            distances: torch.tensor(N_pairs)
                Atom pair distances
            vectors : torch.Tensor
                Atom pair connection vectors
            cutoffs: torch.tensor(N_pairs)
                Atom pair distance cutoffs
            rbfs: torch.tensor(N_pairs, n_radialbasis)
                Atom pair radial basis functions
            idx_i : torch.Tensor(N_pairs)
                Atom i pair index
            idx_j : torch.Tensor(N_pairs)
                Atom j pair index
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary added by module results:
            sfeatures: torch.tensor(N_atoms, n_atombasis)
                Modified scalar atomic feature vectors
            vfeatures: torch.tensor(N_atoms, 3, n_atombasis)
                Modified vector atomic feature vectors

        """

        # Assign input data
        atomic_numbers = batch['atomic_numbers']
        features = batch['features']
        distances = batch['distances']
        vectors = batch['vectors']
        outer_product = batch['outer_product']
        cutoffs = batch['cutoffs']
        rbfs = batch['rbfs']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        
        mlmm_distances = batch['mlmm_distances']
        mlmm_vectors = batch['mlmm_vectors']
        mlmm_outer_product = batch['mlmm_outer_product']
        mlmm_idx_i = batch['mlmm_idx_i']
        mlmm_idx_j = batch['mlmm_idx_j']

        # Initialize electrostatic atomic multipoles and case dependent
        # auxiliaries
        if self.graph_atomic_quadrupoles:
            batch['atomic_quadrupoles'] = torch.zeros(
                (atomic_numbers.size(0), self.graph_n_messagebasis, 3, 3, ),
                device=self.device,
                dtype=self.dtype)
            batch['pol_atomic_quadrupoles'] = torch.zeros(
                (atomic_numbers.size(0), 3, 3, ),
                device=self.device,
                dtype=self.dtype)
            traceless_outer_product = (
                3*outer_product
                - torch.diag_embed(
                    torch.tile(
                        outer_product.diagonal(
                            dim1=-2, dim2=-1).sum(
                                dim=-1, keepdim=True),
                        (1, 3)
                    )
                )
            )
            mlmm_traceless_outer_product = (
                3*mlmm_outer_product
                - torch.diag_embed(
                    torch.tile(
                        mlmm_outer_product.diagonal(
                            dim1=-2, dim2=-1).sum(
                                dim=-1, keepdim=True),
                        (1, 3)
                    )
                )
            )
        if self.graph_atomic_dipoles:
            batch['atomic_dipoles'] = torch.zeros(
                (atomic_numbers.size(0), self.graph_n_messagebasis, 3, ),
                device=self.device,
                dtype=self.dtype)
            batch['pol_atomic_dipoles'] = torch.zeros(
                (atomic_numbers.size(0), 3, ),
                device=self.device,
                dtype=self.dtype)
        if self.graph_atomic_charges:
            batch['atomic_charges'] = torch.zeros(
                (atomic_numbers.size(0), self.graph_n_messagebasis, 1, ),
                device=self.device,
                dtype=self.dtype)

        # Combine atom pair information: RBFs, features i, features j
        rbfs_features = torch.cat(
            (rbfs, features[idx_i], features[idx_j]),
            dim=-1
        )

        # Apply one-hot embedding of the interacting features
        embedded_rbfs = self.embedding_rbfs(rbfs_features)
        rbfs_features = embedded_rbfs.clone().detach() #.clone().detach() #?
        features = self.embedding_features(features)

        # Apply message passing model
        for ii, (eq_message, in_message, in_update) in enumerate(
            zip(
                self.equivariant_message_block,
                self.invariant_message_block,
                self.invariant_update_block
            )
        ):

            # Combine atom pair features and embedded features
            features_pairs = torch.cat(
                (features[idx_i], features[idx_j]),
                dim=-1
            )
            rbfs_features_i = torch.cat(
                (features_pairs, rbfs_features),
                dim=-1
            )

            # Apply equivariant message network to multipole prediction
            coefficients = (
                eq_message(rbfs_features_i)*cutoffs.unsqueeze(-1)
                ).tensor_split(
                    self.graph_max_multipole_order + 1,
                    dim=-1)

            # Predict electrostatic multipoles
            if self.graph_atomic_quadrupoles:

                coeffs = (
                    coefficients[2].unsqueeze(-1).unsqueeze(-1)
                    * traceless_outer_product.unsqueeze(1))
                idx_ic = (
                    idx_i.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                        coeffs.size()))
                batch['atomic_quadrupoles'] = (
                    batch['atomic_quadrupoles'].scatter_add_(
                        0, idx_ic, coeffs)
                    )

            if self.graph_atomic_dipoles:

                coeffs = coefficients[1].unsqueeze(-1)*vectors.unsqueeze(1)
                idx_ic = (
                    idx_i.unsqueeze(-1).unsqueeze(-1).expand(
                        coeffs.size()))
                batch['atomic_dipoles'] = batch['atomic_dipoles'].scatter_add_(
                    0, idx_ic, coeffs)

            if self.graph_atomic_charges:

                coeffs = coefficients[0].unsqueeze(-1)
                idx_ic = (
                    idx_i.unsqueeze(-1).unsqueeze(-1).expand(
                        coeffs.size()))
                batch['atomic_charges'] = batch['atomic_charges'].scatter_add_(
                    0, idx_ic, coeffs)

            # Predict ML electrostatic atomic multipole polarization by the
            # MM atom charges duirng the last message passing step
            if (
                self.graph_max_multipole_order
                and ii == (self.graph_n_blocks - 1) 
            ):

                # Grep ML electrostatic atomic multipoles
                ml_atomic_dipoles = batch['atomic_dipoles'][mlmm_idx_i, 0:1]
                if self.graph_atomic_quadrupoles:
                    ml_atomic_quadrupoles = (
                        batch['atomic_quadrupoles'][mlmm_idx_i, 0:1])

                # Compute auxiliary variables
                dipoles_vectors = torch.sum(
                    ml_atomic_dipoles*mlmm_vectors.unsqueeze(1),
                    dim=-1,
                    keepdim=True)
                if self.graph_atomic_quadrupoles:
                    quadrupoles_outer_product = torch.sum(
                        torch.sum(
                            (
                                ml_atomic_quadrupoles
                                * mlmm_traceless_outer_product.unsqueeze(1)
                            ),
                            dim=-1),
                        dim=-1).unsqueeze(-1)
                
                # Combine auxiliaries to a MM atom stuructural feature variable
                if self.graph_atomic_quadrupoles:
                    mlmm_features = torch.cat(
                        (dipoles_vectors, quadrupoles_outer_product),
                        dim=-1).squeeze(1)
                    mlmm_rbfs_features = torch.cat(
                        (mlmm_features, batch['mlmm_rbfs']),
                        dim=-1)
                else:
                    mlmm_rbfs_features = torch.cat(
                        (
                            dipoles_vectors.squeeze(-1).squeeze(1),
                            batch['mlmm_rbfs']
                        ),
                        dim=-1)

                # Compute polarization factors
                ml_atomic_alphas = self.ml_alpha_block(features)
                
                # Compute MM electric field coefficients
                mm_b_coefficients = self.mm_b_coefficient_block(
                    mlmm_rbfs_features)

                # Compute MM electric field coefficients
                mm_atomic_charges_j = (
                    batch['reference']['atomic_charges'][batch['mlmm_idx_j']])
                mm_electric_field = (
                    mm_atomic_charges_j/mlmm_distances**2).unsqueeze(-1)
                mm_coefficients = mm_b_coefficients*mm_electric_field
                
                # Compute electrostatic multipole polarization correction
                coeff_pol1 = (
                    mm_coefficients[:, 0].unsqueeze(-1)*mlmm_vectors)
                idx_pol1 = (
                    mlmm_idx_i.unsqueeze(-1).expand(
                        coeff_pol1.size()))
                batch['pol_atomic_dipoles'] = (
                    ml_atomic_alphas[:, 0].unsqueeze(-1)
                    * batch['pol_atomic_dipoles'].scatter_add_(
                        0, idx_pol1, coeff_pol1
                    )
                )
                if self.graph_atomic_quadrupoles:
                    coeff_pol2 = (
                        mm_coefficients[:, 1].unsqueeze(-1).unsqueeze(-1)
                        * mlmm_traceless_outer_product)
                    idx_pol2 = (
                        mlmm_idx_i.unsqueeze(-1).unsqueeze(-1).expand(
                            coeff_pol2.size()))
                    batch['pol_atomic_quadrupoles'] = (
                        ml_atomic_alphas[:, 1].unsqueeze(-1).unsqueeze(-1)
                        * batch['pol_atomic_quadrupoles'].scatter_add_(
                            0, idx_pol2, coeff_pol2
                        )
                )

                # Add polarization correction
                batch['atomic_dipoles'] = (
                    batch['atomic_dipoles']
                    + batch['pol_atomic_dipoles'].unsqueeze(1))
                if self.graph_atomic_quadrupoles:
                    batch['atomic_quadrupoles'] = (
                        batch['atomic_quadrupoles']
                        + batch['pol_atomic_quadrupoles'].unsqueeze(1))

            # Construct anisotropic feature vectors
            
            # If atomic charges are predicted
            if self.graph_atomic_charges:
                
                # Grep atomic charges for ML atom pairs
                atomic_charges_i = batch['atomic_charges'][batch['idx_i']]
                atomic_charges_j = batch['atomic_charges'][batch['idx_j']]
            
            # If atomic dipoles are predicted
            if self.graph_atomic_dipoles:
                
                # Grep atomic dipoles for ML atom pairs
                atomic_dipoles_i = batch['atomic_dipoles'][batch['idx_i']]
                atomic_dipoles_j = batch['atomic_dipoles'][batch['idx_j']]

                # Copute scalar products of atomic dipoles and atom pair
                # connection vectors
                atomic_dipoles_vectors_i = torch.sum(
                    atomic_dipoles_i*vectors.unsqueeze(1),
                    dim=-1,
                    keepdim=True)
                atomic_dipoles_vectors_j = torch.sum(
                    atomic_dipoles_j*vectors.unsqueeze(1),
                    dim=-1,
                    keepdim=True)

                # Compute atomic dipole-dipole interaction
                atomic_dipoles_dipoles_ij = torch.sum(
                    atomic_dipoles_i*atomic_dipoles_j, dim=-1, keepdim=True)

            # If atomic quadrupoles are predicted
            if self.graph_atomic_quadrupoles:

                # Grep atomic quadrupoles for ML atom pairs
                atomic_quadrupoles_i = (
                    batch['atomic_quadrupoles'][batch['idx_i']])
                atomic_quadrupoles_j = (
                    batch['atomic_quadrupoles'][batch['idx_j']])

                # Copute matrix products of atomic quadrupoles and atom pair
                # connection vectors
                atomic_quadrupoles_vectors_i = torch.sum(
                    atomic_quadrupoles_i*vectors.unsqueeze(1).unsqueeze(2),
                    dim=-1)
                atomic_quadrupoles_vectors_j = torch.sum(
                    atomic_quadrupoles_j*vectors.unsqueeze(1).unsqueeze(2),
                    dim=-1)
                
                # Copute matrix products of atomic quadrupoles and atom pair
                # detraced outer product
                atomic_quadrupoles_outer_product_i = (
                    torch.sum(
                        torch.sum(
                            atomic_quadrupoles_i
                            * traceless_outer_product.unsqueeze(1),
                            dim=-1),
                        dim=-1,
                        keepdim=True,
                    )
                )
                atomic_quadrupoles_outer_product_j = (
                    torch.sum(
                        torch.sum(
                            atomic_quadrupoles_j
                            * traceless_outer_product.unsqueeze(1),
                            dim=-1),
                        dim=-1,
                        keepdim=True,
                    )
                )

                # Compute atomic quadrupole-dipole interaction
                atomic_quadrupoles_dipoles_ij = torch.sum(
                    atomic_quadrupoles_vectors_i*atomic_dipoles_j,
                    dim=-1,
                    keepdim=True,
                )
                atomic_quadrupoles_dipoles_ji = torch.sum(
                    atomic_quadrupoles_vectors_j*atomic_dipoles_i,
                    dim=-1,
                    keepdim=True,
                )
                
                # Compute atomic quadrupole-quadrupole interaction
                atomic_quadrupoles_quadrupoles_ij = (
                    torch.sum(
                        torch.sum(
                            atomic_quadrupoles_i*atomic_quadrupoles_j,
                            dim=-1),
                        dim=-1,
                        keepdim=True,
                    )
                )

                # Compute atom pair quadrupole-vector product
                atomic_quadrupoles_vectors_ij = torch.sum(
                    atomic_quadrupoles_vectors_i*atomic_quadrupoles_vectors_j,
                    dim=-1,
                    keepdim=True)

            # Combine anisotropic features
            if self.graph_atomic_quadrupoles:
                features_anisotropic = torch.cat(
                    (
                        atomic_charges_i,
                        atomic_charges_j,
                        atomic_dipoles_vectors_i,
                        atomic_dipoles_vectors_j,
                        atomic_dipoles_dipoles_ij,
                        atomic_quadrupoles_outer_product_i,
                        atomic_quadrupoles_outer_product_j,
                        atomic_quadrupoles_dipoles_ij,
                        atomic_quadrupoles_dipoles_ji,
                        atomic_quadrupoles_quadrupoles_ij,
                        atomic_quadrupoles_vectors_ij,
                    ),
                    dim=-1,
                ).reshape(idx_i.size(0), -1)
            elif self.graph_atomic_dipoles:
                features_anisotropic = torch.cat(
                    (
                        atomic_charges_i,
                        atomic_charges_j,
                        atomic_dipoles_vectors_i,
                        atomic_dipoles_vectors_j,
                        atomic_dipoles_dipoles_ij,
                    ),
                    dim=-1,
                ).reshape(idx_i.size(0), -1)
            elif self.graph_atomic_charges:
                features_anisotropic = torch.cat(
                    (
                        atomic_charges_i,
                        atomic_charges_j,
                    ),
                    dim=-1,
                ).reshape(idx_i.size(0), -1)

            # Combine anisotropic features with RBFs as updated atom pair info
            rbfs_features = torch.cat(
                (
                    features_anisotropic,
                    embedded_rbfs
                ),
                dim=-1)

            # Combine ML atom pair feature vectors with anisotropic features
            # and RBFs
            message = torch.cat(
                (
                    features_pairs,
                    rbfs_features
                ),
                dim=-1)

            # Apply invariant message network and weight message vector with
            # distane cutoffs
            message = torch.zeros(
                (atomic_numbers.size(0), self.n_atombasis),
                device=self.device,
                dtype=self.dtype,
                ).scatter_add_(
                    0, idx_i.unsqueeze(-1).repeat(1, self.n_atombasis),
                    in_message(message)*cutoffs.unsqueeze(-1)
                )

            # Apply invariant update network to combined feature and message 
            # vector and add message to the atomic feature vectors
            features = (
                features 
                + in_update(
                    torch.cat((features, message), dim=-1)
                )
            )

        # Final prediction of electrostatic multipoles

        # Combine atom pair features and ML embedded features
        rbfs_features_last = torch.cat(
            (features[idx_i], features[idx_j], rbfs_features),
            dim=-1
        )

        # Apply last equivariant message network to ML multipole prediction
        coefficients_last = (
            self.ml_coefficients(rbfs_features_last)*cutoffs.unsqueeze(-1)
            ).tensor_split(
                self.graph_max_multipole_order + 1,
                dim=-1)

        # Predict MM polarized ML electrostatic multipoles
        if self.graph_atomic_quadrupoles:

            coeffs = (
                coefficients_last[2].unsqueeze(-1).unsqueeze(-1)
                * traceless_outer_product.unsqueeze(1))
            idx_ic = (
                idx_i.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    coeffs.size()))
            batch['atomic_quadrupoles'] = (
                batch['atomic_quadrupoles'].scatter_add_(
                    0, idx_ic, coeffs)
                )[:, 0]
            batch['atomic_quadrupoles'] = (
                batch['atomic_quadrupoles'] + batch['pol_atomic_quadrupoles'])

        if self.graph_atomic_dipoles:

            coeffs = coefficients_last[1].unsqueeze(-1)*vectors.unsqueeze(1)
            idx_ic = (
                idx_i.unsqueeze(-1).unsqueeze(-1).expand(
                    coeffs.size()))
            batch['atomic_dipoles'] = batch['atomic_dipoles'].scatter_add_(
                0, idx_ic, coeffs)[:, 0]
            batch['atomic_dipoles'] = (
                batch['atomic_dipoles'] + batch['pol_atomic_dipoles'])

        if self.graph_atomic_charges:

            coeffs = coefficients_last[0].unsqueeze(-1)
            idx_ic = (
                idx_i.unsqueeze(-1).unsqueeze(-1).expand(
                    coeffs.size()))
            batch['atomic_charges'] = batch['atomic_charges'].scatter_add_(
                0, idx_ic, coeffs)[:, 0].squeeze(-1)

        # Assign result data
        batch['features'] = features
        # batch['embedded_rbfs'] = embedded_rbfs
        # batch['rbfs_features'] = rbfs_features

        return batch

#======================================
# Output Module
#======================================

class Output_AMP(torch.nn.Module): 
    """
    AMP output module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    output_properties: list(str), optional None
        List of output properties to compute by the model
        e.g. ['energy', 'forces', 'atomic_charges']
    output_properties_options: dict(str, Any), optional, default {}
        Dictionary of output block options (item) for a property (key).
        The defined property ouptut block options will update the default
        output block options.
        Dictionary inputs are, e.g.:
        output_properties_options = {
            'atomic_energies': { # Same options as the default scalar ones
                'output_type':          'scalar',
                'n_property':           1,
                'n_layer':              2,
                'n_neurons':            None,
                'activation_fn':        'silu',
                'bias_layer':           True,
                'bias_last':            True,
                'weight_init_layer':    'xavier_uniform_',
                'weight_init_last':     'xavier_uniform_',
                'bias_init_layer':      'zeros_',
                'bias_init_last':       'zeros_',
                },
            }
    output_n_residual: int, optional, default 1
        Number of residual layers for transformation from atomic feature vector
        to output results.
    output_property_scaling: dictionary, optional, default None
        Property average and standard deviation for the use as scaling factor 
        (standard deviation) and shift term (average) parameter pairs
        for each property.
    **kwargs: dict, optional
        Additional arguments

    """

    # Default arguments for graph module
    _default_args = {
        'output_properties':            None,
        'output_properties_options':    {},
        'output_n_residual':            1,
        'output_property_scaling':      None,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'output_properties':            [utils.is_string_array, utils.is_None],
        'output_properties_options':    [utils.is_dictionary],
        'output_n_residual':            [utils.is_integer],
        'output_property_scaling':      [utils.is_dictionary, utils.is_None],
        }

    # Output type module
    _output_type = 'AMP'
    
    # Default output block options for atom-wise properties such as, 
    # e.g., 'atomic_energies'.
    _default_output = {
        'atomic_scaling':           True,
        'n_property':               1,
        'n_layer':                  2,
        'n_neurons':                None,
        'activation_fn':            'silu',
        'bias_layer':               True,
        'bias_last':                False,
        'weight_init_layer':        'xavier_uniform_',
        'weight_init_last':         'xavier_uniform_',
        'bias_init_layer':          'zeros_',
        'bias_init_last':           'zeros_',
        'kwargs_weight_init_layer': {},
        'kwargs_weight_init_last':  {},
        'kwargs_bias_init_layer':   {},
        'kwargs_bias_init_last':    {},
        }
    
    # Property dependent output block options
    _property_output_options = {
        'atomic_energies': {
            **_default_output,
             },
        }
    
    # Default output block assignment to properties
    # key: output property
    # item: list -> [0]:    dict or None, output block options including the
    #                       information about the output block type 
    #                       'output_type': 'scalar' or 'tensor'.
    #                       If None, no output block for this property, but
    #                       the property is expected to get build up from other
    #                       output properties.
    #               [1]:    int or list of property str, if [0] is a dict of
    #                       output block options and the output block has the
    #                       type of 'tensor' providing a scalar and tensor 
    #                       prediction, the integer assigns the scalar '0' or
    #                       tensor '1' to the key property. For 'scalar' types,
    #                       no entry is expected.
    #                       If [0] is None, it is a list of property strings
    #                       regarding required property output blocks to
    #                       compute the key property.
    #       dict -> key:    str, case dependent output block properties if
    #                       this other key property is included in the property
    #                       list. It will take the instructions of the first
    #                       match in the dictionary.
    #                       If none of them, use the 'default' output
    #                       block properties
    #               item:   list -> [0]: Output block type (see item -> list)
    #                               [1]: Output block options ...
    #                               [2]: 'Vector' output block result ...
    _default_property_assignment = {
        'energy': [
            None,
            ['atomic_energies']
            ],
        'atomic_energies': [
            _property_output_options['atomic_energies']
            ],
        'forces': [
            None,
            ['energy']
            ],
        'dipole': [
            None,
            ['atomic_charges', 'atomic_dipoles']
            ],
        'quadrupole': [
            None,
            ['atomic_charges', 'atomic_quadrupoles']
            ],
        'atomic_charges': [
            None,
            [],
            ],
        'atomic_dipoles': [
            None,
            [],
            ],
        'atomic_quadrupoles': [
            None,
            [],
            ],
        }
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        output_properties: Optional[List[str]] = None,
        output_properties_options: Optional[Dict[str, Any]] = None,
        output_property_scaling: Optional[
            Dict[str, Union[List[float], Dict[int, List[float]]]]] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize AMP output model.

        """

        super(Output_AMP, self).__init__()
        output_type = 'AMP'

        ####################################
        # # # Check Module Class Input # # #
        ####################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )

        # Update global configuration dictionary
        config.update(
            config_update,
            verbose=verbose)

        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)

        # Get input to output module interface parameters 
        self.n_maxatom = config.get('input_n_maxatom')
        self.n_atombasis = config.get('input_n_atombasis')

        ##########################################
        # # # Check Output Module Properties # # #
        ##########################################

        # Get model properties to check with output module properties
        
        # collect output properties and model properties
        if self.output_properties is None:
            properties_all = []
        else:
            properties_all = list(self.output_properties) 
        if config.get('model_properties') is not None:
            properties_all += list(config['model_properties'])

        # Check for property completeness
        for prop in properties_all:
            if (
                prop in self._default_property_assignment
                and utils.is_array_like(
                    self._default_property_assignment[prop])
                and self._default_property_assignment[prop][0] is None
            ):
                properties_all += list(
                    self._default_property_assignment[prop][1])

        # Initialize output block module properties
        properties_list = []
        properties_options = {}

        # Check defined output block properties
        for prop in properties_all:

            # Prevent property repetition 
            if prop in properties_list:
                continue

            # Check if property is available
            elif prop in self._default_property_assignment:

                # Get default property output block options
                options = self._default_property_assignment[prop]

                # Check and get options for certain conditions
                if utils.is_dictionary(options):
                    # Select property-dependent output options
                    if any([prop in properties_all for prop in options]):
                        for case_prop, case_options in options.items():
                            if case_prop in properties_all:
                                break
                        options = case_options
                    else:
                        options = options.get('default')

                # Check if ouptut block options is now a list
                if not utils.is_array_like(options) or options is None:
                    raise SyntaxError(
                        f"Output block options for '{prop:s}' is not a list!")

                # Skip properties which are not predicted by an output block
                if options[0] is None:
                    continue

                # Add custom output options if defined
                if prop in self.output_properties_options:
                    
                    # Check output options for completeness by adding 
                    # default options for undefined keyword arguments
                    custom_options = self.output_properties_options[prop]
                    if options[0] is not None:
                        options[0] = {**options[0], **custom_options}

                # If skip label None is found, check for dependencies
                elif options[0] is None and utils.is_array_like(options[1]):

                    if any([
                        prop_required not in properties_all
                        for prop_required in options[1]]
                    ):
                        raise SyntaxError(
                            f"Model property prediction for '{prop:s}' "
                            + f"requires property(ies) {options[1]}, "
                            + "which is not satisified!")

                # Add property and options to list
                properties_list.append(prop)
                properties_options[prop] = options[0]

            else:

                raise NotImplementedError(
                    f"Output module of type '{self.output_type:s}' does not "
                    + f"support the property prediction of '{prop:s}'!")

        # Update output property list and output block options
        self.output_properties = properties_list
        self.output_properties_options = properties_options.copy()

        # Update global configuration dictionary
        config_update = {
            'output_properties': self.output_properties,
            'output_properties_options': self.output_properties_options}
        config.update(
            config_update,
            verbose=verbose)

        #####################################
        # # # Output Module Class Setup # # #
        #####################################
        
        # Initialize property to output blocks dictionary
        self.output_property_block = torch.nn.ModuleDict({})
        
        # Add output blocks for scalar and tensor properties
        for prop, options in self.output_properties_options.items():

            # Check optional parameter of atomic scaling and 
            # number of property output
            if options.get('atomic_scaling') is None:
                atomic_scaling = True
            else:
                atomic_scaling = options.get('atomic_scaling')
            if options.get('n_property') is None:
                n_property = 1
            else:
                n_property = options.get('n_property')

            # Check model parameter initialization functions
            weight_init_layer = self.get_init_function(
                options.get('weight_init_layer'))
            weight_init_last = self.get_init_function(
                options.get('weight_init_last'))
            bias_init_layer = self.get_init_function(
                options.get('bias_init_layer'))
            bias_init_last = self.get_init_function(
                options.get('bias_init_last'))

            # Get activation function
            activation_fn = layer.get_activation_fn(
                options.get('activation_fn'))

            # Initialize scalar output block
            self.output_property_block[prop] = (
                layer_painn.PaiNNOutput_scalar(
                    self.n_atombasis,
                    self.n_maxatom,
                    n_property,
                    atomic_scaling,
                    self.device,
                    self.dtype,
                    n_layer=options.get('n_layer'),
                    n_neurons=options.get('n_neurons'),
                    activation_fn=activation_fn,
                    bias_layer=options.get('bias_last'),
                    bias_last=options.get('bias_last'),
                    weight_init_layer=weight_init_layer,
                    weight_init_last=weight_init_last,
                    bias_init_layer=bias_init_layer,
                    bias_init_last=bias_init_last,
                    kwargs_weight_init_layer=options.get(
                        'kwargs_weight_init_layer'),
                    kwargs_weight_init_last=options.get(
                        'kwargs_weight_init_last'),
                    kwargs_bias_init_layer=options.get(
                        'kwargs_bias_init_layer'),
                    kwargs_bias_init_last=options.get(
                        'kwargs_bias_init_last'),
                    )
                )

        # Assign property and atomic properties scaling parameters
        if self.output_property_scaling is not None:
            self.set_property_scaling(self.output_property_scaling)

        # Set atomic charge flag for eventual molecular charge conservation
        if config.get('model_properties') is None:
            raise SyntaxError(
                "Model properties must be defined when initializing a Graph "
                + "module!")
        if 'atomic_charges' in config['model_properties']:
            self.output_atomic_charges = True
        else:
            self.output_atomic_charges = False

        return

    def __str__(self):
        return self._output_type
    
    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {
            'output_type': self._output_type,
            'output_properties': self.output_properties,
            'output_properties_options': self.output_properties_options,
            }

    def set_property_scaling(
        self,
        scaling_parameters: Dict[str, List[float]],
        set_shift_term: Optional[bool] = True,
        set_scaling_factor: Optional[bool] = True,
    ):
        """
        Update output atomic property scaling factor and shift terms.

        Parameters
        ----------
        scaling_parameters: dict(str, (list(float), dict(int, float))
            Dictionary of shift term and scaling factor for a model property
            ({'property': [shift, scaling]}) or shift term and scaling factor
            of a model property for each atom type
            ({'atomic property': {'atomic number': [shift, scaling]}}).
        set_shift_term: bool, optional, default True
            If True, set or update the shift term. Else, keep previous
            value.
        set_scaling_factor: bool, optional, default True
            If True, set or update the scaling factor. Else, keep previous
            value.

        """

        for prop in scaling_parameters:
            
            # Get current scaling parameter from output block
            if prop in self.output_property_block:
                block = self.output_property_block[prop]
                scaling = block.scaling.detach().cpu().numpy()
                atomic_scaling = block.atomic_scaling
            else:
                raise SyntaxError(
                    f"No output block avaiable for property {prop:s}!")

            # Assign new scaling parameter
            if atomic_scaling:
                
                for ai, pars in scaling_parameters.get(prop).items():

                    if set_shift_term:
                        scaling[ai, 0] = pars[0]
                    if set_scaling_factor:
                        scaling[ai, 1] = pars[1]

            else:
                
                pars = scaling_parameters.get(prop)
                if set_shift_term:
                    scaling[0] = pars[0]
                if set_scaling_factor:
                    scaling[1] = pars[1]

            # Set new scaling parameter to output block
            block.set_scaling(
                torch.tensor(scaling, device=self.device, dtype=self.dtype))

        return

    def get_property_scaling(
        self,
    ) -> Dict[str, Union[List[float], Dict[int, List[float]]]]:
        """
        Get atomic property scaling factor and shift terms.

        Returns
        -------
        dict(str, (list(float), dict(int, float))
            Dictionary of shift term and scaling factor for a model property
            ({'property': [shift, scaling]}) or shift term and scaling factor
            of a model property for each atom type
            ({'atomic property': {'atomic number': [shift, scaling]}}).

        """

        # Get scaling factor and shifts for output properties
        scaling_parameters = {}
        
        # Get current scaling parameter from scalar output block
        for prop in self.output_property_block:
            block = self.output_property_block[prop]
            scaling = block.scaling.detach().cpu().numpy()
            if block.atomic_scaling:
                scaling_parameters[prop] = {}
                for ai, pars in enumerate(scaling):
                    scaling_parameters[prop][ai] = pars
            else:
                scaling_parameters[prop] = scaling

        return scaling_parameters

    def get_init_function(
        self,
        function_name: str,
    ) -> Callable:
        """
        Get parameter initialization function of the given name string from the
        'toch.nn.init' module.
        
        Parameters
        ----------
        function_name: str
            Function name in the 'toch.nn.init' module

        Returns
        -------
        Callable
            Parameter initialization function

        """

        if utils.is_string(function_name):
            try:
                return getattr(torch.nn.init, function_name)
            except AttributeError:
                raise AttributeError(
                    "The parameter initialization function "
                    + f"'{function_name:s}' is not a function of "
                    + "the 'torch.nn.init' module!")
        
        return function_name

    def conserve_charge(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Atomic charge manipulation to conserve molecular charge

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors. Required and optional keys are:
            atoms_numbers: torch.Tensor(N_atoms)
                List of atom numbers per system
            atomic_charges: torch.Tensor(N_atoms)
                List of atomic charges
            charge: torch.Tesnsor(N_system)
                Total charge of molecules in batch
            sys_i: torch.Tensor(N_atoms)
                System indices of atoms in batch

        Returns
        -------
        batch: dict(str, torch.Tensor)
            Dictionary added by module results

        """

        # Apply charge manipulation
        charge_deviation = batch['charge'].clone()
        charge_deviation = charge_deviation.scatter_add_(
            0, batch['sys_i'], -batch['atomic_charges'])/batch['atoms_number']
        batch['atomic_charges'] = (
            batch['atomic_charges'] + charge_deviation[batch['sys_i']])

        return batch

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of output module

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors. Required and optional keys are:
            atomic_numbers: torch.Tensor(N_atoms)
                List of atomic numbers
            features: torch.tensor(N_atoms, n_atombasis)
                Atomic feature vectors
            atoms_number: torch.Tesnsor(N_system)
                Number of atoms per molecule in batch
            charge: torch.Tesnsor(N_system)
                Total charge of molecules in batch
            sys_i: torch.Tensor(N_atoms)
                System indices of atoms in batch
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary added by module results

        """

        # Iterate over output blocks
        for prop, output_block in self.output_property_block.items():
            
            # Compute prediction
            batch[prop] = output_block(
                batch['features'],
                batch['atomic_numbers'])

        # Scale atomic charges to conserve correct molecular charge
        if self.output_atomic_charges:
            batch = self.conserve_charge(batch)

        # If verbose, store a copy of the output block prediction
        if verbose:
            for prop in self.output_property_block:
                verbose_prop = 'output_{:s}'.format(prop)
                batch[verbose_prop] = batch[prop].detach()

        return batch

