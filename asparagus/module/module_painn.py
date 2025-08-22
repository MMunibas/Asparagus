
import logging
from typing import Optional, Union, List, Dict, Callable, Any

import numpy as np

import torch

from asparagus import module
from asparagus import layer
from asparagus import settings
from asparagus import utils

from asparagus.layer import layer_painn

__all__ = ['Input_PaiNN', 'Graph_PaiNN', 'Output_PaiNN']

#======================================
# Input Module
#======================================

class Input_PaiNN(torch.nn.Module):
    """
    PaiNN input module class

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
        'input_radial_fn':              'GaussianRBF',
        'input_n_radialbasis':          64,
        'input_cutoff_fn':              'Poly6',
        'input_radial_cutoff':          5.0,
        'input_rbf_center_start':       1.0,
        'input_rbf_center_end':         None,
        'input_rbf_trainable':          True,
        'input_n_maxatom':              94,
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
        }

    # Input type module
    _input_type = 'PaiNN'

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
        input_atom_features_range: Optional[float] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize PaiNN input model.

        """

        super(Input_PaiNN, self).__init__()
        input_type = 'PaiNN'

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
        
        # Get upper RBF center range
        if self.input_rbf_center_end is None:
            self.input_rbf_center_end = self.input_radial_cutoff
        
        # Initialize Radial basis function
        radial_fn = layer.get_radial_fn(self.input_radial_fn)
        self.radial_fn = radial_fn(
            self.input_n_radialbasis,
            self.input_rbf_center_start, self.input_rbf_center_end,
            self.input_rbf_trainable, 
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

        # Assign input data
        positions = batch['positions']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']

        # Collect atom feature vectors
        features = self.atom_features(batch['atomic_numbers'])

        # Compute pair connection vector
        if 'pbc_offset_ij' in batch:
            vectors = (
                positions[idx_j] - positions[idx_i] + batch['pbc_offset_ij'])
        else:
            vectors = positions[idx_j] - positions[idx_i]

        # Compute pair distances
        distances = torch.norm(vectors, dim=-1)

        # PBC: Supercluster approach - Point from eventual image atoms to 
        # respective primary atoms but keep distances
        if 'pbc_idx_pointer' in batch:
            batch['idx_i'] = batch['pbc_idx_pointer'][batch['idx_i']]
            batch['idx_j'] = batch['pbc_idx_pointer'][batch['pbc_idx_j']]

        # Compute distances and vectors for long-range atom pairs
        if 'idx_u' in batch:

            # Assign input data
            idx_u = batch['idx_u']
            idx_v = batch['idx_v']

            # Compute long-range pair connection vectors
            if 'pbc_offset_uv' in batch:
                vectors_uv = (
                    positions[idx_v] - positions[idx_u]
                    + batch['pbc_offset_uv'])
            else:
                vectors_uv = positions[idx_v] - positions[idx_u]
            distances_uv = torch.norm(vectors_uv, dim=-1)
            
            # PBC: Supercluster approach - Point from eventual image atoms to 
            # respective primary atoms but keep primary-image atoms distances
            if 'pbc_idx_pointer' in batch:
                batch['idx_u'] = batch['pbc_idx_pointer'][batch['idx_u']]
                batch['idx_v'] = batch['pbc_idx_pointer'][batch['pbc_v']]

        else:
            
            # Assign atom pair results as long-range pair results
            batch['idx_u'] = batch['idx_i']
            batch['idx_v'] = batch['idx_j']
            vectors_uv = vectors
            distances_uv = distances

        # Compute distance cutoff values
        cutoffs = self.cutoff(distances)
        
        # Compute radial basis functions
        rbfs = self.radial_fn(distances)

        # Assign result data
        batch['features'] = features
        batch['distances'] = distances
        batch['vectors'] = vectors
        batch['cutoffs'] = cutoffs
        batch['rbfs'] = rbfs
        batch['distances_uv'] = distances_uv
        batch['vectors_uv'] = vectors_uv
        
        return batch


#======================================
# Graph Module
#======================================

class Graph_PaiNN(torch.nn.Module): 
    """
    PaiNN message passing module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    graph_n_blocks: int, optional, default 5
        Number of information processing cycles
    graph_activation_fn: (str, object), optional, default 'shifted_softplus'
        Activation function
    graph_stability_constant: float, optional, default 1.e-8
        Numerical stability constant added to scalar products of Cartesian 
        information vectors (guaranteed to be non-zero).

    """
    
    # Default arguments for graph module
    _default_args = {
        'graph_n_blocks':               5,
        'graph_activation_fn':          'silu',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'graph_n_blocks':               [utils.is_integer],
        'graph_activation_fn':          [utils.is_string, utils.is_callable],
        }

    # Graph type module
    _graph_type = 'PaiNN'

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        graph_n_blocks: Optional[int] = None,
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
        
        super(Graph_PaiNN, self).__init__()
        graph_type = 'PaiNN'
        
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
        
        ####################################
        # # # Graph Module Class Setup # # #
        ####################################
        
        # Initialize activation function
        self.activation_fn = layer.get_activation_fn(
            self.graph_activation_fn)
        
        # Initialize feature-wise, continuous-filter convolution network
        self.descriptors_filter = layer.DenseLayer(
            self.n_radialbasis,
            self.graph_n_blocks*self.n_atombasis*3,
            None,
            True,
            self.device,
            self.dtype)
        
        # Initialize message passing blocks
        self.interaction_block = torch.nn.ModuleList([
            layer_painn.PaiNNInteraction(
                self.n_atombasis, 
                self.activation_fn,
                self.device,
                self.dtype)
            for _ in range(self.graph_n_blocks)
            ])
        self.mixing_block = torch.nn.ModuleList([
            layer_painn.PaiNNMixing(
                self.n_atombasis, 
                self.activation_fn,
                self.device,
                self.dtype,
                stability_constant=self.graph_stability_constant)
            for _ in range(self.graph_n_blocks)
            ])

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
            'graph_activation_fn': str(self.graph_activation_fn),
            'graph_stability_constant': self.graph_stability_constant,
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
        features = batch['features']
        distances = batch['distances']
        vectors = batch['vectors']
        cutoffs = batch['cutoffs']
        rbfs = batch['rbfs']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']

        # Apply feature-wise, continuous-filter convolution
        descriptors = (
            self.descriptors_filter(rbfs.unsqueeze(1))
            * cutoffs.unsqueeze(-1).unsqueeze(-1)
            )
        descriptors_list = torch.split(
            descriptors, 3*self.n_atombasis, dim=-1)
        
        # Normalize atom pair vectors
        vectors_normalized = vectors/distances.unsqueeze(-1)

        # Assign isolated atomic feature vectors as scalar feature 
        # vectors
        fsize = features.shape # (len(atomic_numbers), n_atombasis)
        sfeatures = features.unsqueeze(1)

        # Initialize vector feature vectors
        vfeatures = torch.zeros((fsize[0], 3, fsize[1]), device=self.device)

        # Apply message passing model to modify from isolated atomic features
        # vectors to molecular atomic features as a function of the chemical
        # environment
        for ii, (interaction, mixing) in enumerate(
            zip(self.interaction_block, self.mixing_block)
        ):

            sfeatures, vfeatures = interaction(
                sfeatures, 
                vfeatures, 
                descriptors_list[ii], 
                vectors_normalized, 
                idx_i, 
                idx_j,
                fsize[0], 
                fsize[1])

            sfeatures, vfeatures = mixing(
                sfeatures, 
                vfeatures, 
                fsize[1])

        # Flatten scalar atomic feature vector
        sfeatures = sfeatures.squeeze(1)

        # Assign result data
        batch['sfeatures'] = sfeatures
        batch['vfeatures'] = vfeatures

        return batch


#======================================
# Output Module
#======================================

class Output_PaiNN(torch.nn.Module): 
    """
    PaiNN output module class

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
                }
            'atomic_dipoles': { # Tensor output of tensor type output block
                'output_type':              'tensor',
                # The options of the tensor property dict have priority
                'scalar_property':          'atomic_charges',
                'n_property':               1,
                'n_layer':                  2,
                'n_neurons':                None,
                'scalar_activation_fn':     'silu',
                'hidden_activation_fn':     'silu',
                'bias_layer':               True,
                'bias_last':                True,
                'weight_init_layer':        'xavier_uniform_',
                'weight_init_last':         'xavier_uniform_',
                'kwargs_weight_init_layer': {'gain': 1.E-3},
                'kwargs_weight_init_last':  {'gain': 1.E-3},
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
    _output_type = 'PaiNN'
    
    # Default output block options for atom-wise scalar properties such as, 
    # e.g., 'atomic_energies'.
    _default_output_scalar = {
        'output_type':              'scalar',
        'atomic_scaling':           True,
        'n_property':               1,
        'n_layer':                  2,
        'n_neurons':                None,
        'activation_fn':            'silu',
        'bias_layer':               True,
        'bias_last':                True,
        'weight_init_layer':        'xavier_uniform_',
        'weight_init_last':         'xavier_uniform_',
        'bias_init_layer':          'zeros_',
        'bias_init_last':           'zeros_',
        'kwargs_weight_init_layer': {},
        'kwargs_weight_init_last':  {},
        'kwargs_bias_init_layer':   {},
        'kwargs_bias_init_last':    {},
        }
    
    # Default output block options for atom-wise tensor properties such as, 
    # e.g., 'atomic_dipole'.
    _default_output_tensor = {
        'output_type':              'tensor',
        'atomic_scaling':           True,
        'n_property':               1,
        'n_layer':                  2,
        'n_neurons':                None,
        'scalar_activation_fn':     'silu',
        'hidden_activation_fn':     'silu',
        'bias_layer':               True,
        'bias_last':                True,
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
            **_default_output_scalar,
             },
        'atomic_charges': {
            **_default_output_scalar,
            'weight_init_layer':        'xavier_uniform_',
            'weight_init_last':         'xavier_uniform_',
            'kwargs_weight_init_layer': {'gain': 5.E-1},
            'kwargs_weight_init_last':  {'gain': 5.E-1},
            },
        'atomic_dipoles': {
            **_default_output_tensor,
            'scalar_property':          'atomic_charges',
            'weight_init_layer':        'xavier_uniform_',
            'weight_init_last':         'xavier_uniform_',
            'kwargs_weight_init_layer': {'gain': 5.E-1},
            'kwargs_weight_init_last':  {'gain': 5.E-1},
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
        'atomic_charges': {
            'default': [
                _property_output_options['atomic_charges']
                ],
            'atomic_dipoles': [
                _property_output_options['atomic_dipoles'],
                0
                ],
            },
        'atomic_dipoles': [
            _property_output_options['atomic_dipoles'],
            1
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
        Initialize PaiNN output model.

        """

        super(Output_PaiNN, self).__init__()
        output_type = 'PaiNN'

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
        properties_options_scalar = {}
        properties_options_tensor = {}
        
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

                # Check output block options and to the property list
                if options[0].get('output_type') is None:
                
                    raise SyntaxError(
                        "Mising output block type in the options for property "
                        + f"'{prop:s}'")

                elif (
                    options[0].get('output_type').lower() == 'tensor'
                    and options[0].get('scalar_property') is None
                ):

                    raise SyntaxError(
                        "Mising output block 'scalar_property' entry for the "
                        + f"'tensor' output block of property '{prop:s}'!")

                else:

                    # Add scalar property to list
                    properties_list.append(prop)
                    if options[0].get('scalar_property') is not None:
                        properties_list.append(
                            options[0].get('scalar_property'))

                    # Add options to repsective dictionary
                    if options[0].get('output_type').lower() == 'scalar':
                        properties_options_scalar[prop] = options[0]
                    elif options[0].get('output_type').lower() == 'tensor':
                        properties_options_tensor[prop] = options[0]

            else:

                raise NotImplementedError(
                    f"Output module of type '{self.output_type:s}' does not "
                    + f"support the property prediction of '{prop:s}'!")

        # Update output property list and output block options
        self.output_properties = properties_list
        self.output_properties_options = properties_options_scalar.copy()
        for prop, options in properties_options_tensor.items():
            self.output_properties_options[prop] = options
            self.output_properties_options[options['scalar_property']] = (
                options)

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
        self.output_property_scalar_block = torch.nn.ModuleDict({})
        self.output_property_tensor_block = torch.nn.ModuleDict({})
        
        # Initialize list of scalar property of each the tensor output block
        self.output_tensor_scalar = {}
        self.output_scalar_tensor = {}

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

            # Initialize scalar and tensor output blocks
            if options.get('output_type').lower() == 'scalar':
                
                # Get activation function
                activation_fn = layer.get_activation_fn(
                    options.get('activation_fn'))

                # Initialize scalar output block
                self.output_property_scalar_block[prop] = (
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

            elif (
                options.get('output_type').lower() == 'tensor'
                and prop != options['scalar_property']
            ):

                # Assing scalar property to and tensor property tag
                self.output_tensor_scalar[prop] = options['scalar_property']
                self.output_scalar_tensor[options['scalar_property']] = prop 

                # Get scalar and hidden activation function
                scalar_activation_fn = layer.get_activation_fn(
                    options['scalar_activation_fn'])
                hidden_activation_fn = layer.get_activation_fn(
                    options['hidden_activation_fn'])

                # Initialize tensor output block
                self.output_property_tensor_block[prop] = (
                    layer_painn.PaiNNOutput_tensor(
                        self.n_atombasis,
                        self.n_maxatom,
                        n_property,
                        atomic_scaling,
                        self.device,
                        self.dtype,
                        n_layer=options.get('n_layer'),
                        n_neurons=options.get('n_neurons'),
                        scalar_activation_fn=scalar_activation_fn,
                        hidden_activation_fn=hidden_activation_fn,
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

        # Set atomic charge flag for eventual molecualr charge conservation
        if 'atomic_charges' in self.output_properties:
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
            if prop in self.output_property_scalar_block:
                block = self.output_property_scalar_block[prop]
                scaling = block.scaling.detach().cpu().numpy()
                atomic_scaling = block.atomic_scaling
            elif prop in self.output_property_tensor_block:
                block = self.output_property_tensor_block[prop]
                scaling = block.vscaling.detach().cpu().numpy()
                atomic_scaling = block.atomic_scaling
            elif prop in self.output_scalar_tensor:
                vprop = self.output_scalar_tensor[prop]
                block = self.output_property_tensor_block[vprop]
                scaling = block.sscaling.detach().cpu().numpy()
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
        for prop in self.output_property_scalar_block:
            block = self.output_property_scalar_block[prop]
            scaling = block.scaling.detach().cpu().numpy()
            if block.atomic_scaling:
                scaling_parameters[prop] = {}
                for ai, pars in enumerate(scaling):
                    scaling_parameters[prop][ai] = pars
            else:
                scaling_parameters[prop] = scaling

        # Get current scaling parameters from tensor output block for tensor
        # and scalar property
        for prop in self.output_property_tensor_block:
            block = self.output_property_tensor_block[prop]
            scaling = block.vscaling.detach().cpu().numpy()
            if block.atomic_scaling:
                scaling_parameters[prop] = {}
                for ai, pars in enumerate(scaling):
                    scaling_parameters[prop][ai] = pars
            else:
                scaling_parameters[prop] = scaling

            sprop = self.output_tensor_scalar[prop]
            scaling = block.sscaling.detach().cpu().numpy()
            if block.atomic_scaling:
                scaling_parameters[sprop] = {}
                for ai, pars in enumerate(scaling):
                    scaling_parameters[sprop][ai] = pars
            else:
                scaling_parameters[sprop] = scaling

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
            sfeatures: torch.tensor(N_atoms, n_atombasis)
                Scalar atomic feature vectors
            vfeatures: torch.tensor(N_atoms, 3, n_atombasis)
                Vector atomic feature vectors
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

        # Assign input data
        atomic_numbers = batch['atomic_numbers']
        sfeatures = batch['sfeatures']
        vfeatures = batch['vfeatures']
        sys_i = batch['sys_i']

        # Iterate over scalar output blocks
        for prop, output_block in self.output_property_scalar_block.items():
            
            # Compute prediction
            batch[prop] = output_block(sfeatures, atomic_numbers)

        # Iterate over tensor output blocks
        for vprop, output_block in self.output_property_tensor_block.items():
            
            # Get scalar tensor property tag
            sprop = self.output_tensor_scalar[vprop]

            # Compute prediction
            batch[sprop], batch[vprop] = (
                output_block(sfeatures, vfeatures, atomic_numbers))
               
        # Scale atomic charges to conserve correct molecular charge
        if self.output_atomic_charges:
            batch = self.conserve_charge(batch)

        # If verbose, store a copy of the output block prediction
        if verbose:
            for prop in self.output_property_scalar_block:
                verbose_prop = 'output_{:s}'.format(prop)
                batch[verbose_prop] = batch[prop].detach()
            for vprop in self.output_property_tensor_block:
                sprop = self.output_tensor_scalar[vprop]
                verbose_sprop = 'output_{:s}'.format(sprop)
                verbose_vprop = 'output_{:s}'.format(vprop)
                batch[verbose_sprop] = batch[sprop].detach()
                batch[verbose_vprop] = batch[vprop].detach()

        return batch

