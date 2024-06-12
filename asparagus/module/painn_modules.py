import logging
import numpy as np
from typing import Optional, Union, List, Dict, Callable, Any

import torch

from .. import module
from .. import layer
from .. import settings
from .. import utils

from ..layer import painn_layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        'input_radial_cutoff':          8.0,
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
        config.update(config_update)
        
        # Assign module variable parameters from configuration
        self.device = config.get('device')
        self.dtype = config.get('dtype')
        
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
            max_norm=1.0,
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
            device=self.device,
            dtype=self.dtype)

        return

    def __str__(self):
        return self.input_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'input_type': self.input_type,
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
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        pbc_offset_ij: Optional[torch.Tensor] = None,
        idx_u: Optional[torch.Tensor] = None,
        idx_v: Optional[torch.Tensor] = None,
        pbc_offset_uv: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass of the input module.

        Parameters
        ----------
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

        Returns
        -------
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

        """
        
        # Collect atom feature vectors
        features = self.atom_features(atomic_numbers)

        # Compute pair connection vector
        if pbc_offset_ij is None:
            vectors = positions[idx_j] - positions[idx_i]
        else:
            vectors = positions[idx_j] - positions[idx_i] + pbc_offset_ij

        # Compute pair distances
        distances = torch.norm(vectors, dim=-1)

        # Compute long-range cutoffs
        if pbc_offset_uv is None and idx_u is not None:
            distances_uv = torch.norm(
                positions[idx_u] - positions[idx_v], dim=-1)
        elif idx_u is not None:
            distances_uv = torch.norm(
                positions[idx_v] - positions[idx_u] + pbc_offset_uv, dim=-1)
        else:
            distances_uv = distances

        # Compute distance cutoff values
        cutoffs = self.cutoff(distances)

        # Compute radial basis functions
        rbfs = self.radial_fn(distances)

        return features, distances, vectors, cutoffs, rbfs, distances_uv


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

    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        graph_n_blocks: Optional[int] = None,
        graph_activation_fn: Optional[Union[str, object]] = None,
        graph_stability_constant: Optional[float] = None,
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
        config.update(config_update)

        # Assign module variable parameters from configuration
        self.device = config.get('device')
        self.dtype = config.get('dtype')

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
            activation_fn=None,
            bias=True,
            device=self.device,
            dtype=self.dtype)
        
        # Initialize message passing blocks
        self.interaction_block = torch.nn.ModuleList([
            painn_layers.PaiNNInteraction(
                self.n_atombasis, 
                activation_fn=self.activation_fn,
                device=self.device,
                dtype=self.dtype)
            for _ in range(self.graph_n_blocks)
            ])
        self.mixing_block = torch.nn.ModuleList([
            painn_layers.PaiNNMixing(
                self.n_atombasis, 
                activation_fn=self.activation_fn,
                stability_constant=self.graph_stability_constant,
                device=self.device,
                dtype=self.dtype)
            for _ in range(self.graph_n_blocks)
            ])

        return

    def __str__(self):
        return self.graph_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'graph_type': self.graph_type,
            'graph_n_blocks': self.graph_n_blocks,
            'graph_activation_fn': str(self.graph_activation_fn),
            'graph_stability_constant': self.graph_stability_constant,
            }

    def forward(
        self, 
        features: torch.Tensor,
        distances: torch.Tensor,
        vectors: torch.Tensor,
        cutoffs: torch.Tensor,
        rbfs: torch.Tensor,
        idx_i: torch.Tensor, 
        idx_j: torch.Tensor,
    ) -> List[torch.Tensor]:

        """
        Forward pass of the graph model.
        
        Parameter
        ---------
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

        Returns
        -------
        sfeatures: torch.tensor(N_atoms, n_atombasis)
            Modified scalar atomic feature vectors
        vfeatures: torch.tensor(N_atoms, 3, n_atombasis)
            Modified vector atomic feature vectors

        """
        
        # Apply feature-wise, continuous-filter convolution
        descriptors = (
            self.descriptors_filter(rbfs[:, None, :])*cutoffs[:, None, None])
        descriptors_list = torch.split(
            descriptors, 3*self.n_atombasis, dim=-1)
        
        # Normalize atom pair vectors
        vectors_normalized = vectors/distances[:, None]

        # Assign isolated atomic feature vectors as scalar feature 
        # vectors
        fsize = features.shape # (len(atomic_numbers), n_atombasis)
        sfeatures = features[:, None, :]

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
        
        return sfeatures, vfeatures


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
    output_properties: list(str), optional '['energy', 'forces']'
        List of output properties to compute by the model
        e.g. ['energy', 'forces', 'atomic_charges']
    output_properties_options: dict(str, Any), optional, default {}
        Dictionary of output block options (item) for a property (key).
        Dictionary inputs are, e.g.:
        output_properties_options = {
            'atomic_energies': {    # Output of a scalar type output block
                'output_type':          'scalar',
                'n_property':           1,
                'n_layer':              2,
                'n_neurons':            None,
                'activation_fn':        'silu',
                'bias_layer':           True,
                'bias_last':            True,
                'weight_init_layer':    torch.nn.init.xavier_uniform_,
                'weight_init_last':     torch.nn.init.xavier_uniform_,
                'bias_init_layer':      torch.nn.init.zeros_,
                'bias_init_last':       torch.nn.init.zeros_,
                }
            'atomic_charges': { # Scalar output of tensor type output block
                'output_type':          'tensor',
                'properties':           ['atomic_charges', 'atomic_dipoles'],
                'n_property':           1,
                'n_layer':              2,
                'n_neurons':            None,
                'scalar_activation_fn': 'silu',
                'hidden_activation_fn': 'silu',
                'bias_layer':           True,
                'bias_last':            True,
                'weight_init_layer':    torch.nn.init.xavier_uniform_,
                'weight_init_last':     torch.nn.init.xavier_uniform_,
                'bias_init_layer':      torch.nn.init.zeros_,
                'bias_init_last':       torch.nn.init.zeros_,
                },
            'atomic_dipoles': { # Tensor output of tensor type output block
                'output_type':          'tensor',
                'properties':           ['atomic_charges', 'atomic_dipoles'],
                'n_property':           1,
                'n_layer':              2,
                'n_neurons':            None,
                'scalar_activation_fn': 'silu',
                'hidden_activation_fn': 'silu',
                'bias_layer':           True,
                'bias_last':            True,
                'weight_init_layer':    torch.nn.init.xavier_uniform_,
                'weight_init_last':     torch.nn.init.xavier_uniform_,
                'bias_init_layer':      torch.nn.init.zeros_,
                'bias_init_last':       torch.nn.init.zeros_,
                },
            }
    output_n_residual: int, optional, default 1
        Number of residual layers for transformation from atomic feature vector
        to output results.
    output_scaling_parameter: dictionary, optional, default None
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
        'output_scaling_parameter':     None,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'output_properties':            [utils.is_string_array, utils.is_None],
        'output_properties_options':    [utils.is_dictionary],
        'output_n_residual':            [utils.is_integer],
        'output_scaling_parameter':     [utils.is_dictionary],
        }
    
    # Output module specially handled properties
    #_property_special = ['energy', 'atomic_energies', 'atomic_charges']
    
    # Default output block options for atom-wise scalar properties such as, 
    # e.g., 'atomic_energies'.
    _default_output_scalar = {
        'output_type':          'scalar',
        'n_property':           1,
        'n_layer':              2,
        'n_neurons':            None,
        'activation_fn':        'silu',
        'bias_layer':           True,
        'bias_last':            True,
        'weight_init_layer':    torch.nn.init.xavier_uniform_,
        'weight_init_last':     torch.nn.init.xavier_uniform_,
        'bias_init_layer':      torch.nn.init.zeros_,
        'bias_init_last':       torch.nn.init.zeros_,
        }
    
    # Default output block options for atom-wise tensor properties such as, 
    # e.g., 'atomic_dipole'.
    _default_output_tensor = {
        'output_type':          'tensor',
        'properties':           [None, None],
        'n_property':           1,
        'n_layer':              2,
        'n_neurons':            None,
        'scalar_activation_fn': 'silu',
        'hidden_activation_fn': 'silu',
        'bias_layer':           True,
        'bias_last':            True,
        'weight_init_layer':    torch.nn.init.xavier_uniform_,
        'weight_init_last':     torch.nn.init.xavier_uniform_,
        'bias_init_layer':      torch.nn.init.zeros_,
        'bias_init_last':       torch.nn.init.zeros_,
        }
    
    
    # Default output block assignment to properties
    # key: output property
    # item: list -> [0]:    str, Output block type 'scalar' or 'tensor'
    #                       None, no output block (skip property)
    #               [1]:    dict, output block options if [0] str
    #                       str, dependency to other property if [0] None
    #               [2]:    optional, int: scalar or tensor result from a
    #                           'tensor' type output block
    #       dict -> key:    str, either default case or case if respective
    #                           property included in output properties list
    #               item:   list -> [0]: Output block type (see item -> list)
    #                               [1]: Output block options ...
    #                               [2]: 'Vector' output block result ...
    _default_property_assignment = {
        'energy': [None, ['atomic_energies']],
        'atomic_energies': [_default_output_scalar],
        'forces': [None, ['energy']],
        'dipole': [None, ['atomic_charges', 'atomic_dipoles']],
        'atomic_charges': {
            'default': [_default_output_scalar],
            'atomic_dipoles': [_default_output_tensor, 0],
            },
        'atomic_dipoles': [_default_output_tensor, 1],
        }
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        output_properties: Optional[List[str]] = None,
        output_properties_options: Optional[Dict[str, Any]] = None,
        output_scaling_parameter: Optional[Dict[str, List[float]]] = None,
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
        config.update(config_update)

        # Assign module variable parameters from configuration
        self.device = config.get('device')
        self.dtype = config.get('dtype')

        # Get input and graph to output module interface parameters 
        self.n_maxatom = config.get('input_n_maxatom')
        self.n_atombasis = config.get('input_n_atombasis')
        self.n_blocks = config.get('graph_n_blocks')

        ##########################################
        # # # Check Output Module Properties # # #
        ##########################################

        # Get model properties to check with output module properties
        model_properties = config.get('model_properties')
        properties_all = []
        if self.output_properties is not None:
            properties_all += list(self.output_properties) 
        if model_properties is not None:
            properties_all += list(model_properties)

        # Initialize output module properties
        properties_list = []
        properties_options_scalar = {}
        properties_options_tensor = {}
        
        # Check defined output properties
        for prop in properties_all:
            # Prevent property repetition 
            if prop in properties_list:
                continue
            # Check if property is available
            elif prop in self._default_property_assignment:
                
                # Get default property output block instruction
                instructions = self._default_property_assignment[prop]
                
                # Check and get instruction for certain conditions
                if utils.is_dictionary(instructions):
                    # Select property-dependent output instructions
                    for case_prop, case_instructions in instructions.items():
                        if case_prop == 'default':
                            instructions = instructions.get('default')
                        elif case_prop in properties_all:
                            instructions = case_instructions
                            break
                
                # Add custom output options if defined
                if prop in self.output_properties_options:
                    
                    # Add property and output options
                    properties_list.append(prop)
                    
                    # Check output options for completeness by adding 
                    # default options for undefined keyword arguments
                    option = self.output_properties_options[prop]
                    if instructions[0] is not None:
                        option = {**instructions[0], **option}

                    # Assign output block to scalar or tensor property list
                    out_type = option.get('output_type')
                    if out_type == 'scalar':
                        properties_options_scalar[prop] = option
                    elif out_type == 'tensor':                        
                        properties_options_tensor[prop] = option
                    else:
                        raise SyntaxError(
                            "Costum output block options for property "
                            + f"{prop:s} has unknown type '{out_type:s}'!")

                # If skip label is found in instructions
                elif instructions[0] is None:

                    # Check dependencies and skip property
                    if any([
                        prop_required not in properties_all
                        for prop_required in instructions[1]]
                    ):
                        raise SyntaxError(
                            f"Model property prediction for '{prop:s}' "
                            + f"requires property '{instructions[1]}', "
                            + "which is not defined!")
                    else:
                        pass

                # Else  output instructions are given
                else:

                    # Add property
                    properties_list.append(prop)

                    # Prepare output block options
                    option = instructions[0]
                    if len(instructions) == 1:
                        properties_options_scalar[prop] = option
                    elif len(instructions) > 1 and 'properties' in option:
                        option['properties'][instructions[1]] = prop
                        properties_options_tensor[prop] = option
                    else:
                        raise SyntaxError()

            else:
                raise NotImplementedError(
                    f"Output module of type {self.output_type:s} does not "
                    + f"support the property prediction of {prop:s}!")

        # Update output property list and global configuration dictionary
        self.output_properties = properties_list
        self.output_properties_options = {
            **properties_options_scalar, **properties_options_tensor}
        config_update = {
            'output_properties': self.output_properties,
            'output_properties_options': self.output_properties_options}
        config.update(config_update)
        
        #####################################
        # # # Output Module Class Setup # # #
        #####################################
        
        # Initialize property to output blocks dictionary
        self.output_property_scalar_block = torch.nn.ModuleDict({})
        self.output_property_tensor_block = torch.nn.ModuleDict({})
        
        # Initialize number of property per output block dictionary and 
        # output block tag list
        self.output_n_property = {}
        self.output_tag_properties = {}
        
        # Add output blocks for scalar properties
        for prop, options in properties_options_scalar.items():

            # Get activation function
            activation_fn = layer.get_activation_fn(
                options.get('activation_fn'))

            # Check essential number of property output parameter
            if options.get('n_property') is None:
                raise SynaxError(
                    "Number of output properties 'n_property' for property "
                    + f"{prop:s} is not defined!")
            self.output_n_property[prop] = options.get('n_property')

            # Initialize scalar output block
            self.output_property_scalar_block[prop] = (
                painn_layers.PaiNNOutput_scalar(
                    self.n_atombasis,
                    options.get('n_property'),
                    n_layer=options.get('n_layer'),
                    n_neurons=options.get('n_neurons'),
                    activation_fn=activation_fn,
                    bias_layer=options.get('bias_last'),
                    bias_last=options.get('bias_last'),
                    weight_init_layer=options.get('weight_init_layer'),
                    weight_init_last=options.get('weight_init_last'),
                    bias_init_layer=options.get('bias_init_layer'),
                    bias_init_last=options.get('bias_init_last'),
                    device=self.device,
                    dtype=self.dtype)
                )
        
        # Add output blocks for (scalar +) tensor properties
        for prop, options in properties_options_tensor.items():

            # Get activation function
            scalar_activation_fn = layer.get_activation_fn(
                options.get('scalar_activation_fn'))
            hidden_activation_fn = layer.get_activation_fn(
                options.get('hidden_activation_fn'))

            # Check essential number of property output parameter
            if options.get('n_property') is None:
                raise SynaxError(
                    "Number of output properties 'n_property' for property "
                    + f"{prop:s} is not defined!")
            self.output_n_property[prop] = options.get('n_property')

            # Get combined scalar and tensor property tag, skip if already done
            prop_tuple = tuple(options.get('properties'))
            prop_tag = '_&_'.join([str(p) for p in prop_tuple])
            if prop_tag in self.output_tag_properties:
                continue
            self.output_tag_properties[prop_tag] = prop_tuple

            # Initialize scalar output block
            self.output_property_tensor_block[prop_tag] = (
                painn_layers.PaiNNOutput_tensor(
                    self.n_atombasis,
                    options.get('n_property'),
                    n_layer=options.get('n_layer'),
                    n_neurons=options.get('n_neurons'),
                    scalar_activation_fn=scalar_activation_fn,
                    hidden_activation_fn=hidden_activation_fn,
                    bias_layer=options.get('bias_last'),
                    bias_last=options.get('bias_last'),
                    weight_init_layer=options.get('weight_init_layer'),
                    weight_init_last=options.get('weight_init_last'),
                    bias_init_layer=options.get('bias_init_layer'),
                    bias_init_last=options.get('bias_init_last'),
                    device=self.device,
                    dtype=self.dtype)
                )
        
        # Initialize property scaling dictionary and atomic energy shift
        self.set_property_scaling(self.output_scaling_parameter)

        return

    def set_property_scaling(
        self,
        scaling_parameter: Dict[str, List[float]],
    ):
        """
        Update output property scaling factor and shift term dictionary.
        
        Parameters
        ----------
        scaling_parameter: dict(str, torch.Tensor(2))
            Property average and standard deviation for the use as
            scaling factor (standard deviation) and shift term (average) 
            parameter pairs (item) for each property (key).

        """

        # Set scaling factor and shifts for output properties
        output_scaling = {}
        for prop in self.output_properties:
            
            # If property scaling input is missing, initialize default
            if (
                scaling_parameter is None
                or scaling_parameter.get(prop) is None
            ):
            
                output_scaling[prop] = torch.nn.Parameter(
                    torch.tensor(
                        [[1.0, 0.0] for _ in range(self.n_maxatom)],
                        device=self.device, 
                        dtype=self.dtype)
                    )

            else:
                
                # Assign scaling factor and shift
                (shift, scale) = scaling_parameter.get(prop)
                output_scaling[prop] = torch.nn.Parameter(
                    torch.tensor(
                        [[scale, shift] for _ in range(self.n_maxatom)],
                        device=self.device, 
                        dtype=self.dtype)
                    )

        # Convert model scaling to torch dictionary
        self.output_scaling = torch.nn.ParameterDict(output_scaling)

        return
    
    def __str__(self):
        return self.output_type
    
    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {
            'output_type': self.output_type,
            'output_properties': self.output_properties,
            'output_properties_options': self.output_properties_options,
            }
    
    def forward(
        self,
        sfeatures: torch.Tensor,
        vfeatures: torch.Tensor,
        atomic_numbers: Optional[torch.Tensor] = None,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of output module

        Parameters
        ----------
        sfeatures: torch.tensor(N_atoms, n_atombasis)
            Scalar atomic feature vectors
        vfeatures: torch.tensor(N_atoms, 3, n_atombasis)
            Vector atomic feature vectors
        atomic_numbers: torch.Tensor(N_atoms), optional, default None
            List of atomic numbers
        properties: list(str), optional, default None
            List of properties to compute by the model. If None, all properties
            are predicted.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary of predicted properties

        """
        
        # Initialize predicted properties dictionary
        output_prediction = {}
        
        # Check requested properties
        if properties is None:
            predict_all = True
        else:
            predict_all = False

        # Iterate over scalar output blocks
        for prop, output_block in self.output_property_scalar_block.items():
            
            # Skip if property not requested
            if not predict_all and prop not in properties:
                continue
            
            # Compute prediction
            output_prediction[prop] = output_block(sfeatures)

            # Flatten prediction for scalar properties
            if self.output_n_property[prop] == 1:
                output_prediction[prop] = torch.flatten(
                    output_prediction[prop], start_dim=0)

        # Iterate over tensor output blocks
        for prop, output_block in self.output_property_tensor_block.items():
            
            # Get scalar and tensor property tags
            (sprop, vprop) = self.output_tag_properties[prop]
            
            # Skip if property not requested
            if (
                not predict_all 
                and not any([propi in properties for prop_i in (sprop, vprop)])
            ):
                continue
            
            # Compute prediction
            output_prediction[sprop], output_prediction[vprop] = (
                output_block(sfeatures, vfeatures))

            # Flatten prediction for scalar properties
            if self.output_n_property[sprop] == 1:
                output_prediction[sprop] = torch.flatten(
                    output_prediction[sprop], start_dim=0)
                
            # Flatten prediction for single vector properties
            if self.output_n_property[vprop] == 1:
                output_prediction[vprop] = (
                    output_prediction[vprop].reshape(-1, 3))

        # Apply property scaling
        for prop, scaling in self.output_scaling.items():
            (scale, shift) = scaling[atomic_numbers].T
            compatible_shape = (
                [atomic_numbers.shape[0]]
                + (len(output_prediction[prop].shape) - 1)*[1])
            output_prediction[prop] = (
                output_prediction[prop]*scale.reshape(compatible_shape)
                + shift.reshape(compatible_shape))

        return output_prediction
