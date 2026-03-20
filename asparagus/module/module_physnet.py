#from memory_profiler import profile

import logging
from typing import Optional, Union, List, Dict, Callable, Any

import numpy as np

import torch

from asparagus import module
from asparagus import layer
from asparagus import settings
from asparagus import utils

from asparagus.layer import layer_physnet

__all__ = ['Input_PhysNet', 'Graph_PhysNet', 'Output_PhysNet']

#======================================
# Input Module
#======================================

class Input_PhysNet(torch.nn.Module):
    """
    PhysNet input module class

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
    input_atom_features_range: float, optional, default sqrt(3)
        Range for uniform distribution randomizer for initial atom feature
        vector.
    **kwargs: dict, optional
        Additional arguments for parameter initialization

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

    # Input type module
    _input_type = 'PhysNet'

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
        device: Optional[str] = None,
        dtype: Optional['device'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize PhysNet input module.

        """

        super(Input_PhysNet, self).__init__()

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
            self.input_rbf_center_start,
            self.input_rbf_center_end,
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

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary added by module results:
            features: torch.tensor(N_atoms, n_atombasis)
                Atomic feature vectors
            distances: torch.tensor(N_pairs)
                Atom pair distances
            cutoffs: torch.tensor(N_pairs)
                Atom pair distance cutoffs
            rbfs: torch.tensor(N_pairs, n_radialbasis)
                Atom pair radial basis functions
            distances_uv: torch.tensor(N_pairs_uv)
                Long-range atom pair distances

        """

        # Assign input data
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']

        # Collect atom feature vectors
        features = self.atom_features(atomic_numbers)

        # Compute atom pair distances
        if 'pbc_offset_ij' in batch:
            vectors = (
                positions[idx_j] - positions[idx_i] + batch['pbc_offset_ij'])
        else:
            vectors = positions[idx_j] - positions[idx_i]
        distances = torch.norm(vectors, dim=-1)

        # ML/MM approach - Point ML atom pair indices from full ML/MM system to
        # the ML system indices (ML/MM index number of, e.g., 41 for the first
        # ML atom in the ML/MM system becomes index 0 for the ML system).
        if 'ml_idx_p' in batch:
            batch['idx_i'] = batch['ml_idx_p'][idx_i]
            batch['idx_j'] = batch['ml_idx_p'][idx_j]

            # PBC supercluster approach - Point from ML atom pair index j of
            # atoms in the image cell to the respective primary cell ML
            # atom index,
            if 'ml_idx_jp' in batch:
                batch['idx_j'] = batch['ml_idx_p'][batch['ml_idx_jp']]

        # Check long-range atom pair indices
        if 'idx_u' in batch:

            # Assign input data
            idx_u = batch['idx_u']
            idx_v = batch['idx_v']

            # Compute long-range cutoffs
            if 'pbc_offset_uv' in batch:
                vectors_uv = (
                    positions[idx_v] - positions[idx_u]
                    + batch['pbc_offset_uv'])
            else:
                vectors_uv = positions[idx_v] - positions[idx_u]
            distances_uv = torch.norm(vectors_uv, dim=-1)
            
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
            
            batch['idx_u'] = batch['idx_i']
            batch['idx_v'] = batch['idx_j']
            distances_uv = distances

        # Compute distance cutoff values
        cutoffs = self.cutoff(distances)

        # Compute radial basis functions
        rbfs = self.radial_fn(distances)

        # Assign result data
        batch['features'] = features
        batch['distances'] = distances
        batch['cutoffs'] = cutoffs
        batch['rbfs'] = rbfs
        batch['distances_uv'] = distances_uv

        return batch


#======================================
# Graph Module
#======================================

class Graph_PhysNet(torch.nn.Module): 
    """
    PhysNet message passing module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    graph_n_blocks: int, optional, default 5
        Number of information processing cycles
    graph_n_residual_interaction: int, optional, default 3
        Number of residual layers for atomic feature and radial basis vector
        interaction.
    graph_n_residual_features: int, optional, default 2
        Number of residual layers for atomic feature interactions.
    graph_activation_fn: (str, object), optional, default 'shifted_softplus'
        Residual layer activation function.

    """
    
    # Default arguments for graph module
    _default_args = {
        'graph_n_blocks':               5,
        'graph_n_residual_interaction': 3,
        'graph_n_residual_features':    2,
        'graph_activation_fn':          'shifted_softplus',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'graph_n_blocks':               [utils.is_integer],
        'graph_n_residual_interaction': [utils.is_integer],
        'graph_n_residual_features':    [utils.is_integer],
        'graph_activation_fn':          [utils.is_string, utils.is_callable],
        }

    # Graph type module
    _graph_type = 'PhysNet'

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        graph_n_blocks: Optional[int] = None,
        graph_n_residual_interaction: Optional[int] = None,
        graph_n_residual_features: Optional[int] = None,
        graph_activation_fn: Optional[Union[str, object]] = None,
        device: Optional[str] = None,
        dtype: Optional['device'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize PhysNet message passing module.

        """
        
        super(Graph_PhysNet, self).__init__()
        
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

        # Initialize message passing blocks
        self.interaction_blocks = torch.nn.ModuleList([
            layer_physnet.InteractionBlock(
                self.n_atombasis, 
                self.n_radialbasis, 
                self.graph_n_residual_interaction,
                self.graph_n_residual_features,
                self.activation_fn,
                device=self.device,
                dtype=self.dtype)
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
            'graph_n_residual_interaction': self.graph_n_residual_interaction,
            'graph_n_residual_features': self.graph_n_residual_features,
            'graph_activation_fn': self.graph_activation_fn,
            }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the graph module.
        
        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors. Required and optional keys are:
            features: torch.tensor(N_atoms, n_atombasis)
                Atomic feature vectors
            distances: torch.tensor(N_pairs)
                Atom pair distances
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
        dict(str, torch.Tensor)
            Dictionary added by module results:
            features_list: [torch.tensor(N_atoms, n_atombasis)]*n_blocks
                List of modified atomic feature vectors

        """

        # Assign input data
        features = batch['features']
        distances = batch['distances']
        cutoffs = batch['cutoffs']
        rbfs = batch['rbfs']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']

        # Compute descriptor vectors
        descriptors = cutoffs.unsqueeze(-1)*rbfs
        
        # Initialize refined feature vector list
        x_list = torch.zeros(
            (self.graph_n_blocks, features.shape[0], features.shape[1]),
            device=self.device,
            dtype=self.dtype)
        
        # Apply message passing model
        x = features
        for iblock, interaction_block in enumerate(self.interaction_blocks):
            x = interaction_block(x, descriptors, idx_i, idx_j)
            x_list[iblock] = x.clone()

        # Assign result data
        batch['features_list'] = x_list

        return batch


#======================================
# Output Module
#======================================

class Output_PhysNet(torch.nn.Module): 
    """
    PhysNet output module class

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
    output_n_residual: int, optional, default 1
        Number of residual layers for transformation from atomic feature vector
        to output results.
    output_activation_fn: (str, callable), optional, default 'shifted_softplus'
        Residual layer activation function.
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
        'output_n_residual':            1,
        'output_activation_fn':         'shifted_softplus',
        'output_property_scaling':      None,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'output_properties':            [utils.is_string_array, utils.is_None],
        'output_n_residual':            [utils.is_integer],
        'output_activation_fn':         [utils.is_string, utils.is_callable],
        'output_property_scaling':      [utils.is_dictionary, utils.is_None],
        }
    
    # Output type module
    _output_type = 'PhysNet'

    # Property exclusion lists for properties (keys) derived from other 
    # properties (items) but in the model class
    _property_exclusion = {
        'energy': ['atomic_energies'],
        'forces': [], 
        'hessian': [],
        'dipole': ['atomic_charges']}
    
    # Output module properties that are handled specially
    _property_special = ['energy', 'atomic_energies', 'atomic_charges']
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        output_properties: Optional[List[str]] = None,
        output_n_residual: Optional[int] = None,
        output_activation_fn: Optional[Union[str, object]] = None,
        output_property_scaling: Optional[
            Dict[str, Union[List[float], Dict[int, List[float]]]]] = None,
        device: Optional[str] = None,
        dtype: Optional['device'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize PhysNet output module.

        """

        super(Output_PhysNet, self).__init__()

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

        # Get input and graph to output module interface parameters 
        self.n_maxatom = config.get('input_n_maxatom')
        n_atombasis = config.get('input_n_atombasis')
        graph_n_blocks = config.get('graph_n_blocks')

        ##########################################
        # # # Check Output Module Properties # # #
        ##########################################

        # Get model properties to check with output module properties
        model_properties = config.get('model_properties')

        # Initialize output module properties
        properties_list = []
        if self.output_properties is not None:
            for prop in self.output_properties:
                if prop in self._property_exclusion:
                    for prop_der in self._property_exclusion[prop]:
                        properties_list.append(prop_der)
                else:
                    properties_list.append(prop)

        # Check output module properties with model properties
        for prop in model_properties:
            if (
                prop not in properties_list
                and prop in self._property_exclusion
            ):
                for prop_der in self._property_exclusion[prop]:
                    if prop_der not in properties_list:
                        properties_list.append(prop_der)
            elif prop not in properties_list:
                properties_list.append(prop)

        # Update output property list and global configuration dictionary
        self.output_properties = properties_list
        config_update = {
            'output_properties': self.output_properties}
        config.update(
            config_update,
            verbose=verbose)

        #####################################
        # # # Output Module Class Setup # # #
        #####################################

        # Initialize case flags
        self.output_energies_charges = False
        self.output_atomic_charges = False

        # Initialize activation function
        self.activation_fn = layer.get_activation_fn(
            self.output_activation_fn)

        # Initialize property to output block dictionary
        self.output_property_block = torch.nn.ModuleDict({})

        # Check special case: atom energies and charges from one output block
        if all([
            prop in self.output_properties
            for prop in ['atomic_energies', 'atomic_charges']]
        ):
            # Set case flag for output module predicting atomic energies and
            # charges
            self.output_energies_charges = True
            self.output_atomic_charges = True

            # Number of property outputs
            output_n_property = 2

            # PhysNet energy and atom charges output block
            output_block = layer_physnet.OutputBlock(
                graph_n_blocks,
                n_atombasis,
                self.n_maxatom,
                output_n_property,
                self.output_n_residual,
                self.activation_fn,
                True,
                True,
                True,
                device=self.device,
                dtype=self.dtype)

            # Assign output block to dictionary
            self.output_property_block['atomic_energies_charges'] = (
                output_block)

        elif any([
            prop in self.output_properties
            for prop in ['atomic_energies', 'atomic_charges']]
        ):
            
            # Get property label
            if 'atomic_energies' in self.output_properties:
                prop = 'atomic_energies'
            else:
                prop = 'atomic_charges'
                self.output_atomic_charges = True

            # Number of property outputs
            output_n_property = 1

            # PhysNet energy only output block
            output_block = layer_physnet.OutputBlock(
                graph_n_blocks,
                n_atombasis,
                self.n_maxatom,
                output_n_property,
                self.output_n_residual,
                self.activation_fn,
                True,
                True,
                False,
                device=self.device,
                dtype=self.dtype)

            # Assign output block to dictionary
            self.output_property_block[prop] = output_block

        # Create further output blocks for properties with certain exceptions
        for prop in self.output_properties:

            # Skip deriving properties
            if prop in self._property_exclusion:
                continue

            # No output_block for already covered special properties
            if prop in self._property_special:
                continue

            # Number of property outputs
            output_n_property = 1

            # Initialize output block
            output_block = layer_physnet.OutputBlock(
                graph_n_blocks,
                n_atombasis,
                self.n_maxatom,
                output_n_property,
                self.output_n_residual,
                self.activation_fn,
                True,
                True,
                False,
                device=self.device,
                dtype=self.dtype)

            # Assign output block to dictionary
            self.output_property_block[prop] = output_block

        # Assign property and atomic properties scaling parameters
        if self.output_property_scaling is not None:
            self.set_property_scaling(self.output_property_scaling)

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
            'output_n_residual': self.output_n_residual,
            'output_activation_fn': self.output_activation_fn,
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
                split_scaling = block.split_scaling
            elif (
                prop == 'atomic_energies'
                and 'atomic_energies_charges' in self.output_property_block
            ):
                block = self.output_property_block['atomic_energies_charges']
                scaling = block.scaling.detach().cpu().numpy()
                atomic_scaling = block.atomic_scaling
                split_scaling = block.split_scaling
            else:
                raise SyntaxError(
                    f"No output block avaiable for property {prop:s}!")

            # Assign new scaling parameter
            if atomic_scaling:
                
                for ai, pars in scaling_parameters.get(prop).items():
                    
                    if split_scaling and prop == 'atomic_energies':
                        if set_shift_term:
                            scaling[ai, 0, 0] = pars[0]
                        if set_scaling_factor:
                            scaling[ai, 0, 1] = pars[1]

                    elif split_scaling and prop == 'atomic_charges':
                        if set_shift_term:
                            scaling[ai, 1, 0] = pars[0]
                        if set_scaling_factor:
                            scaling[ai, 1, 1] = pars[1]
                    else:
                        if set_shift_term:
                            scaling[ai, 0] = pars[0]
                        if set_scaling_factor:
                            scaling[ai, 1] = pars[1]

            else:
                
                pars = scaling_parameters.get(prop)

                if split_scaling and prop == 'atomic_energies':
                    if set_shift_term:
                        scaling[0, 0] = pars[0]
                    if set_scaling_factor:
                        scaling[0, 1] = pars[1]
                if split_scaling and prop == 'atomic_charges':
                    if set_shift_term:
                        scaling[1, 0] = pars[0]
                    if set_scaling_factor:
                        scaling[1, 1] = pars[1]
                else:
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
        for prop in self.output_property_block:

            block = self.output_property_block[prop]
            scaling = block.scaling.detach().cpu().numpy()

            # Check for atom or system resolved scaling
            if block.atomic_scaling:
                if block.split_scaling and prop == 'atomic_energies_charges':
                    scaling_parameters['atomic_energies'] = {}
                    scaling_parameters['atomic_charges'] = {}
                    for ai, pars in enumerate(scaling):
                        scaling_parameters['atomic_energies'][ai] = pars[0]
                        scaling_parameters['atomic_charges'][ai] = pars[1]
                    pass
                else:
                    scaling_parameters[prop] = {}
                    for ai, pars in enumerate(scaling):
                        scaling_parameters[prop][ai] = pars
            else:
                if block.split_scaling and prop == 'atomic_energies_charges':
                    scaling_parameters['atomic_energies'] = scaling[0]
                    scaling_parameters['atomic_charges'] = scaling[1]
                else:
                    scaling_parameters[prop] = scaling

        return scaling_parameters

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
            features_list : torch.Tensor(n_blocks, N_atoms, n_atombasis)
                List of atom feature vectors
            atomic_numbers: torch.Tensor(N_atoms)
                List of atomic numbers
            charge: torch.Tesnsor(N_system)
                Total charge of molecules in batch
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary of predicted properties

        """

       # Assign input data
        atomic_numbers = batch['atomic_numbers']
        features_list = batch['features_list']

        # Initialize additional loss value if training mode is active
        if self.training:
            batch['model_loss'] = torch.tensor(
                0.0, device=self.device, dtype=self.dtype)

        # Iterate over output blocks
        for prop, output_block in self.output_property_block.items():

            # Compute prediction and loss function contribution
            batch[prop], model_loss = output_block(
                features_list, atomic_numbers)

            # Update additional loss value if training mode is active
            if self.training:
                batch['model_loss'] = batch['model_loss'] + model_loss

        # Post-process atomic energies/charges case
        if self.output_energies_charges:

            batch['atomic_energies'], \
                batch['atomic_charges'] = (
                    batch['atomic_energies_charges'][:, 0],
                    batch['atomic_energies_charges'][:, 1])

        # Scale atomic charges to conserve correct molecular charge
        if self.output_atomic_charges:
            batch = self.conserve_charge(batch)

        # If verbose, store a copy of the output block prediction
        if verbose:
            for prop in self.output_property_block:
                if prop == 'atomic_energies_charges':
                    batch['output_atomic_energies'] = (
                        batch['atomic_energies'].detach())
                    batch['output_atomic_charges'] = (
                        batch['atomic_charges'].detach())
                else:
                    verbose_prop = 'output_{:s}'.format(prop)
                    batch[verbose_prop] = batch[prop].detach()

        return batch
