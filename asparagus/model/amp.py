import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import ase
import torch

from asparagus import model
from asparagus import module
from asparagus import settings
from asparagus import utils

__all__ = ['Model_AMP']

#======================================
# Calculator Models
#======================================

class Model_AMP(model.BaseModel):
    """
    "Anisotropic Message Passing" (AMP) model calculator

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to config json file (str)
    model_properties: list(str), optional, default '['energy', 'forces']'
        Properties to predict by calculator model
    model_unit_properties: dict, optional, default {}
        Unit labels of the predicted model properties. If not defined,
        prediction results are assumed as ASE units but for during training the
        units from the reference data container are adopted.
    model_ml_fragment: int, optional, default 0:
        Fragment number index that marks all ML atoms in the system.
    model_mm_fragment: int, optional, default 1:
        Fragment number index that marks all Mm atoms in the system.
    model_cutoff: float, optional, default 12.0
        Upper ML-ML atom pair long-range interaction cutoff
    model_cuton: float, optional, default None
        Lower ML-ML atom pair distance to start long-range 
        interaction switch-off
    model_switch_range: float, optional, default 2.0
        ML-ML and ML-MM atom pair long-range interaction cutoff switch range to
        switch of long-range interaction to zero.
        If 'model_cuton' is defined, this input will be ignored.
    model_mlmm_cutoff: float, optional, default None
        Upper ML-MM atom pair long-range interaction cutoff. If None, the
        'model_cutoff' value is used.
    model_repulsion: bool, optional, default False
        Use close-range atom repulsion model.
    model_repulsion_cutoff: float, optional, default 1.0
        Nuclear repulsion model cutoff range.
    model_repulsion_cuton: float, optional, default 0.0
        Nuclear repulsion model inner cutoff (cuton) radii to start
        switch-off function.
    model_repulsion_trainable: bool, optional, default True
        If True, repulsion model parameter are trainable. Else, default
        parameter values are fix.
    model_electrostatic: bool, optional, default None
        Use long-range electrostatic potential between atomic charges for 
        energy prediction. If None, electrostatic potential model is applied if
        atomic charges are available.
    model_electrostatic_dipole: bool, optional, default True
        Include atomic dipole moments to compute the electrostatic potential
        between atom pairs for energy prediction if available.
    model_electrostatic_quadrupole: bool, optional, default True
        Include atomic quadrupole moments to compute the electrostatic
        potential between atom pairs for energy prediction if available.
    model_dispersion: bool, optional, default True
        Use Grimme's D3 dispersion model for energy prediction.
    model_dispersion_trainable: bool, optional, default False
        If True, empirical parameter in the D3 dispersion model are
        trainable. If False, empirical parameter are fixed to default
    model_mlmm_embedding: bool, optional, default True
        Use ML/MM embedding scheme if two fragements (ML and MM) are available
        in the input.
    model_num_threads: int, optional, default 4
        Sets the number of threads used for intraop parallelism on CPU.
    device: str, optional, default global setting
        Device type for model variable allocation
    dtype: dtype object, optional, default global setting
        Model variables data type

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for AMP model
    _default_args = {
        'model_properties':             None,
        'model_unit_properties':        None,
        'model_ml_fragment':            0,
        'model_mm_fragment':            1,
        'model_cutoff':                 12.0,
        'model_cuton':                  None,
        'model_switch_range':           2.0,
        'model_mlmm_cutoff':            None,
        'model_repulsion':              False,
        'model_repulsion_cutoff':       1.0,
        'model_repulsion_cuton':        0.0,
        'model_repulsion_trainable':    True,
        'model_electrostatic':          None,
        'model_electrostatic_dipole':   True,
        'model_electrostatic_quadrupole':
            True,
        'model_dispersion':             True,
        'model_dispersion_trainable':   False,
        'model_mlmm_embedding':         True,
        'model_num_threads':            4,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'model_properties':             [utils.is_string_array, utils.is_None],
        'model_unit_properties':        [utils.is_dictionary, utils.is_None],
        'model_cutoff':                 [utils.is_numeric],
        'model_cuton':                  [utils.is_numeric, utils.is_None],
        'model_switch_range':           [utils.is_numeric],
        'model_mlmm_cutoff':            [utils.is_numeric],
        'model_repulsion':              [utils.is_bool],
        'model_repulsion_cutoff':       [utils.is_numeric],
        'model_repulsion_cuton':        [utils.is_numeric, utils.is_None],
        'model_repulsion_trainable':    [utils.is_bool],
        'model_electrostatic':          [utils.is_bool, utils.is_None],
        'model_electrostatic_dipole':   [utils.is_bool],
        'model_electrostatic_quadrupole':
            [utils.is_bool],
        'model_dispersion':             [utils.is_bool],
        'model_dispersion_trainable':   [utils.is_bool],
        'model_mlmm_embedding':         [utils.is_bool],
        'model_num_threads':            [utils.is_integer],
        }
    
    # Model type label
    _model_type = 'AMP'

    # Default module types of the model calculator
    _default_modules = {
        'input_type':                   'AMP',
        'graph_type':                   'AMP',
        'output_type':                  'AMP',
        }

    _default_model_properties = ['energy', 'forces', 'dipole']

    _supported_model_properties = [
        'energy',
        'atomic_energies',
        'forces',
        'atomic_charges',
        'dipole',
        'atomic_dipoles',
        'quadrupole',
        'atomic_quadrupoles',
        ]

    _required_input_properties = ['atomic_charges']

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_properties: Optional[List[str]] = None,
        model_unit_properties: Optional[Dict[str, str]] = None,
        model_ml_fragment: Optional[int] = None,
        model_mm_fragment: Optional[int] = None,
        model_cutoff: Optional[float] = None,
        model_cuton: Optional[float] = None,
        model_switch_range: Optional[float] = None,
        model_mlmm_cutoff: Optional[float] = None,
        model_repulsion: Optional[bool] = None,
        model_repulsion_cutoff: Optional[float] = None,
        model_repulsion_cuton: Optional[float] = None,
        model_repulsion_trainable: Optional[bool] = None,
        model_electrostatic: Optional[bool] = None,
        model_electrostatic_dipole: Optional[bool] = None,
        model_dispersion: Optional[bool] = None,
        model_dispersion_trainable: Optional[bool] = None,
        model_mlmm_embedding: Optional[bool] = None,
        model_num_threads: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize AMP calculator model.

        """

        super(Model_AMP, self).__init__()
        model_type = 'AMP'

        #############################
        # # # Check Class Input # # #
        #############################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, model),
            check_dtype=utils.get_dtype_args(self, model))

        # Update global configuration dictionary
        config.update(
            config_update,
            verbose=verbose)

        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)

        # Set model calculator number of threads
        if self.model_num_threads is not None:
            torch.set_num_threads(self.model_num_threads)

        #################################
        # # # Check AMP Model Input # # #
        #################################

        # Check model properties
        self.model_properties = self.check_model_properties(
            config,
            self.model_properties)

        # Check model properties - Energy and energy gradient properties
        self.model_properties = self.set_model_energy_properties(
            self.model_properties,
            model_energy_properties=['atomic_energies', 'energy'])

        # Check model properties - Electrostatics properties
        self.model_properties = self.set_model_electrostatic_properties(
            self.model_properties,
            model_electrostatics_properties=[
                'atomic_charges',
                'atomic_dipoles', 'dipole',
                'atomic_quadrupoles', 'quadrupole']
            )

        # Check model property units
        self.model_unit_properties = self.check_model_property_units(
            self.model_properties,
            self.model_unit_properties,
            model_default_properties=['positions', 'charge'])

        # Check lower cutoff switch-off range
        self.model_cuton, self.model_switch_range = self.check_cutoff_ranges(
            self.model_cutoff,
            self.model_cuton,
            self.model_switch_range)

        # Check ML-MM cutoff
        if self.model_mlmm_cutoff is None:
            self.model_mlmm_cutoff = self.model_cutoff

        # Update global configuration dictionary
        config_update = {
            'model_properties': self.model_properties,
            'model_unit_properties': self.model_unit_properties,
            'model_cutoff': self.model_cutoff,
            'model_cuton': self.model_cuton,
            'model_switch_range': self.model_switch_range,
            'model_mlmm_cutoff': self.model_mlmm_cutoff}
        config.update(
            config_update,
            verbose=verbose)

        #############################
        # # # AMP Modules Setup # # #
        #############################

        # Assign model calculator base modules
        input_module, graph_module, output_module = (
            self.base_modules_setup(
                config,
                verbose=verbose,
                **kwargs)
            )

        # Initialize module dictionary with base modules
        self.module_dict = torch.nn.ModuleDict({
            'input': input_module,
            'graph': graph_module,
            'output': output_module,
            })

        # If electrostatic energy contribution is undefined, activate 
        # contribution if atomic charges are predicted.
        if self.model_electrostatic is None:
            if self.model_atomic_charges:
                self.model_electrostatic = True
            else:
                self.model_electrostatic = False

        # Check repulsion, electrostatic and dispersion module requirement
        if self.model_repulsion and not self.model_energy:
            raise SyntaxError(
                "Nuclear repulsion energy contribution is requested without "
                + "having 'energy' assigned as model property!")
        if self.model_electrostatic and not self.model_energy:
            raise SyntaxError(
                "Electrostatic energy contribution is requested without "
                + "having 'energy' assigned as model property!")
        if self.model_electrostatic and not self.model_atomic_charges:
            raise SyntaxError(
                "Electrostatic energy contribution is requested without "
                + "having 'atomic_charges' or 'dipole' assigned as model "
                + "property!")
        if self.model_dispersion and not self.model_energy:
            raise SyntaxError(
                "Dispersion energy contribution is requested without "
                + "having 'energy' assigned as model property!")

        # Assign atom repulsion module
        if self.model_repulsion:

            # Check nuclear repulsion cutoff
            input_radial_cutoff = config.get('input_radial_cutoff')
            if (
                input_radial_cutoff is not None
                and self.model_repulsion_cutoff > input_radial_cutoff
            ):
                raise SyntaxError(
                    "Nuclear repulsion cutoff radii is larger than the "
                    + "input module radial cutoff!")

            # Assign Ziegler-Biersack-Littmark style nuclear repulsion
            # potential
            repulsion_module = module.ZBL_repulsion(
                self.model_repulsion_cutoff,
                self.model_repulsion_cuton,
                self.model_repulsion_trainable,
                self.device,
                self.dtype,
                unit_properties=self.model_unit_properties,
                **kwargs)
            self.module_dict['repulsion'] = repulsion_module

        # Assign ML/ML and ML/MM electrostatic interaction module
        if self.model_electrostatic:

            electrostatic_module = module.Damped_electrostatics(
                self.model_cutoff,
                config.get('input_radial_cutoff'),
                self.device,
                self.dtype,
                unit_properties=self.model_unit_properties,
                truncation='force',
                atomic_dipoles=self.model_atomic_dipoles,
                atomic_quadrupoles=self.model_atomic_quadrupoles,
                **kwargs)
            self.module_dict['electrostatic'] = electrostatic_module

            mlmm_electrostatic_module = module.MLMM_electrostatics(
                self.model_mlmm_cutoff,
                self.device,
                self.dtype,
                unit_properties=self.model_unit_properties,
                truncation='None',
                atomic_dipoles=self.model_atomic_dipoles,
                atomic_quadrupoles=self.model_atomic_quadrupoles,
                **kwargs)
            self.module_dict['mlmm_electrostatic'] = mlmm_electrostatic_module

        # Assign dispersion interaction module
        if self.model_dispersion:

            # Grep dispersion correction parameters
            d3_s6 = config.get("model_dispersion_d3_s6")
            d3_s8 = config.get("model_dispersion_d3_s8")
            d3_a1 = config.get("model_dispersion_d3_a1")
            d3_a2 = config.get("model_dispersion_d3_a2")

            # Get Grimme's D3 dispersion model calculator
            dispersion_module = module.D3_dispersion(
                self.model_cutoff,
                self.model_cuton,
                self.model_dispersion_trainable,
                self.device,
                self.dtype,
                unit_properties=self.model_unit_properties,
                truncation='force',
                d3_s6=d3_s6,
                d3_s8=d3_s8,
                d3_a1=d3_a1,
                d3_a2=d3_a2,
            )
            self.module_dict['dispersion'] = dispersion_module

        ###################################
        # # # AMP Miscellaneous Setup # # #
        ###################################
        
        # Assign atomic masses list for center of mass recentering
        if self.model_dipole:
            
            # Convert atomic masses list to requested data type
            self.atomic_masses = torch.tensor(
                utils.atomic_masses,
                device=self.device,
                dtype=self.dtype)

        return

    def __str__(self):
        return self.model_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return model and module information
        """

        # Initialize info dictionary
        info = {}

        # Collect module info
        if hasattr(self.module_dict['input'], "get_info"):
            info = {**info, **self.module_dict['input'].get_info()}
        if hasattr(self.module_dict['graph'], "get_info"):
            info = {**info, **self.module_dict['graph'].get_info()}
        if hasattr(self.module_dict['output'], "get_info"):
            info = {**info, **self.module_dict['output'].get_info()}
        if (
            self.model_repulsion
            and hasattr(self.module_dict['repulsion'], "get_info")
        ):
            info = {**info, **self.module_dict['repulsion'].get_info()}
        if (
            self.model_electrostatic
            and hasattr(self.module_dict['electrostatic'], "get_info")
        ):
            info = {**info, **self.module_dict['electrostatic'].get_info()}
        if (
            self.model_electrostatic
            and hasattr(self.module_dict['mlmm_electrostatic'], "get_info")
        ):
            info = {
                **info, **self.module_dict['mlmm_electrostatic'].get_info()}
        if (
            self.model_dispersion
            and hasattr(self.module_dict['dispersion'], "get_info")
        ):
            info = {**info, **self.module_dict['dispersion'].get_info()}

        return {
            **info,
            'model_type': self._model_type,
            'model_properties': self.model_properties,
            'model_unit_properties': self.model_unit_properties,
            'model_cutoff': self.model_cutoff,
            'model_cuton': self.model_cuton,
            'model_switch_range': self.model_switch_range,
            'model_repulsion': self.model_repulsion,
            'model_repulsion_trainable': self.model_repulsion_trainable,
            'model_electrostatic': self.model_electrostatic,
            'model_dispersion': self.model_dispersion,
            'model_dispersion_trainable': self.model_dispersion_trainable,
        }

    def set_model_electrostatic_properties(
        self,
        model_properties: List[str],
        model_electrostatics_properties: Optional[List[str]] = [
            'atomic_charges',
            'atomic_dipoles', 'dipole',
            'atomic_quadrupoles', 'quadrupole'],
    ) -> List[str]:
        """
        Set model energy property parameters.
        
        Parameters
        ----------
        model_properties: list(str)
            Properties to predict by calculator model
        model_electrostatic_properties: list(str)
            Model electrostatics related properties

        Returns
        ----------
        list(str)
            Checked property labels

        """

        # Check model properties - Electrostatics properties
        if 'quadrupole' in model_properties:
            self.model_atomic_charges = True
            self.model_atomic_dipoles = True
            self.model_dipole = True
            self.model_atomic_quadrupoles = True
            self.model_quadrupole = True
            for prop in model_electrostatics_properties:
                if prop not in model_properties:
                    model_properties.append(prop)
        elif 'dipole' in model_properties:
            self.model_atomic_charges = True
            self.model_atomic_dipoles = True
            self.model_dipole = True
            self.model_atomic_quadrupoles = False
            self.model_quadrupole = False
            for prop in model_electrostatics_properties:
                if (
                    prop not in model_properties
                    and prop not in ['quadrupole', 'atomic_quadrupoles']
                ):
                    model_properties.append(prop)
        elif 'atomic_charges' in self.model_properties:
            self.model_atomic_charges = True
            self.model_atomic_dipoles = False
            self.model_dipole = False
            self.model_atomic_quadrupoles = False
            self.model_quadrupole = False
            for prop in model_electrostatics_properties:
                if (
                    prop not in model_properties
                    and prop not in [
                        'quadrupole', 'atomic_quadrupoles',
                        'dipole', 'atomic_dipoles']
                ):
                    model_properties.append(prop)
        else:
            self.model_atomic_charges = False
            self.model_atomic_dipoles = False
            self.model_dipole = False
            self.model_atomic_quadrupoles = False
            self.model_quadrupole = False

        return model_properties

    def set_model_unit_properties(
        self,
        model_unit_properties: Dict[str, str],
    ):
        """
        Set or change unit property parameter in respective model layers

        Parameter
        ---------
        model_unit_properties: dict
            Unit labels of the predicted model properties

        """

        # Change unit properties for electrostatic and dispersion layers
        if self.model_electrostatic:
            # Synchronize total and atomic charge units
            if model_unit_properties.get('charge') is not None:
                model_unit_properties['atomic_charges'] = (
                    model_unit_properties.get('charge'))
            elif model_unit_properties.get('atomic_charges') is not None:
                model_unit_properties['charge'] = (
                    model_unit_properties.get('atomic_charges'))
            else:
                raise SyntaxError(
                    "For electrostatic potential contribution either the"
                    + "model unit for the 'charge' or 'atomic_charges' must "
                    + "be defined!")
            self.module_dict['electrostatic'].set_unit_properties(
                model_unit_properties)
        if self.model_dispersion:
            self.module_dict['dispersion'].set_unit_properties(
                model_unit_properties)

        return

    def get_trainable_parameters(
        self,
        no_weight_decay: Optional[bool] = True,
    ) -> Dict[str, List]:
        """
        Return a  dictionary of lists for different optimizer options.

        Parameters
        ----------
        no_weight_decay: bool, optional, default True
            Separate parameters on which weight decay should not be applied

        Returns
        -------
        dict(str, List)
            Dictionary of trainable model parameters. Contains 'default' entry
            for all parameters not affected by special treatment. Further
            entries are, if true, the parameter names of the input
        """

        # Trainable parameter dictionary
        trainable_parameters = {}
        trainable_parameters['default'] = []
        if no_weight_decay:
            trainable_parameters['no_weight_decay'] = []

        # Iterate over all trainable model parameters
        for name, parameter in self.named_parameters():
            # Catch all parameters to not apply weight decay on
            if no_weight_decay and 'output_scaling' in name:
                trainable_parameters['no_weight_decay'].append(parameter)
            elif no_weight_decay and 'dispersion_module' in name:
                trainable_parameters['no_weight_decay'].append(parameter)
            else:
                trainable_parameters['default'].append(parameter)

        return trainable_parameters

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        no_derivation: bool = False,
        create_graph: bool = False,
        verbose_results: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of AMP calculator model.

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of input data tensors for forward pass.
            Basic keys are:
                'atoms_number': torch.Tensor(n_systems)
                    Number of atoms per molecule in batch
                'atomic_numbers': torch.Tensor(n_atoms)
                    Atomic numbers of the batch of molecules
                'positions': torch.Tensor(n_atoms, 3)
                    Atomic positions of the batch of molecules
                'charge': torch.Tensor(n_systems)
                    Total charge of molecules in batch
                'idx_i': torch.Tensor(n_pairs)
                    Atom i pair index
                'idx_j': torch.Tensor(n_pairs)
                    Atom j pair index
                'sys_i': torch.Tensor(n_atoms)
                    System indices of atoms in batch
            Extra keys are:
                'pbc_offset': torch.Tensor(n_pairs)
                    Periodic boundary atom pair vector offset
                'ml_idx': torch.Tensor(n_atoms)
                    Primary atom indices for the supercluster approach
                'ml_idx_p': torch.Tensor(n_pairs)
                    Image atom to primary atom index pointer for the atom
                    pair indices in a supercluster
                'ml_idx_jp': torch.Tensor(n_pairs)
                    Atom j pair index pointer from image atom to repsective
                    primary atom index in a supercluster
        no_derivation: bool, optional, default False
            If True, only predict non-derived properties.
            Else, predict all properties even if backwards derivation is
            required (e.g. forces).
        create_graph: bool, optional, default False
            Parameter for 'torch.autograd.grad' to force keeping derivative
            graph if set to true. Necessary when further derivatives needs to
            be computed from the results.
        verbose_results: bool, optional, default False
            If True, store extended model property contributions in the result
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Activate back propagation if derivatives with regard to
        # atom positions is requested.
        if self.model_forces and not no_derivation:
            batch['positions'].requires_grad_(True)

        # Run modules
        for module in self.module_dict.values():
            batch = module(batch, verbose=verbose_results)

        # Compute property - Energy
        if self.model_energy:
            batch = self.compute_energy(batch, verbose=verbose_results)

        # Compute gradients and Hessian if demanded
        if self.model_forces and not no_derivation:
            batch = self.compute_forces(batch, create_graph=create_graph)

        # Compute molecular dipole
        if self.model_dipole:
            batch = self.compute_dipole(batch)

        # Compute molecular quadrupole
        if self.model_quadrupole:
            batch = self.compute_quadrupole(batch)

        return batch
