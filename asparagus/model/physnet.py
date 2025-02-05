import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import ase
import torch

from asparagus import model
from asparagus import module
from asparagus import settings
from asparagus import utils

__all__ = ['Model_PhysNet']

#======================================
# Calculator Models
#======================================

class Model_PhysNet(model.BaseModel): 
    """
    PhysNet model calculator

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
    model_cutoff: float, optional, default 12.0
        Upper atom interaction cutoff
    model_cuton: float, optional, default None
        Lower atom pair distance to start interaction switch-off
    model_switch_range: float, optional, default 2.0
        Atom interaction cutoff switch range to switch of interaction to zero.
        If 'model_cuton' is defined, this input will be ignored.
    model_repulsion: bool, optional, default False
        Use close-range nuclear repulsion model.
    model_repulsion_cutoff: float, optional, default 1.0
        Nuclear repulsion model cutoff range.
    model_repulsion_cuton: float, optional, default 0.0
        Nuclear repulsion model inner cutoff (cuton) radii to start
        switch-off function.
    model_repulsion_trainable: bool, optional, default True
        If True, repulsion model parameter are trainable. Else, default
        parameter values are fix.
    model_electrostatic: bool, optional, default True
        Use electrostatic potential between atomic charges for energy
        prediction.
    model_dispersion: bool, optional, default True
        Use Grimme's D3 dispersion model for energy prediction.
    model_dispersion_trainable: bool, optional, default False
        If True, empirical parameter in the D3 dispersion model are
        trainable. If False, empirical parameter are fixed to default
    model_num_threads: int, optional, default None
        Sets the number of threads used for intraop parallelism on CPU.
        if None, no thread number is set.
    device: str, optional, default global setting
        Device type for model variable allocation
    dtype: dtype object, optional, default global setting
        Model variables data type

    """
    
    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for PhysNet model
    _default_args = {
        'model_properties':             None,
        'model_unit_properties':        None,
        'model_cutoff':                 12.0,
        'model_cuton':                  None,
        'model_switch_range':           2.0,
        'model_repulsion':              False,
        'model_repulsion_cutoff':       1.0,
        'model_repulsion_cuton':        0.0,
        'model_repulsion_trainable':    True,
        'model_electrostatic':          None,
        'model_dispersion':             True,
        'model_dispersion_trainable':   False,
        'model_num_threads':            None,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'model_properties':             [utils.is_string_array, utils.is_None],
        'model_unit_properties':        [utils.is_dictionary, utils.is_None],
        'model_cutoff':                 [utils.is_numeric],
        'model_cuton':                  [utils.is_numeric, utils.is_None],
        'model_switch_range':           [utils.is_numeric],
        'model_repulsion':              [utils.is_bool],
        'model_repulsion_cutoff':       [utils.is_numeric],
        'model_repulsion_cuton':        [utils.is_numeric, utils.is_None],
        'model_repulsion_trainable':    [utils.is_bool],
        'model_electrostatic':          [utils.is_bool, utils.is_None],
        'model_dispersion':             [utils.is_bool],
        'model_dispersion_trainable':   [utils.is_bool],
        'model_num_threads':            [utils.is_integer, utils.is_None],
        }

    # Model type label
    _model_type = 'PhysNet'

    # Default module types of the model calculator
    _default_modules = {
        'input_type':                   'PhysNet',
        'graph_type':                   'PhysNet',
        'output_type':                  'PhysNet',
        }

    _default_model_properties = ['energy', 'forces', 'dipole']

    _supported_model_properties = [
        'energy',
        'atomic_energies',
        'forces',
        'atomic_charges',
        'dipole']

    def __init__(
        self,
        config: Optional[Union[str, dict, settings.Configuration]] = None,
        config_file: Optional[str] = None,
        model_properties: Optional[List[str]] = None,
        model_unit_properties: Optional[Dict[str, str]] = None,
        model_cutoff: Optional[float] = None,
        model_cuton: Optional[float] = None,
        model_switch_range: Optional[float] = None,
        model_repulsion: Optional[bool] = None,
        model_repulsion_cutoff: Optional[float] = None,
        model_repulsion_cuton: Optional[float] = None,
        model_repulsion_trainable: Optional[bool] = None,
        model_electrostatic: Optional[bool] = None,
        model_dispersion: Optional[bool] = None,
        model_dispersion_trainable: Optional[bool] = None,
        model_num_threads: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize PhysNet Calculator model.

        """

        super(Model_PhysNet, self).__init__()
        self.model_type = 'PhysNet'

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

        #####################################
        # # # Check PhysNet Model Input # # #
        #####################################

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
            model_electrostatics_properties=['atomic_charges', 'dipole'])

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

        # Update global configuration dictionary
        config_update = {
            'model_properties': self.model_properties,
            'model_unit_properties': self.model_unit_properties,
            'model_cutoff': self.model_cutoff,
            'model_cuton': self.model_cuton,
            'model_switch_range': self.model_switch_range}
        config.update(
            config_update,
            verbose=verbose)

        #################################
        # # # PhysNet Modules Setup # # #
        #################################

        # Assign model calculator base modules
        self.input_module, self.graph_module, self.output_module = (
            self.base_modules_setup(
                config,
                verbose=verbose,
                **kwargs)
            )

        # If electrostatic energy contribution is undefined, activate 
        # contribution if atomic charges are predicted.
        if self.model_electrostatic is None:
            if self.model_energy and self.model_atomic_charges:
                self.model_electrostatic = True
            else:
                self.model_electrostatic = False

        # Check repulsion, electrostatic and dispersion module requirement
        if self.model_repulsion and not self.model_energy:
            self.logger.error(
                "Repulsion energy contribution is requested without "
                + "having 'energy' assigned as model property!\n"
                + "Repulsion potential module will not be used!")
            self.model_repulsion = False
        if self.model_electrostatic and not self.model_energy:
            self.logger.error(
                "Electrostatic energy contribution is requested without "
                + "having 'energy' assigned as model property!\n"
                + "Electrostatic potential module will not be used!")
            self.model_electrostatic = False
        if self.model_electrostatic and not self.model_atomic_charges:
            self.logger.error(
                "Electrostatic energy contribution is requested without "
                + "having 'atomic_charges' or 'dipole' assigned as model "
                + "property!\n"
                + "Electrostatic potential module will not be used!")
            self.model_electrostatic = False
        if self.model_dispersion and not self.model_energy:
            self.logger.error(
                "Dispersion energy contribution is requested without "
                + "having 'energy' assigned as model property!\n"
                + "Dispersion potential module will not be used!")
            self.model_dispersion = False

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
            # Get Ziegler-Biersack-Littmark style nuclear repulsion potential
            self.repulsion_module = module.ZBL_repulsion(
                self.model_repulsion_cutoff,
                self.model_repulsion_cuton,
                self.model_repulsion_trainable,
                self.device,
                self.dtype,
                unit_properties=self.model_unit_properties,
                **kwargs)

        # Assign electrostatic interaction module
        if self.model_electrostatic:
            # Get electrostatic point charge model calculator
            self.electrostatic_module = module.PC_damped_electrostatics(
                self.model_cutoff,
                config.get('input_radial_cutoff'),
                self.device,
                self.dtype,
                unit_properties=self.model_unit_properties,
                truncation='force',
                **kwargs)

        # Assign dispersion interaction module
        if self.model_dispersion:

            # Grep dispersion correction parameters
            d3_s6 = config.get("model_dispersion_d3_s6")
            d3_s8 = config.get("model_dispersion_d3_s8")
            d3_a1 = config.get("model_dispersion_d3_a1")
            d3_a2 = config.get("model_dispersion_d3_a2")

            # Get Grimme's D3 dispersion model calculator
            self.dispersion_module = module.D3_dispersion(
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

        #######################################
        # # # PhysNet Miscellaneous Setup # # #
        #######################################
        
        # Assign atomic masses list for center of mass calculation
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
        if hasattr(self.input_module, "get_info"):
            info = {**info, **self.input_module.get_info()}
        if hasattr(self.graph_module, "get_info"):
            info = {**info, **self.graph_module.get_info()}
        if hasattr(self.output_module, "get_info"):
            info = {**info, **self.output_module.get_info()}
        if (
            self.model_repulsion
            and hasattr(self.electrostatic_module, "get_info")
        ):
            info = {**info, **self.repulsion_module.get_info()}
        if (
            self.model_electrostatic
            and hasattr(self.electrostatic_module, "get_info")
        ):
            info = {**info, **self.electrostatic_module.get_info()}
        if (
            self.model_dispersion
            and hasattr(self.dispersion_module, "get_info")
        ):
            info = {**info, **self.dispersion_module.get_info()}

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
        model_electrostatics_properties: 
            Optional[List[str]] = ['atomic_charges', 'dipole'],
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
        if 'dipole' in model_properties:
            self.model_atomic_charges = True
            self.model_dipole = True
            for prop in model_electrostatics_properties:
                if prop not in model_properties:
                    model_properties.append(prop)
        elif 'atomic_charges' in model_properties:
            self.model_atomic_charges = True
            self.model_dipole = False
        else:
            self.model_atomic_charges = False
            self.model_dipole = False

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
            self.electrostatic_module.set_unit_properties(
                model_unit_properties)
        if self.model_dispersion:
            self.dispersion_module.set_unit_properties(model_unit_properties)

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

    # @torch.compile # Not supporting backwards propagation with torch.float64
    # @torch.jit.export  # No effect, as 'forward' already is
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        no_derivation: Optional[bool] = False,
        verbose_results: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of PhysNet Calculator model.

        Parameters
        ----------
        batch : dict(str, torch.Tensor)
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
                'pbc_atoms': torch.Tensor(n_atoms)
                    Primary atom indices for the supercluster approach
                'pbc_idx': torch.Tensor(n_pairs)
                    Image atom to primary atom index pointer for the atom
                    pair indices in a supercluster
                'pbc_idx_j': torch.Tensor(n_pairs)
                    Atom j pair index pointer from image atom to respective
                    primary atom index in a supercluster
        no_derivation: bool, optional, default False
            If True, only predict non-derived properties.
            Else, predict all properties even if backwards derivation is
            required (e.g. forces).
        verbose_results: bool, optional, default False
            If True, store extended model property contributions in the result
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Assign input
        atoms_number = batch['atoms_number']
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions']
        charge = batch['charge']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        idx_u = batch.get('idx_u')
        idx_v = batch.get('idx_v')
        sys_i = batch['sys_i']

        # PBC: Cartesian offset method
        pbc_offset_ij = batch.get('pbc_offset_ij')
        pbc_offset_uv = batch.get('pbc_offset_uv')

        # PBC: Supercluster method
        pbc_atoms = batch.get('pbc_atoms')
        pbc_idx_pointer = batch.get('pbc_idx')
        pbc_idx_j = batch.get('pbc_idx_j')

        # Activate back propagation if derivatives with regard to atom positions
        # is requested.
        if self.model_forces:
            positions.requires_grad_(True)

        # Run input model
        features, distances, cutoffs, rbfs, distances_uv = self.input_module(
            atomic_numbers, positions,
            idx_i, idx_j, pbc_offset_ij=pbc_offset_ij,
            idx_u=idx_u, idx_v=idx_v, pbc_offset_uv=pbc_offset_uv)

        # PBC: Supercluster approach - Point from image atoms to primary atoms
        if pbc_idx_pointer is not None:
            idx_i = pbc_idx_pointer[idx_i]
            idx_j = pbc_idx_pointer[pbc_idx_j]

        # Check long-range atom pair indices
        if idx_u is None:
            # Assign atom pair indices
            idx_u = idx_i
            idx_v = idx_j
        elif pbc_idx_pointer is not None:
            idx_u = pbc_idx_pointer[idx_u]
            idx_v = pbc_idx_pointer[idx_v]

        # Run graph model
        features_list = self.graph_module(
            features, distances, cutoffs, rbfs, idx_i, idx_j)

        # Run output model
        results = self.output_module(
            features_list,
            atomic_numbers=atomic_numbers)
        if verbose_results:
            for prop in self.output_module.output_properties:
                verbose_prop = f"output_{prop:s}"
                results[verbose_prop] = results[prop].detach()

        # Add repulsion model contribution
        if self.model_repulsion:
            repulsion_atomic_energies = self.repulsion_module(
                atomic_numbers, distances, idx_i, idx_j)
            results['atomic_energies'] = (
                results['atomic_energies'] + repulsion_atomic_energies)
            if verbose_results:
                results['repulsion_atomic_energies'] = (
                    repulsion_atomic_energies.detach())

        # Add dispersion model contributions
        if self.model_dispersion:
            dispersion_atomic_energies = self.dispersion_module(
                atomic_numbers, distances_uv, idx_u, idx_v)
            results['atomic_energies'] = (
                results['atomic_energies'] + dispersion_atomic_energies)
            if verbose_results:
                results['dispersion_atomic_energies'] = (
                    dispersion_atomic_energies.detach())

        # Scale atomic charges to ensure correct total charge
        if self.model_atomic_charges:
            charge_deviation = (
                charge - utils.scatter_sum(
                    results['atomic_charges'], sys_i, dim=0,
                    shape=charge.shape))/atoms_number
            results['atomic_charges'] = (
                results['atomic_charges'] + charge_deviation[sys_i])

        # Add electrostatic model contribution
        if self.model_electrostatic:
            electrostatic_atomic_energies = self.electrostatic_module(
                results, distances_uv, idx_u, idx_v)
            results['atomic_energies'] = (
                results['atomic_energies'] + electrostatic_atomic_energies)
            if verbose_results:
                results['electrostatic_atomic_energies'] = (
                    electrostatic_atomic_energies.detach())

        # Compute property - Energy
        if self.model_energy:
            results['energy'] = torch.squeeze(
                utils.scatter_sum(
                    results['atomic_energies'], sys_i, dim=0,
                    shape=atoms_number.shape)
            )
            if verbose_results:
                atomic_energies_properies = [
                    prop for prop in results
                    if 'atomic_energies' in prop[-len('atomic_energies'):]]
                for prop in atomic_energies_properies:
                    verbose_prop = (
                        f"{prop[:-len('atomic_energies')]:s}energy")
                    results[verbose_prop] = torch.squeeze(
                        utils.scatter_sum(
                            results[prop], sys_i, dim=0,
                            shape=atoms_number.shape)
                    )

        # Compute gradients and Hessian if demanded
        if self.model_forces and not no_derivation:

            gradient = torch.autograd.grad(
                torch.sum(results['energy']),
                positions,
                create_graph=True)[0]

            # Avoid crashing if forces are none
            if gradient is not None:
                results['forces'] = -gradient
            else:
                self.logger(
                    "WARNING:\nError in force calculation "
                    + "(backpropagation)!")
                results['forces'] = torch.zeros_like(positions)

            if self.model_hessian:
                hessian = results['energy'].new_zeros(
                    (3*gradient.size(0), 3*gradient.size(0)))
                #for ig in range(3*gradient.size(0)):
                for ig, grad_i in enumerate(gradient.view(-1)):
                    hessian_ig = torch.autograd.grad(
                        [grad_i],
                        positions,
                        retain_graph=(ig < 3*gradient.size(0)))[0]
                    if hessian_ig is not None:
                        hessian[ig] = hessian_ig.view(-1)
                results['hessian'] = hessian

        # Compute molecular dipole if demanded
        if self.model_dipole:

            # For supercluster method, just use primary cell atom positions
            if pbc_atoms is None:
                positions_dipole = positions
            else:
                positions_dipole = positions[pbc_atoms]

            # In case of non-zero system charges, shift origin to center of
            # mass
            atomic_masses = self.atomic_masses[atomic_numbers]
            system_mass = utils.scatter_sum(
                atomic_masses, sys_i, dim=0,
                shape=atoms_number.shape)
            system_com = (
                utils.scatter_sum(
                    atomic_masses[..., None]*positions_dipole,
                    sys_i, dim=0, shape=(*atoms_number.shape, 3)
                    ).reshape(-1, 3)
                )/system_mass[..., None]
            positions_com = positions_dipole - system_com[sys_i]

            # Compute molecular dipole moment from atomic charges
            results['dipole'] = utils.scatter_sum(
                results['atomic_charges'][..., None]*positions_com,
                sys_i, dim=0, shape=(*atoms_number.shape, 3)
                ).reshape(-1, 3)

        return results
