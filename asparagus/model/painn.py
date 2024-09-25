import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import ase
import torch

from asparagus import model
from asparagus import module
from asparagus import settings
from asparagus import utils

__all__ = ['Model_PaiNN']

#======================================
# Calculator Models
#======================================

class Model_PaiNN(model.BaseModel):
    """
    PaiNN calculator model

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
        Use close-range atom repulsion model.
    model_repulsion_trainable: bool, optional, default True
        If True, repulsion model parameter are trainable. Else, default
        parameter values are fix.
    model_electrostatic: bool, optional, default True
        Use electrostatic potential between atomic charges for energy
        prediction.
    model_dispersion: bool, optional, default True
        Use Grimme's D3 dispersion model for energy prediction.
    model_dispersion_trainable: bool, optional, default True
        If True, empirical parameter in the D3 dispersion model are
        trainable. If False, empirical parameter are fixed to default
    model_num_threads: int, optional, default 4
        Sets the number of threads used for intraop parallelism on CPU.

    """
    
    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for graph module
    _default_args = {
        'model_properties':             None,
        'model_unit_properties':        None,
        'model_cutoff':                 12.0,
        'model_cuton':                  None,
        'model_switch_range':           2.0,
        'model_repulsion':              False,
        'model_repulsion_trainable':    True,
        'model_electrostatic':          None,
        'model_dispersion':             True,
        'model_dispersion_trainable':   True,
        'model_num_threads':            4,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'model_properties':             [utils.is_string_array, utils.is_None],
        'model_unit_properties':        [utils.is_dictionary, utils.is_None],
        'model_cutoff':                 [utils.is_numeric],
        'model_cuton':                  [utils.is_numeric, utils.is_None],
        'model_switch_range':           [utils.is_numeric],
        'model_repulsion':              [utils.is_bool],
        'model_repulsion_trainable':    [utils.is_bool],
        'model_electrostatic':          [utils.is_bool, utils.is_None],
        'model_dispersion':             [utils.is_bool],
        'model_dispersion_trainable':   [utils.is_bool],
        'model_num_threads':            [utils.is_integer],
        }

    # Default module types of the model calculator
    _default_modules = {
        'input_type':                   'PaiNN',
        'graph_type':                   'PaiNN',
        'output_type':                  'PaiNN',
        }

    _default_model_properties = ['energy', 'forces', 'dipole']

    _supported_model_properties = [
        'energy',
        'atomic_energies',
        'forces',
        'atomic_charges',
        'dipole',
        'atomic_dipoles']

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_properties: Optional[List[str]] = None,
        model_unit_properties: Optional[Dict[str, str]] = None,
        model_cutoff: Optional[float] = None,
        model_cuton: Optional[float] = None,
        model_switch_range: Optional[float] = None,
        model_repulsion: Optional[bool] = None,
        model_repulsion_trainable: Optional[bool] = None,
        model_electrostatic: Optional[bool] = None,
        model_dispersion: Optional[bool] = None,
        model_dispersion_trainable: Optional[bool] = None,
        model_num_threads: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        **kwargs
    ):
        """
        Initialize PaiNN Calculator model.

        """

        super(Model_PaiNN, self).__init__()
        model_type = 'PaiNN'

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
        config.update(config_update)

        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)

        # Set model calculator number of threads
        if self.model_num_threads is not None:
            torch.set_num_threads(self.model_num_threads)

        ###################################
        # # # Check PaiNN Model Input # # #
        ###################################

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
                'atomic_charges', 'atomic_dipoles', 'dipole'])

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
        config.update(config_update)

        ###############################
        # # # PaiNN Modules Setup # # #
        ###############################

        # Assign model calculator base modules
        self.input_module, self.graph_module, self.output_module = (
            self.base_modules_setup(
                config,
                **kwargs)
            )

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
                "Repulsion energy contribution is requested without "
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
            # Get Ziegler-Biersack-Littmark style nuclear repulsion potential
            self.repulsion_module = module.ZBL_repulsion(
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
                d3_s6=d3_s6,
                d3_s8=d3_s8,
                d3_a1=d3_a1,
                d3_a2=d3_a2,
            )

        #####################################
        # # # PaiNN Miscellaneous Setup # # #
        #####################################
        
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

        # Collect model info
        if hasattr(self.input_module, "get_info"):
            info = {**info, **self.input_module.get_info()}
        if hasattr(self.graph_module, "get_info"):
            info = {**info, **self.graph_module.get_info()}
        if hasattr(self.output_module, "get_info"):
            info = {**info, **self.output_module.get_info()}
        if (
            self.model_repulsion
            and hasattr(self.repulsion_module, "get_info")
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
            'atomic_charges', 'atomic_dipoles', 'dipole'],
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
            self.model_atomic_dipoles = True
            self.model_dipole = True
            for prop in model_electrostatics_properties:
                if prop not in model_properties:
                    model_properties.append(prop)
        elif 'atomic_dipoles' in self.model_properties:
            self.model_atomic_charges = True
            self.model_atomic_dipoles = True
            self.model_dipole = False
            for prop in model_electrostatics_properties:
                if (
                    prop not in model_properties
                    and prop not in ['dipole']
                ):
                    model_properties.append(prop)
        elif 'atomic_charges' in self.model_properties:
            self.model_atomic_charges = True
            self.model_atomic_dipoles = False
            self.model_dipole = False
            for prop in model_electrostatics_properties:
                if (
                    prop not in model_properties
                    and prop not in ['dipole', 'atomic_dipoles']
                ):
                    model_properties.append(prop)
        else:
            self.model_atomic_dipoles = False
            self.model_atomic_charges = False
            self.model_dipole = False

        return model_properties

    #def set_property_scaling(
        #self,
        #scaling_parameter: Optional[Dict[str, List[float]]] = None,
        #atomic_energies_shifts: Optional[Dict[Union[int, str], float]] = None
    #):
        #"""
        #Prepare property scaling factor and shift terms and set atomic type
        #energies shift.

        #"""

        ## Set property scaling factors and shift terms
        #if scaling_parameter is not None:
            #self.output_module.set_property_scaling(scaling_parameter)

        ## Set atomic type energies shift
        #if atomic_energies_shifts is not None:
            #self.output_module.set_atomic_energies_shift(
                #atomic_energies_shifts)

        #return

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

    # @torch.jit.export  # No effect, as 'forward' already is
    # @torch.compile # Not supporting double backwards autograd (forces, loss)
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        no_derivation: Optional[bool] = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of PaiNN Calculator model.

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
                    Atom j pair index pointer from image atom to repsective
                    primary atom index in a supercluster
        no_derivation: bool, optional, default False
            If False, predict all properties even if backwards derivation is
            required such as forces. Else, only predict non-derived properties.

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

        # Activate back propagation if derivatives with regard to
        # atom positions is requested.
        if self.model_forces:
            positions.requires_grad_(True)

        # Run input model
        features, distances, vectors, cutoffs, rbfs, distances_uv = (
            self.input_module(
                atomic_numbers, positions,
                idx_i, idx_j, pbc_offset_ij=pbc_offset_ij,
                idx_u=idx_u, idx_v=idx_v, pbc_offset_uv=pbc_offset_uv)
        )

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
        sfeatures, efeatures = self.graph_module(
            features, distances, vectors, cutoffs, rbfs, idx_i, idx_j)

        # Run output model
        results = self.output_module(
            sfeatures,
            efeatures,
            atomic_numbers=atomic_numbers)

        # Add repulsion model contribution
        if self.model_repulsion:
            results['atomic_energies'] = (
                results['atomic_energies']
                + self.repulsion_module(
                    atomic_numbers, distances, cutoffs, idx_i, idx_j))

        # Add dispersion model contributions
        if self.model_dispersion:
            results['atomic_energies'] = (
                results['atomic_energies']
                + self.dispersion_module(
                    atomic_numbers, distances_uv, idx_u, idx_v))

        # Scale atomic charges to ensure correct total charge
        if self.model_atomic_charges:
            charge_deviation = (
                charge - utils.scatter_sum(
                    results['atomic_charges'], sys_i, dim=0,
                    shape=charge.shape)
                / atoms_number
            )
            results['atomic_charges'] = (
                results['atomic_charges'] + charge_deviation[sys_i])

        # Add electrostatic model contribution
        if self.model_electrostatic:
            # Apply electrostatic model
            results['atomic_energies'] = (
                results['atomic_energies']
                + self.electrostatic_module(
                    results, distances_uv, idx_u, idx_v))

        # Compute property - Energy
        if self.model_energy:
            results['energy'] = torch.squeeze(
                utils.scatter_sum(
                        results['atomic_energies'], sys_i, dim=0,
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
                self.logger.warning(
                    "Error in force calculation "
                    + "(backpropagation)!")
                results['forces'] = torch.zeros_like(positions)

            if self.model_hessian:
                hessian = results['energy'].new_zeros(
                    (gradient.size(0), gradient.size(0)))
                for ig in range(gradient.size(0)):
                    hessian_ig = torch.autograd.grad(
                        [gradient[ig]],
                        positions,
                        retain_graph=(ig < gradient.size(0)))[0]
                    if hessian_ig is not None:
                        hessian[ig] = hessian_ig.view(-1)
                results['hessian'] = hessian

        # Compute molecular dipole
        if self.model_dipole:

            # For supercluster method, just use primary cell atom positions
            if pbc_atoms is None:
                positions_dipole = positions
            else:
                positions_dipole = positions[pbc_atoms]

            # For non-zero system charges, shift origin to center of mass
            if torch.any(charge):
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
            else:
                positions_com = positions_dipole

            # Compute molecular dipole moment from atomic charges
            results['dipole'] = utils.scatter_sum(
                results['atomic_charges'][..., None]*positions_com,
                sys_i, dim=0, shape=(*atoms_number.shape, 3)
                ).reshape(-1, 3)

            # Refine molecular dipole moment with atomic dipole moments
            if self.model_atomic_dipoles:
                results['dipole'] = (
                    results['dipole'] + utils.scatter_sum(
                        results['atomic_dipoles'],
                        sys_i, dim=0, shape=(*atoms_number.shape, 3)
                        ).reshape(-1, 3)
                )

        return results
