import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import ase
import torch

from asparagus import data
from asparagus import model
from asparagus import module
from asparagus import settings
from asparagus import utils

__all__ = ['BaseModel']

#======================================
# Calculator Models
#======================================

class BaseModel(torch.nn.Module): 
    """
    Asparagus calculator base model

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    _default_model_properties = ['energy', 'forces']

    _supported_model_properties = [
        'energy',
        'atomic_energies',
        'forces']

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
        **kwargs
    ):
        """
        Initialize BaseModel Calculator

        Parameters
        ----------
        config: (str, dict, object), optional, default None
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of Asparagus parameters
        device: str, optional, default global setting
            Device type for model variable allocation
        dtype: dtype object, optional, default global setting
            Model variables data type

        """

        super(BaseModel, self).__init__()
        self.model_type = 'BaseModel'
        
        # Initialize loaded checkpoint flag
        self.checkpoint_loaded = False
        self.checkpoint_file = None

        # Initialize general flags
        self.model_energy = False
        self.model_forces = False
        self.model_repulsion = False
        self.model_electrostatic = False
        self.model_dispersion = False
        self.model_atomic_charges = False
        self.model_atomic_dipoles = False
        self.model_dipole = False
        
        # Initialize ML/MM embedding flags
        self.model_mlmm_embedding = False
        self.model_ml_fragment = 0
        self.model_mm_fragment = 1

        return

    def __str__(self):
        return self.model_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return model and module information
        """
        return {}

    def load(
        self,
        checkpoint: Dict[str, Any],
        checkpoint_file: Optional[str] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Load model parameters from checkpoint file.

        Parameters
        ----------
        checkpoint: dict(str, Any)
            Torch module checkpoint file for the model calculator
        checkpoint_file: str, optional, default None
            Torch checkpoint file path for logger info

        """

        # Load model checkpoint file
        if checkpoint is None and verbose:
            
            if checkpoint_file is None:
                checkpoint_state = "."
            else:
                checkpoint_state = f" from file '{checkpoint_file:s}'."
            self.logger.info(
                f"No checkpoint file is loaded{checkpoint_state:s}")

        else:

            self.load_state_dict(
                checkpoint['model_state_dict'])
            self.checkpoint_loaded = True
            self.checkpoint_file = checkpoint_file

            if verbose:
                if checkpoint_file is None:
                    checkpoint_state = "."
                else:
                    checkpoint_state = f" from file '{checkpoint_file:s}'."
                self.logger.info(
                    f"Checkpoint file is loaded{checkpoint_state:s}")

        return

    def check_model_properties(
        self,
        config: settings.Configuration,
        model_properties: List[str],
    ) -> List[str]:
        """
        Check model property input.
        
        Parameters
        ----------
        config: settings.Configuration
            Asparagus settings.Configuration class object of global parameter
        model_properties: list(str)
            Properties to predict by calculator model

        Returns
        ----------
        list(str)
            Checked property labels

        """
        
        # Check model properties - Selection
        if model_properties is None:
            # If no model properties are defined but data properties are,
            # adopt all supported data properties as model properties,
            # else adopt default model properties
            if config.get('data_properties') is None:
                model_properties = self._default_model_properties
            else:
                model_properties = []
                for prop in config.get('data_properties'):
                    if prop in self._supported_model_properties:
                        model_properties.append(prop)

        # Check model properties - Labels
        for prop in model_properties:
            if not utils.check_property_label(prop, return_modified=False):
                raise SyntaxError(
                    f"Model property label '{prop:s}' is not a valid "
                    + "property label! Valid property labels are:\n"
                    + str(list(settings._valid_properties)))
        
        return list(model_properties)

    def set_model_energy_properties(
        self,
        model_properties: List[str],
        model_energy_properties: 
            Optional[List[str]] = ['atomic_energies', 'energy'],
    ) -> List[str]:
        """
        Set model energy property parameters.
        
        Parameters
        ----------
        model_properties: list(str)
            Properties to predict by calculator model
        model_energy_properties: list(str)
            Model energy related properties

        Returns
        ----------
        list(str)
            Checked property labels

        """

        # Check model properties - Energy and energy gradient properties
        if any([
            prop in model_properties
            for prop in model_energy_properties]
        ):
            self.model_energy = True
            for prop in model_energy_properties:
                if prop not in model_properties:
                    model_properties.append(prop)
        else:
            self.model_energy = False
        if 'hessian' in model_properties:
            self.model_forces = True
            self.model_hessian = True
        elif 'forces' in model_properties:
            self.model_forces = True
            self.model_hessian = False
        else:
            self.model_forces = False
            self.model_hessian = False
        if self.model_forces and not self.model_energy:
            raise SyntaxError(
                f"{self.model_type:s} Model cannot predict energy gradient "
                + "properties such as forces or hessians without predicting "
                + "energies!")

        return model_properties

    def check_model_property_units(
        self,
        model_properties: List[str],
        model_unit_properties: Dict[str, str],
        model_default_properties: 
            Optional[List[str]] = ['positions', 'charge'],
    ) -> Dict[str, str]:
        """
        Check model property units input.
        
        Parameters
        ----------
        model_properties: list(str)
            Properties to predict by calculator model
        model_unit_properties: dict
            Unit labels of the predicted model properties.
        model_default_properties: list(str), optional, 
                default ['positions', 'charge']
            Default properties where default unit settings are adopted even
            if not defined as model property.

        Returns
        ----------
        dict(str, str)
            Checked unit labels of the predicted model properties.

        """

        # Check property units input
        if model_unit_properties is None:
            model_unit_properties = {}

        # Initialize checked property units dictionary
        checked_unit_properties = {}

        # Check if default properties units are defined
        for prop in model_default_properties:
            if prop not in model_unit_properties:
                checked_unit_properties[prop] = (
                    settings._default_units[prop])
                self.logger.info(
                    f"Unit for property '{prop}' is set to the default unit "
                    + f"'{settings._default_units[prop]}'!")
            else:
                checked_unit_properties[prop] = (
                    model_unit_properties[prop])

        # Check if all model property units are defined
        for prop in model_properties:
            if prop not in model_unit_properties:
                # Check if a related property unit is defined
                for rel_props in settings._related_unit_properties:
                    if prop in rel_props:
                        for rprop in rel_props:
                            if rprop in model_unit_properties:
                                checked_unit_properties[prop] = (
                                    model_unit_properties[rprop])
                                break
                # Else, take default
                self.logger.warning(
                    f"No unit defined for property '{prop}'!\n"
                    + f"Default unit of '{settings._default_units[prop]}' "
                    + "will be used.")
                checked_unit_properties[prop] = (
                    settings._default_units[prop])
            else:
                checked_unit_properties[prop] = (
                    model_unit_properties[prop])

        return checked_unit_properties

    def check_cutoff_ranges(
        self,
        model_cutoff: float,
        model_cuton: float,
        model_switch_range: float,
    ) -> (float, float, float):
        """
        Check model cutoff range option.
        
        Parameters
        ----------
        model_cutoff: float
            Upper atom interaction cutoff
        model_cuton: float
            Lower atom pair distance to start interaction switch-off
        model_switch_range: float
            Atom interaction cutoff switch range to switch of interaction to 
            zero. If 'model_cuton' is defined, this input will be ignored.

        Returns
        ----------
        float
            Model start interaction switch-off range
        float
            Model interaction cutoff switch range

        """
        
        # Check lower cutoff switch-off range
        if model_cuton is None:
            if model_switch_range > model_cutoff:
                raise SyntaxError(
                    "Model cutoff switch-off range "
                    + f"({model_switch_range:.2f}) is larger than the "
                    + f"upper cutoff range ({model_cutoff:.2f})!")
            model_cuton = model_cutoff - model_switch_range
        elif model_cuton > model_cutoff:
            message = (
                    "Lower atom pair cutoff distance 'model_cuton' "
                    + f"({model_cuton:.2f}) is larger than the upper cutoff "
                    + f"distance ({model_cutoff:.2f})!")
            if model_switch_range is None:
                raise SyntaxError(message)
            else:
                model_cuton = model_cutoff - model_switch_range
                self.logger.warning(
                    f"{message:s}\n"
                    + "Lower atom pair cutoff distance is changed switched "
                    + f"to '{model_cuton:.2f}'.")
        else:
            model_switch_range = model_cutoff - model_cuton
        if model_cuton < 0.0:
            raise SyntaxError(
                "Lower atom pair cutoff distance 'model_cuton' is negative "
                + f"({model_cuton:.2f})!")
        
        return model_cuton, model_switch_range

    def get_cutoff_ranges(self) -> List[float]:
        """
        Get model cutoff or, eventually, short range descriptor and long
        range cutoff list.

        Return
        ------
        list(float)
            List of the long range model and, eventually, short range
            descriptor cutoff (if defined and not short range equal long range
            cutoff).

        """

        long_range_cutoff = self.model_cutoff
        if hasattr(self.module_dict['input'], 'input_radial_cutoff'):
            short_range_cutoff = (
                self.module_dict['input'].input_radial_cutoff)
            if short_range_cutoff != long_range_cutoff:
                cutoffs = [short_range_cutoff, long_range_cutoff]
        else:
            cutoffs = [long_range_cutoff]

        return cutoffs

    def get_mlmm_cutoff_ranges(self) -> List[float]:
        """
        Get model ML and MM atom pair cutoff or, eventually, short range 
        descriptor and long range cutoff list between

        Return
        ------
        list(float)
            List of ML and MM atom pair long range model and, eventually, short
            range descriptor cutoff (if defined and not short range equal long
            range cutoff).

        """

        if hasattr(self, 'model_mlmm_cutoff'):
            long_range_cutoff = self.model_mlmm_cutoff
        else:
            long_range_cutoff = self.model_cutoff
        if hasattr(self.module_dict['input'], 'input_mlmm_radial_cutoff'):
            short_range_cutoff = (
                self.module_dict['input'].input_mlmm_radial_cutoff)
            if short_range_cutoff != long_range_cutoff:
                mlmm_cutoffs = [short_range_cutoff, long_range_cutoff]
        elif hasattr(self.module_dict['input'], 'input_radial_cutoff'):
            short_range_cutoff = (
                self.module_dict['input'].input_radial_cutoff)
            if short_range_cutoff != long_range_cutoff:
                mlmm_cutoffs = [short_range_cutoff, long_range_cutoff]
        else:
            mlmm_cutoffs = [long_range_cutoff]

        return mlmm_cutoffs

    def base_modules_setup(
        self,
        config: settings.Configuration,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Setup model calculator base modules input, graph and output
        
        Parameters
        ----------
        config: settings.Configuration
            Asparagus settings.Configuration class object of global parameter
        **kwargs
            Base module options

        """

        # Check for input module object in configuration input,
        # otherwise initialize input module
        if config.get('input_module') is not None:

            input_module = config.get('input_module')

        else:

            if config.get('input_type') is None:
                input_type = self._default_modules.get('input_type')
            else:
                input_type = config.get('input_type')
            input_module = module.get_input_module(
                input_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                verbose=verbose,
                **kwargs)

        # Check for graph module object in configuration input,
        # otherwise initialize graph module
        if config.get('graph_module') is not None:

            graph_module = config.get('graph_module')

        else:

            if config.get('graph_type') is None:
                graph_type = self._default_modules.get('graph_type')
            else:
                graph_type = config.get('graph_type')
            graph_module = module.get_graph_module(
                graph_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                verbose=verbose,
                **kwargs)

        # Check for output module object in input,
        # otherwise initialize output module
        if config.get('output_module') is not None:

            output_module = config.get('output_module')

        else:

            if config.get('output_type') is None:
                output_type = self._default_modules.get('output_type')
            else:
                output_type = config.get('output_type')
            output_module = module.get_output_module(
                output_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                verbose=verbose,
                **kwargs)
        
        return input_module, graph_module, output_module

    def get_scaleable_properties(
        self
    ) -> List[str]:
        """
        Return list of properties which are scaled by a scaling factor and
        shift term, which initial guess are derived from reference data.

        Returns
        -------
        list(str)
            Scalable model properties

        """
        return self.module_dict['output'].output_properties

    def set_property_scaling(
        self,
        property_scaling: Optional[
            Dict[str, Union[List[float], Dict[int, List[float]]]]] = None,
        set_shift_term: Optional[bool] = True,
        set_scaling_factor: Optional[bool] = True,
    ):
        """
        Set property scaling factor and shift terms and set atomic type
        energies shift.

        Parameters
        ----------
        property_scaling: dict(str, (list(float), dict(int, float)), optional,
                default None
            Model property scaling parameter to shift and scale output module
            results
        set_shift_term: bool, optional, default True
            If True, set or update the shift term. Else, keep previous
            value.
        set_scaling_factor: bool, optional, default True
            If True, set or update the scaling factor. Else, keep previous
            value.

        """

        # Set property scaling factors and shift terms
        if (
            property_scaling is not None
            and hasattr(self.module_dict['output'], "set_property_scaling")
        ):

            self.module_dict['output'].set_property_scaling(
                property_scaling,
                set_shift_term=set_shift_term,
                set_scaling_factor=set_scaling_factor)

        return

    def get_property_scaling(
        self,
    ) -> Dict[str, Union[List[float], Dict[int, List[float]]]]:
        """
        Get current property scaling factor and shift term dictionary.

        Returns
        -------
        dict(str, (list(float), dict(int, float))
            Current model property scaling parameter to shift and scale output
            module results

        """

        # Get property scaling factors and shift terms
        if hasattr(self.module_dict['output'], "get_property_scaling"):
            property_scaling = self.module_dict['output'].get_property_scaling()
        else:
            property_scaling = {}

        return property_scaling

    def set_model_unit_properties(
        self,
        model_unit_properties: Dict[str, str],
    ):
        """
        Set or change unit property parameter in model layers or modules

        Parameters
        ----------
        model_unit_properties: dict
            Dictionary with the units of the model properties to initialize 
            correct conversion factors.

        """
        raise NotImplementedError

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
            if no_weight_decay and 'scaling' in name.split('.')[0].split('_'):
                trainable_parameters['no_weight_decay'].append(parameter)
            elif no_weight_decay and 'shift' in name.split('.')[0].split('_'):
                trainable_parameters['no_weight_decay'].append(parameter)
            else:
                trainable_parameters['default'].append(parameter)

        return trainable_parameters

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
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
                'ml_idx': torch.Tensor(n_atoms)
                    Primary atom indices for the supercluster approach
                'ml_idx_p': torch.Tensor(n_pairs)
                    Image atom to primary atom index pointer for the atom
                    pair indices in a supercluster
                'ml_idx_jp': torch.Tensor(n_pairs)
                    Atom j pair index pointer from image atom to respective
                    primary atom index in a supercluster

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """
        raise NotImplementedError

    def calculate(
        self,
        atoms: ase.Atoms,
        charge: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of the calculator model with an ASE Atoms object.

        Parameters
        ----------
        atoms: ase.Atoms
            ASE Atoms object to calculate properties
        charge: float, optional, default None
            Total system charge. If None, charge is estimated from atomic
            charges, if available, Else, charge is set as zero.

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Create atoms batch dictionary
        atoms_batch = self.create_batch(
            atoms,
            charge=charge)

        return self.forward(atoms_batch, **kwargs)

    def create_batch(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        charge: Optional[Union[float, List[float]]] = None,
        conversion: Optional[Dict[str, float]] = {},
    ) -> Dict[str, torch.Tensor]:
        """
        Create a systems batch dictionary as input for the model calculator.
        
        Parameters
        ----------
        atoms: (ase.Atoms, list(ase.Atoms))
            ASE Atoms object or list of ASE Atoms object to prepare batch
            dictionary.
        charge: (float list(float)), optional, default 0.0
            Total system charge or list of total system charge.
            If None, charge is estimated from the ASE Atoms objects, mostly
            set as zero.
        conversion: dict(str, float), optional, default {}
            ASE Atoms conversion dictionary from ASE units to model units.

        Returns
        -------
        dict(str, torch.Tensor)
            System(s) batch dictionary used as model calculator input

        """

        # Check atoms input
        if utils.is_ase_atoms(atoms):
            atoms = [atoms]
        elif not utils.is_ase_atoms_array(atoms):
            raise ValueError(
                "Input 'atoms' is not an ASE Atoms object or list of ASE "
                + "atoms objects!")

        # Initialize atoms batch
        batch = {}

        # Number of atoms
        batch['atoms_number'] = torch.tensor(
            [len(image) for image in atoms],
            device=self.device, dtype=torch.int64)

        # Atomic numbers properties
        batch['atomic_numbers'] = torch.cat(
            [
                torch.tensor(image.get_atomic_numbers(), dtype=torch.int64)
                for image in atoms
            ], 0).to(
                device=self.device, dtype=torch.int64)

        # Atom positions
        if conversion.get('positions') is None:
            fconv = 1.0
        else:
            fconv = conversion['positions']
        batch['positions'] = torch.cat(
            [
                torch.tensor(image.get_positions()*fconv, dtype=self.dtype)
                for image in atoms
            ], 0).to(
                device=self.device, dtype=self.dtype)

        # Atom periodic boundary conditions
        batch['pbc'] = torch.tensor(
            np.array([image.get_pbc() for image in atoms]),
            dtype=torch.bool, device=self.device)

        # Atom cell information
        if conversion.get('positions') is None:
            fconv = 1.0
        else:
            fconv = conversion['positions']
        batch['cell'] = torch.tensor(
            np.array([image.get_cell()[:]*fconv for image in atoms]),
            dtype=self.dtype, device=self.device)

        # Total atomic system charge
        if conversion.get('charge') is None:
            fconv = 1.0
        else:
            fconv = conversion['charge']
        if charge is None:
            try:
                charge = [np.sum(image.get_charges())*fconv for image in atoms]
            except RuntimeError:
                charge = [
                    np.sum(image.get_initial_charges())*fconv
                    for image in atoms]
        elif utils.is_numeric(charge):
            charge = [charge*fconv]*len(atoms)
        elif utils.is_numeric_array(charge):
            charge = np.array(charge)*fconv
        else:
            raise ValueError("Input 'charge' is not a numeric value or array!")
        batch['charge'] = torch.tensor(
            charge, dtype=self.dtype, device=self.device)

        # Compute atom pair indices
        batch = self.compute_neighborlist(batch)

        return batch

    def create_batch_copies(
        self,
        atoms: ase.Atoms,
        ncopies: Optional[int] = None,
        positions: Optional[List[float]] = None,
        cell: Optional[List[float]] = None,
        charge: Optional[float] = None,
        conversion: Optional[Dict[str, float]] = {},
    ) -> Dict[str, torch.Tensor]:
        """
        Create a systems batch dictionary as input for the model calculator.

        Parameters
        ----------
        atoms: (ase.Atoms)
            ASE Atoms object to prepare multiple copies in a batch dictionary.
        ncopies: int, optional, default None
            Number of copies of the ASE atoms system in the batch.
            If None, number of copies are taken from 'positions' or 'cell'
            input, otherwise is 1.
        positions: list(float), optional, default None
            Array of shape ('Ncopies', 'Natoms', 3) where 'Ncopies' is the
            number of copies of the ASE Atoms system in the batch and 'Natoms'
            is the number of atoms.
            If None, the positions of the ASE Atoms object is taken.
        cell: list(float), optional, default None
            Array of ASE Atoms cell parameter of shape ('Ncopies', 3).
            If None, the cell parameters from the ASE Atoms object is taken.
        charge: float, optional, default 0.0
            Total system charge of the ASE atoms object.
            If None, charge is estimated from the ASE Atoms objects, mostly
            set as zero.
        conversion: dict(str, float), optional, default {}
            ASE Atoms conversion dictionary from ASE units to model units.

        Returns
        -------
        dict(str, torch.Tensor)
            System(s) batch dictionary used as model calculator input

        """

        # Get number of copies
        if ncopies is None and positions is None and cell is None:
            ncopies = 1
        elif ncopies is None and positions is not None:
            ncopies = len(positions)
        elif ncopies is None and cell is not None:
            ncopies = len(cell)

        # Check positions and cell input
        if positions is not None and ncopies != len(positions):
            raise SyntaxError(
                f"Number of copies ({ncopies:d}) and positions "
                + f"({len(positions):d}) does not match!")
        if cell is not None and ncopies != len(cell):
            raise SyntaxError(
                f"Number of copies ({ncopies:d}) and cells "
                + f"({len(cell):d}) does not match!")

        # Initialize atoms batch
        batch = {}

        # Number of atoms
        batch['atoms_number'] = torch.tensor(
            [len(atoms)]*ncopies, device=self.device, dtype=torch.int64)

        # System segment index of atom i
        batch['sys_i'] = torch.repeat_interleave(
            torch.arange(ncopies, device=self.device, dtype=torch.int64),
            repeats=len(atoms), dim=0).to(
                device=self.device, dtype=torch.int64)

        # Atomic numbers properties
        batch['atomic_numbers'] = torch.cat(
            [
                torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64)
                for _ in range(ncopies)
            ], 0).to(
                device=self.device, dtype=torch.int64)

        # Atom positions
        if conversion.get('positions') is None:
            fconv = 1.0
        else:
            fconv = conversion['positions']
        if positions is None:
            batch['positions'] = torch.cat(
                [
                    torch.tensor(atoms.get_positions()*fconv, dtype=self.dtype)
                    for _ in range(ncopies)
                ], 0).to(
                    device=self.device, dtype=self.dtype)
        else:
            batch['positions'] = torch.cat(
                [
                    torch.tensor(positions_i, dtype=self.dtype)*fconv
                    for positions_i in positions
                ], 0).to(
                    device=self.device, dtype=self.dtype)

        # Atom periodic boundary conditions
        batch['pbc'] = torch.tensor(
            atoms.get_pbc().repeat(ncopies).reshape(ncopies, 3),
            dtype=torch.bool, device=self.device)

        # Atom cell information
        if conversion.get('positions') is None:
            fconv = 1.0
        else:
            fconv = conversion['positions']
        if cell is None:
            batch['cell'] = torch.tensor(
                (atoms.get_cell()[:]*fconv).repeat(ncopies).reshape(
                    ncopies, 3, 3),
                dtype=self.dtype, device=self.device)
        else:
            batch['cell'] = torch.tensor(
                [cell_i for cell_i in cell],
                dtype=self.dtype, device=self.device)*fconv

        # Total atomic system charge
        if conversion.get('charge') is None:
            fconv = 1.0
        else:
            fconv = conversion['charge']
        if charge is None:
            try:
                charge = [
                    np.sum(atoms.get_charges())*fconv for _ in range(ncopies)]
            except RuntimeError:
                charge = [
                    np.sum(atoms.get_initial_charges())*fconv
                    for _ in range(ncopies)]
        elif utils.is_numeric(charge):
            charge = [charge*fconv]*ncopies
        elif utils.is_numeric_array(charge):
            charge = np.array(charge)*fconv
        else:
            charge = [0.0]*ncopies
        batch['charge'] = torch.tensor(
            charge, dtype=self.dtype, device=self.device)

        # Compute atom pair indices
        if not hasattr(self, 'neighbor_list'):
            self.neighbor_list = module.TorchNeighborListRangeSeparated(
                self.get_cutoff_ranges(),
                self.device,
                self.dtype)
        batch = self.neighbor_list(batch)

        return batch

    def calculate_data(
        self,
        dataset: Union[data.DataContainer, data.DataSet, data.DataSubSet],
        batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of the calculator model with an Asparagus data set.

        Parameters
        ----------
        dataset: (data.DataContainer, data.DataSet, data.DataSubSet)
            Asparagus DataContainer or DataSet object
        batch_size: int, optional, default 32
            Data loader batch size
        num_workers: int, optional, default 1
            Number of data loader workers

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Prepare data loader for the data set
        dataloader = data.DataLoader(
            dataset,
            batch_size,
            [],
            False,
            num_workers,
            self.device,
            self.dtype)

        # Iterate over data batches
        for ib, batch in enumerate(dataloader):
            
            # Predict model properties from data batch
            prediction = self.forward(batch, **kwargs)

            # Append prediction to result dictionary
            for prop in self.model_properties:
                if results.get(prop) is None:
                    results[prop] = [prediction[prop].cpu().detach()]
                else:
                    results[prop].append(prediction[prop].cpu().detach())
    
        # Concatenate results
        for prop in self.model_properties:
            if ib and prediction[prop][0].shape:
                results[prop] = torch.cat(prediction[prop])
            elif ib:
                results[prop] = torch.cat(
                    [result.reshape(1) for result in prediction[prop]], dim=0)
            else:
                results[prop] = prediction[prop][0]

        return prediction

    def compute_neighborlist(
        self,
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute neighbor list for data batch

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of input and result data tensors
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Data batch including neighbor list data

        """
        
        # System segment index of atom i
        batch['sys_i'] = torch.repeat_interleave(
            torch.arange(
                batch['atoms_number'].shape[0],
                device=self.device, dtype=torch.int64),
            repeats=batch['atoms_number'], dim=0).to(
                device=self.device, dtype=torch.int64)

        # Compute neighbor list
        if not hasattr(self, 'neighbor_list'):
            self.neighbor_list = module.TorchNeighborListRangeSeparated(
                self.get_cutoff_ranges(),
                self.device,
                self.dtype)
        batch = self.neighbor_list(batch)

        # Compute ML-MM neighbor list
        if self.model_mlmm_embedding:
            if not hasattr(self, 'mlmm_neighbor_list'):
                self.mlmm_neighbor_list = (
                    module.TorchNeighborListRangeSeparatedMLMM(
                        self.get_mlmm_cutoff_ranges(),
                        self.device,
                        self.dtype)
                    )
            batch = self.mlmm_neighbor_list(batch)

        return batch

    def compute_energy(
        self,
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute system energy from atomic energy contributions

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of input and result data tensors
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """
        
        # Compute system energy
        batch['energy'] = torch.zeros_like(batch['charge']).scatter_add_(
            0,
            batch['sys_i'],
            batch['atomic_energies'])

        # If verbose, store system energy of each single atomic energy
        # contribution known
        if verbose:
            verbose_properties = [
                'output',
                'repulsion',
                'dispersion',
                'electrostatic'
                ]
            for prop in verbose_properties:
                verbose_prop = '{:s}_atomic_energies'.format(prop)
                if verbose_prop in batch:
                    verbose_prop_energy = '{:s}_energy'.format(prop)
                    batch[verbose_prop_energy] = torch.zeros_like(
                        batch['energy']
                        ).scatter_add_(
                            0,
                            batch['sys_i'],
                            batch[verbose_prop]
                        )

        return batch

    def compute_forces(
        self,
        batch: Dict[str, torch.Tensor],
        create_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute forces and, eventually, Hessian of the potential energy with
        respect to atom positions

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of input and result data tensors for gradient and 
            Hessian computation
        create_graph: bool, optional, default False
            Parameter for 'torch.autograd.grad' to force keeping derivative
            graph if set to true.

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """
        
        # Compute energy gradient, keep backpropagation graph only if model
        # is in training mode or Hessian is demanded
        gradient = torch.autograd.grad(
            [torch.sum(batch['energy']),],
            [batch['positions'],],
            create_graph=(
                self.training or self.model_hessian or create_graph)
        )[0]
        
        # Provoke crashing if forces are none
        assert gradient is not None
        batch['forces'] = -gradient
        
        # If demanded, compute Hessian (computationally expensive!)
        if self.model_hessian:
            dim_hessian = 3*batch['positions'].size(0)
            hessian = batch['energy'].new_zeros((dim_hessian, dim_hessian))
            for ig, grad_i in enumerate(gradient.view(-1)):
                hessian_ig = torch.autograd.grad(
                    [grad_i],
                    [batch['positions']],
                    retain_graph=(ig < dim_hessian))[0]
                if hessian_ig is not None:
                    hessian[ig] = hessian_ig.view(-1)
            batch['hessian'] = hessian

        return batch

    def compute_dipole(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the molecular dipole moment from atom-centred charges and
        atomic dipole moments
        
        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of input data and predicted properties

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary with additional molecular dipole prediction

        """
        
        # For supercluster method, just use primary cell atom positions
        if 'ml_idx' in batch:
            positions_dipole = batch['positions'][batch['ml_idx']]
        else:
            positions_dipole = batch['positions']

        # In case of non-zero system charges, shift origin to center of
        # mass
        atomic_masses = self.atomic_masses[batch['atomic_numbers']]
        system_mass = torch.zeros_like(batch['charge']).scatter_add_(
            0, batch['sys_i'], atomic_masses)

        system_com = torch.zeros(
            (batch['atoms_number'].size(0), 3),
            device=self.device, dtype=self.dtype)
        system_com = (
            system_com.scatter_add_(
                0, batch['sys_i'].unsqueeze(-1).repeat(1, 3),
                atomic_masses.unsqueeze(-1)*positions_dipole
                ).reshape(-1, 3)
            / system_mass.unsqueeze(-1)
            )
        positions_com = positions_dipole - system_com[batch['sys_i']]

        # Compute molecular dipole moment from atomic charges
        batch['dipole'] = torch.zeros_like(system_com).scatter_add_(
            0, batch['sys_i'].unsqueeze(-1).repeat(1, 3),
            batch['atomic_charges'].unsqueeze(-1)*positions_com)

        # Refine molecular dipole moment with atomic dipole moments
        if self.model_atomic_dipoles:
            batch['dipole'] = batch['dipole'].scatter_add_(
                0, batch['sys_i'].unsqueeze(-1).repeat(1, 3),
                batch['atomic_dipoles'])

        return batch

    def compute_quadrupole(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the molecular quadrupole moment from atom-centred charges and
        atomic quadrupole moments
        
        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of input data and predicted properties

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary with additional molecular quadrupole prediction

        """
        
        # For supercluster method, just use primary cell atom positions
        if 'ml_idx' in batch:
            positions_quadrupole = batch['positions'][batch['ml_idx']]
        else:
            positions_quadrupole = batch['positions']

        # Shift origin to center of geometry
        system_cog = torch.zeros(
            (batch['atoms_number'].size(0), 3),
            device=self.device, dtype=self.dtype)
        system_cog = (
            system_cog.scatter_add_(
                0,
                batch['sys_i'].unsqueeze(-1).repeat(1, 3),
                positions_quadrupole
                ).reshape(-1, 3)
            / batch['atoms_number']
            )
        positions_cog = positions_quadrupole - system_cog[batch['sys_i']]

        # Compute detraced outer product
        outer_product = positions_cog.unsqueeze(-1)*positions_cog.unsqueeze(-2)
        outer_product = (
            outer_product
            - torch.diag_embed(
                torch.tile(
                    outer_product.diagonal(
                        dim1=-2, dim2=-1).mean(
                            dim=-1, keepdim=True),
                    (1, 3)
                )
            )
        )

        # Compute molecular quadrupole moment from atomic charges
        batch['quadrupole'] = torch.zeros(
            (batch['atoms_number'].size(0), 3, 3),
            device=self.device,
            dtype=self.dtype).scatter_add_(
                0,
                batch['sys_i'].unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 3),
                batch['atomic_charges'].unsqueeze(-1).unsqueeze(-1)
                * outer_product
            )

        # Refine molecular quadrupole moment with atomic quadrupole moments
        if self.model_atomic_quadrupoles:
            batch['quadrupole'] = batch['quadrupole'].scatter_add_(
                0, batch['sys_i'].unsqueeze(-1).unsqueeze(-1).repeat(1, 3, 3),
                batch['atomic_quadrupoles'])

        return batch
