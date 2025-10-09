import sys
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Callable, Any

import torch

from asparagus import utils
from asparagus import layer

try:
    # Import OpenMM modules if possible
    import openmm
    import openmmtorch
    from openmmml import mlpotential

except ModuleNotFoundError as e:
    raise ImportError(
        "OpenMM is not installed. Install it before continuing."
    ) from e

#except ModuleNotFoundError:

#    pass
    

__all__ = ['OpenMM_Calculator']

OpenMM_calculator_units = {
    'positions':        'nm',
    'energy':           'kJ/mol',
    'atomic_energies':  'kJ/mol',
    'forces':           'kJ/mol/nm',
    'hessian':          'kJ/mol/nm/nm',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'e*Ang',
    'atomic_dipoles':   'e*Ang',
    'mass':             'amu',
    }


class AsparagusPotentialImplFactory(mlpotential.MLPotentialImplFactory):
    """This is the factory that creates OpenMM_Calculator objects."""

    def createImpl(
        self,
        name: str,
        model_calculator: Union[torch.nn.Module, List[torch.nn.Module]],
        **args,
    ) -> mlpotential.MLPotentialImpl:

        return OpenMM_Calculator(
            name,
            model_calculator,
            **args,
            )

# # Register Asparagus potential factory to OpenMM-ML registry
# mlpotential.MLPotential.registerImplFactory(
#     'asparagus', AsparagusPotentialImplFactory())
    
class OpenMM_Calculator(mlpotential.MLPotentialImpl):
    """
    Interface between Asparagus potential models to OpenMM.

    Parameters
    ----------
    model_calculator: torch.nn.Module
        Asparagus model calculator object with already loaded parameter set

    """

    def __init__(
        self,
        name: str,
        model_calculator: Union[torch.nn.Module, List[torch.nn.Module]],
        **args
    ):

        # Assign device and dtype
        self.device = model_calculator.device
        self.dtype = model_calculator.dtype

        ################################
        # # # Set Model Calculator # # #
        ################################

        # Assign model calculator
        self.model_calculator = model_calculator

        # Get implemented model properties
        self.model_properties = (
            self.model_calculator.model_properties)
        self.model_unit_properties = (
            self.model_calculator.model_unit_properties)

        # Check if model calculator has loaded a checkpoint file or stored
        # if model parameters are stored in a checkpoint file
        if not self.model_calculator.checkpoint_loaded:
            raise SyntaxError(
                "The model calculator does not seem to have a proper "
                + "parameter set loaded from a checkpoint file.\nMake sure "
                + "model parameters are loaded.")

        # Get property unit conversions from model units to OpenMM units
        self.model2openmm_unit_conversion = {}

        # Positions unit conversion
        conversion, _ = utils.check_units(
            OpenMM_calculator_units['positions'],
            self.model_unit_properties.get('positions'))
        self.model2openmm_unit_conversion['positions'] = torch.tensor(
                conversion, device=self.device, dtype=self.dtype)

        # Implemented property units conversion
        for prop in self.model_properties:
            conversion, _ = utils.check_units(
                OpenMM_calculator_units[prop],
                self.model_unit_properties.get(prop))
            self.model2openmm_unit_conversion[prop] = torch.tensor(
                conversion, device=self.device, dtype=self.dtype)

        return

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: List[int],
        forceGroup: int,
        ml_atomic_numbers: Optional[List[int]] = None,
        ml_charge: Optional[float] = 0.0,
        ml_fluctuating_charges: Optional[bool] = None,
        mlmm_atomic_charges: Optional[List[float]] = None,
        mlmm_cutoff: Optional[float] = None,
        mlmm_cuton: Optional[float] = None,
        ml_cutoff_shell: Optional[float] = 0.2,
        mlmm_cutoff_shell: Optional[float] = 0.2,
        mlmm_lambda: Optional[float] = None,
        **kwargs,
    ):
        """
        Add the Asparagus model force class to the OpenMM System.

        Parameters
        ----------
        topology: openmm.app.Topology
            The topology of the system
        system: openmm.System
            The system to which the force will be added
        atoms: list(int)
            The indices of the atoms to include in the model. If 'None',
            all atoms are included
        forceGroup: int
            The force group to which the force should be assigned
        ml_atom_indices: list(int)
            List of atom indices referring to the ML treated atoms in the total 
            system loaded in CHARMM
        ml_atomic_numbers: list(int), optional, default None
            Respective atomic numbers of the ML atom selection. If 'None',
            get ML atomic numbers from openmm topology.
        ml_charge: float, optional, default 0.0
            Total charge of the partial ML atom selection
        ml_fluctuating_charges: bool
            If True, electrostatic interaction contribution between the MM atom
            charges and the model predicted ML atom charges. Else, the ML atom
            charges are considered fixed as defined by the CHARMM psf file.
        mlmm_atomic_charges: list(float), optional, default None
            List of all atomic charges of the system.
            If 'None', atomic charges are taken from 'NonbondedForce'.
            If 'ml_fluctuating_charges' is True and the Asparagus potential
            model does support atomic charge predictions, this atomic charges
            definitiion of the ML atoms are ignored.
        mlmm_cutoff: float, optional, default None
            Interaction cutoff distance for ML/MM electrostatic interactions.
            If None, get cutoff from the OpenMM non-bonded method.
        mlmm_cuton: float, optional, default None
            Lower atom pair distance to start interaction switch-off for ML/MM
            electrostatic interactions.
            If None, get switch-off from the OpenMM non-bonded method.
        ml_cutoff_shell: float, optional, default 0.2 nm
            Buffer region range, that ML atoms positions can change before a
            neighbor list update is performed.
        mlmm_cutoff_shell: float, optional, default 0.2 nm
            Buffer region range, that ML and MM atoms positions can change
            before a neighbor list update is performed.
        mlmm_lambda: float, optional, default None
            ML/MM electrostatic interactions scaling factor. If None, no
            scaling is applied.
        **kwargs
            Additional keyword arguments.

        """

        #######################################
        # # # Prepare ML system Parameter # # #
        #######################################
        
        # ML atom indices
        if atoms is None:
            self.ml_atom_indices = torch.tensor(
                [index for index in range(topology._numAtoms)],
                device=self.device,
                dtype=torch.int64)
        else:
            self.ml_atom_indices = torch.tensor(
                atoms, device=self.device, dtype=torch.int64)

        # Number of machine learning (ML) atoms
        self.ml_num_atoms = torch.tensor(
            len(self.ml_atom_indices), device=self.device, dtype=torch.int64)

        # ML atomic numbers
        if ml_atomic_numbers is None:
            try:
                self.ml_atomic_numbers = torch.tensor(
                    [
                        atom.element._atomic_number
                        for index, atom in enumerate(topology.atoms())
                        if index in self.ml_atom_indices
                    ],
                    device=self.device,
                    dtype=torch.int64)
            except TypeError:
                raise TypeError(
                    "Automatic atomic numbers detection failed. Provide "
                    + "a list of atomic number of the whole system with the "
                    + "keyword argument 'species', or just for the QM atoms "
                    + "but together with the QM atom indices 'atoms'.")
        else:
            self.ml_atomic_numbers = torch.tensor(
                ml_atomic_numbers, device=self.device, dtype=torch.int64)

        if (
            self.ml_atomic_numbers.shape[0] != self.ml_atom_indices.shape[0]
        ):
            raise ValueError(
                "Number of defined ML atomic numbers "
                + f"({self.ml_atomic_numbers.shape[0]:d}) differs from "
                + "the number of ML atom indices "
                + f"({self.ml_atom_indices.shape[0]:d})!")

        # ML atom total charge
        self.ml_charge = torch.tensor(
            [ml_charge], device=self.device, dtype=self.dtype)

        # ML fluctuating charges
        if (
            ml_fluctuating_charges is None
            and 'atomic_charges' in self.model_properties
        ):
            self.ml_fluctuating_charges = True
        elif ml_fluctuating_charges is None:
            self.ml_fluctuating_charges = False
        elif (
            ml_fluctuating_charges
            and 'atomic_charges' not in self.model_properties
        ):
            raise SyntaxError(
                "Fluctuating charge model option is enabled but the "
                + "potential model does not predict atomic charges!")
        else:
            self.ml_fluctuating_charges = ml_fluctuating_charges

        # Set cutoff shell parameter
        self.ml_cutoff_shell = torch.tensor(
            ml_cutoff_shell, device=self.device, dtype=self.dtype)

        ##########################################
        # # # Prepare ML/MM system Parameter # # #
        ##########################################

        # Number of ML and MM atoms
        self.mlmm_num_atoms = torch.tensor(
            topology._numAtoms, device=self.device, dtype=torch.int64)

        # ML and MM atom charges
        if mlmm_atomic_charges is None:
            for force in system.getForces():
                if (
                    'NonbondedForce' in str(force)
                    and hasattr(force, 'getParticleParameters')
                ):
                    mlmm_atomic_charges = torch.tensor(
                        [
                            force.getParticleParameters(index)[0]._value
                            for index in range(topology._numAtoms)
                        ],
                        device=self.device,
                        dtype=self.dtype)
            if self.ml_num_atoms == topology._numAtoms:
                mlmm_atomic_charges = torch.zeros(
                    self.ml_num_atoms, device=self.device, dtype=self.dtype)
            if mlmm_atomic_charges is None:
                raise ValueError(
                    "System atom centred charges couldn't be extracted from "
                    + "the system topology. Please provide the atomic charges "
                    + "of the system when creating the sytem "
                    + "('createSystem', 'createMixedSystem').")
            else:
                self.mlmm_atomic_charges = mlmm_atomic_charges
        else:
            self.mlmm_atomic_charges = torch.tensor(
                mlmm_atomic_charges, device=self.device, dtype=self.dtype)

        # ML and MM number of atoms
        self.mlmm_num_atoms = len(self.mlmm_atomic_charges)

        # In case of fluctuating charge option, set ML charges to zero
        if self.ml_fluctuating_charges:
            for force in system.getForces():
                if (
                    'NonbondedForce' in str(force)
                    and hasattr(force, 'getParticleParameters')
                ):
                    for index in self.ml_atom_indices:
                        nb_params = force.getParticleParameters(index)
                        nb_params[0]._value = 0.0
                        force.setParticleParameters(index, *nb_params)

        # Non-bonding interaction range
        if mlmm_cutoff is None:
            self.mlmm_cutoff = (
                1.0/self.model2openmm_unit_conversion['positions'])
            for force in system.getForces():
                if (
                    'NonbondedForce' in str(force)
                    and hasattr(force, 'getCutoffDistance')
                ):
                    self.mlmm_cutoff = (
                        force.getCutoffDistance()._value
                        / self.model2openmm_unit_conversion['positions']
                        )
        elif hasattr(mlmm_cutoff, '_value'):
            self.mlmm_cutoff = (
                mlmm_cutoff._value
                / self.model2openmm_unit_conversion['positions']
                )
        else:
            self.mlmm_cutoff = (
                mlmm_cutoff/self.model2openmm_unit_conversion['positions'])
        
        if mlmm_cuton is None:
            self.mlmm_cuton = (
                0.8/self.model2openmm_unit_conversion['positions'])
            for force in system.getForces():
                if (
                    'NonbondedForce' in str(force)
                    and hasattr(force, 'getCutoffDistance')
                ):
                    self.mlmm_cuton = (
                        self.mlmm_cutoff - (
                            force.getSwitchingDistance()._value
                            / self.model2openmm_unit_conversion['positions']
                            )
                        )
        elif hasattr(mlmm_cuton, '_value'):
            self.mlmm_cuton = (
                mlmm_cuton._value
                / self.model2openmm_unit_conversion['positions']
                )
        else:
            self.mlmm_cuton = (
                mlmm_cuton/self.model2openmm_unit_conversion['positions'])

        # Set cutoff shell parameter
        self.mlmm_cutoff_shell = torch.tensor(
            mlmm_cutoff_shell, device=self.device, dtype=self.dtype)

        # Non-bonding electrostatic scaling factor
        if mlmm_lambda is None:
            self.mlmm_lambda = torch.tensor(
                1.0, device=self.device, dtype=self.dtype)
        else:
            self.mlmm_lambda = torch.tensor(
                mlmm_lambda, device=self.device, dtype=self.dtype)

        # Get periodic boundary condtions
        self.pbc = (
            topology.getPeriodicBoxVectors() is not None
        ) or system.usesPeriodicBoundaryConditions()

        ###########################
        # # # Prepare Modules # # #
        ###########################

        # If fluctuating charge option is enabled, initialize the 
        # ML-MM electrostatic interaction calculator
        if self.ml_fluctuating_charges:

            self.mlmm_electrostatics_calc = MLMM_electrostatics(
                self.mlmm_cutoff,
                self.mlmm_cuton,
                self.device,
                self.dtype,
                self.model_unit_properties,
                self.mlmm_lambda,
                atomic_dipoles=self.model_calculator.model_atomic_dipoles,
                **kwargs)

        else:

            self.mlmm_electrostatics_calc = None

        # initialize ML-ML atoms neighbor list generator with cutoff in model
        # units
        self.ml_cutoffs = torch.tensor(
            self.model_calculator.get_cutoff_ranges(),
            device=self.device,
            dtype=self.dtype
            )
        self.ml_fragment = 0
        self.ml_neighbor_list_calc = (
            asparagus_module.TorchNeighborListRangeSeparated(
                    self.ml_cutoffs,
                    self.device,
                    self.dtype,
                    fragment=self.ml_fragment)
            )

        # initialize ML-MM atoms neighbor list generator with cutoff in model
        # units
        self.mm_fragment = 1
        self.mlmm_neighbor_list_calc = (
            asparagus_module.TorchNeighborListRangeSeparatedMLMM(
                self.mlmm_cutoff,
                self.device,
                self.dtype,
                ml_fragment=self.ml_fragment,
                mm_fragment=self.mm_fragment)
            )

        class ModelForce(torch.nn.Module):
            """
            Asparagus model potential force class for OpenMM.

            Parameters
            ----------
            model_calculator: callable
                The prepared Asparagus potential model calculation funcion
            ml_atom_indices: torch.Tensor(int)
                Indices of the atoms to use with the model. 
                If 'None', all atoms are used.
            ml_atomic_numbers: torch.Tensor(int)
                Atomic numbers of the ML atoms in the system
            ml_charge: int
                Total charge of the ML atom system
            mlmm_atomic_charges: torch.Tensor
                Atom centred point charges of the ML and MM atoms
            pbc: bool
                Perdiodic boundary conditions of the system
            ml_neighbor_list_calc: callable
                ML/ML atoms neighbor list generator function
            mlmm_neighbor_list_calc: callable
                ML/MM atoms neighbor list generator function
            mlmm_electrostatics_calc: callable
                ML/MM atoms electrostatic interaction calculator function
            ml_cutoff_shell: float
                Buffer region distance for ML atoms in nm
            mlmm_cutoff_shell: float
                Buffer region distance for ML to MM atoms in nm
            ml_fragment: int
                ML atoms fragment index
            mm_fragment: int
                MM atoms fragment index
            conversion_positions: float
                The length conversion factor from the model units to
                nanometers
            conversion_energy: float
                The energy conversion factor from the model units to kJ/mol
            device: str
                Device type for variable allocation
            dtype: dtype object
                Floating point variables data type

            """
            
            def __init__(
                self,
                model_calculator: torch.nn.Module,
                ml_atom_indices: torch.Tensor,
                ml_atomic_numbers: torch.Tensor,
                ml_charge: float,
                mlmm_atomic_charges: torch.Tensor,
                pbc: bool,
                ml_neighbor_list_calc: torch.nn.Module,
                mlmm_neighbor_list_calc: torch.nn.Module,
                mlmm_electrostatics_calc: torch.nn.Module,
                ml_cutoff_shell: float,
                mlmm_cutoff_shell: float,
                ml_fragment: int,
                mm_fragment: int,
                conversion_positions: float,
                conversion_energy: float,
                device: str,
                dtype: 'dtype',
            ):

                super(ModelForce, self).__init__()

                # Assign class variables
                self.ml_atom_indices = ml_atom_indices
                self.ml_atomic_numbers = ml_atomic_numbers
                self.mlmm_atomic_charges = mlmm_atomic_charges
                self.ml_cutoff_shell = ml_cutoff_shell
                self.mlmm_cutoff_shell = mlmm_cutoff_shell
                self.ml_fragment = ml_fragment
                self.mm_fragment = mm_fragment
                self.conversion_positions = conversion_positions
                self.conversion_energy = conversion_energy
                self.device = device
                self.dtype = dtype

                # Assign class modules
                self.model_calculator = model_calculator
                self.ml_neighbor_list_calc = ml_neighbor_list_calc
                self.mlmm_neighbor_list_calc = mlmm_neighbor_list_calc
                self.mlmm_electrostatics_calc = mlmm_electrostatics_calc

                # Get ML and ML/MM atom numbers
                ml_num_atoms = len(ml_atom_indices)
                self.ml_num_atoms = torch.tensor(
                    [ml_num_atoms],
                    device=self.device,
                    dtype=torch.int64)
                mlmm_num_atoms = len(mlmm_atomic_charges)
                self.mlmm_num_atoms = torch.tensor(
                    [mlmm_num_atoms],
                    device=self.device,
                    dtype=torch.int64)

                # Define ML and ML/MM system index
                self.ml_sys_i = torch.zeros(
                    ml_num_atoms,
                    device=self.device,
                    dtype=torch.int64)
                self.mlmm_sys_i = torch.zeros(
                    mlmm_num_atoms,
                    device=self.device,
                    dtype=torch.int64)

                # Define fragments and get ML atoms mask
                fragment_numbers = torch.full(
                    (mlmm_num_atoms, ),
                    self.mm_fragment,
                    device=self.device,
                    dtype=torch.int64)
                fragment_numbers[self.ml_atom_indices] = self.ml_fragment
                self.ml_mask = (fragment_numbers == self.ml_fragment)

                # If multiple fragments are defined, create index pointer from
                # full system to fragment system (e.g. atom with index 42 in
                # the full system has only index 2 in the fragment subsystem).
                ml_idxp = torch.full(
                    (mlmm_num_atoms, ),
                    -1,
                    device=self.device,
                    dtype=torch.int64)
                for ia, ai in enumerate(self.ml_atom_indices):
                    ml_idxp[ai] = ia
                
                # Create model calculator batch
                self.batch = {
                    'atomic_numbers': self.ml_atomic_numbers,
                    'pbc': torch.tensor(
                        [[pbc, pbc, pbc]],
                        device=self.device,
                        dtype=torch.bool),
                    'cell': torch.zeros(
                        (1, 3, 3,),
                        device=ml_charge.device,
                        dtype=self.dtype),
                    'charge': ml_charge,
                    'fragment_numbers': fragment_numbers,
                    'mlmm_atomic_charges': self.mlmm_atomic_charges,
                    }
                
                # Previous ML atom positions and ML/MM atom positions for
                # determining neighbor list update requirements
                self.ml_old_positions = torch.zeros(
                    (ml_num_atoms, 3),
                    device=self.device,
                    dtype=self.dtype)
                self.mlmm_old_positions = torch.zeros(
                    (mlmm_num_atoms, 3),
                    device=self.device,
                    dtype=self.dtype)
                self.torch_true = torch.tensor(
                    True, device=self.device, dtype=torch.bool)

                return

            def update_required(
                self,
                positions: torch.Tensor,
                old_positions: torch.Tensor,
                cutoff_shell: float,
            ) -> torch.Tensor:
                """
                Check if a neighbor list update is required if positions from
                the last update are changed more than half the cutoff shell
                
                Parameters
                ----------
                positions: torch.Tensor
                    The current positions of the atoms
                old_positions: torch.Tensor
                    The positions of the atoms from the last update
                cutoff_shell: float
                    Buffer region distance in nm

                Returns
                -------
                bool
                    Is the neighbor list update requirement meet

                """

                # Check neighbor list update requirement
                if old_positions is None:
                    return self.torch_true
                else:
                    return torch.any(
                        torch.sum((positions - old_positions)**2, dim=1)
                        > (0.5*cutoff_shell)**2)

            def forward(
                self,
                positions: torch.Tensor,
                boxvectors: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:#Tuple[torch.Tensor, torch.Tensor]:
                """
                Forward pass of the NequIP model.

                Parameters
                ----------
                positions: torch.Tensor
                    The positions of the atoms
                boxvector : torch.Tensor
                    The box vectors

                Returns
                -------
                torch.Tensor
                    The predicted energy in kJ/mol
                torch.Tensor
                    The predicted forces in kJ/mol/nm

                """

                # Create a forces tensor
                # forces = torch.zeros_like(positions).to(self.dtype)
                energy = torch.tensor(
                    0.0,
                    dtype=self.dtype,
                    device=self.device
                    )
                
                # Check input device
                if positions.device != self.device:
                    positions = positions.to(device=self.device)
                if boxvectors is not None and boxvectors.device != self.device:
                    boxvectors = boxvectors.to(
                        device=self.device)

                # Update system batch with ML/MM inputs
                self.batch['positions'] = positions/self.conversion_positions
                if boxvectors is not None:
                    self.batch['cell'] = (
                        boxvectors.unsqueeze(0)/self.conversion_positions)
                self.batch['atoms_number'] = self.mlmm_num_atoms
                self.batch['sys_i'] = self.mlmm_sys_i

                # Update ML neighbor list if required
                ml_positions = positions[self.ml_mask].reshape(-1, 3)
                if self.update_required(
                    ml_positions,
                    self.ml_old_positions,
                    self.ml_cutoff_shell,
                ):
                    self.ml_old_positions = ml_positions.clone()
                    self.batch = self.ml_neighbor_list_calc(self.batch)

                # Update ML/MM neighbor list if required
                if self.update_required(
                    positions,
                    self.mlmm_old_positions,
                    self.mlmm_cutoff_shell,
                ):
                    self.mlmm_old_positions = positions.clone()
                    self.batch = self.mlmm_neighbor_list_calc(self.batch)

                # Update system batch  with ML inputs
                self.batch['atoms_number'] = self.ml_num_atoms
                self.batch['sys_i'] = self.ml_sys_i

                # Predict ML energy
                self.batch = self.model_calculator(
                    self.batch, no_derivation=True)
                energy = self.batch['energy']

                # Compute ML-MM electrostatic Coulumb potential
                if self.mlmm_electrostatics_calc is not None:
                    
                    # Compute ML-MM electrostatic Coulomb potential
                    self.batch = self.mlmm_electrostatics_calc(self.batch)
                    energy = energy + self.batch['mlmm_energy']

                # Convert energy from model unit to OpenMM energy unit
                energy = energy*self.conversion_energy

                return energy

        # initialize model potential force instance
        modelForce = ModelForce(
            self.model_calculator,
            self.ml_atom_indices,
            self.ml_atomic_numbers,
            self.ml_charge,
            self.mlmm_atomic_charges,
            self.pbc,
            self.ml_neighbor_list_calc,
            self.mlmm_neighbor_list_calc,
            self.mlmm_electrostatics_calc,
            self.ml_cutoff_shell,
            self.mlmm_cutoff_shell,
            self.ml_fragment,
            self.mm_fragment,
            self.model2openmm_unit_conversion['positions'],
            self.model2openmm_unit_conversion['energy'],
            self.device,
            self.dtype,
            )

        # Convert model force instance to TorchScript
        module = torch.jit.script(modelForce)

        # Create the TorchForce and add it to the system
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(self.pbc)
        system.addForce(force)

        return


class MLMM_electrostatics(torch.nn.Module):
    """
    Torch implementation of a ML point charge and, eventually, atomic dipole 
    to MM point charge electrostatic model.

    Parameters
    ----------
    cutoff: torch.Tensor
        Interaction cutoff distance for ML/MM electrostatic interactions
    cuton: torch.Tensor
        Lower atom pair distance to start interaction switch-off for ML/MM
        electrostatic interactions
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type
    unit_properties: dict, optional, default None
        Dictionary with the units of the model properties to initialize correct
        conversion factors.
    lambda_value: torch.Tensor
        Scaling factor for the ML/MM interaction energy
    switch_fn: (str, callable), optional, default 'Poly6_range'
        Type of switch off function
    truncation: str, optional, default 'force'
        Truncation method of the Coulomb potential at the cutoff range:
            None, 'None': 
                No Coulomb potential shift applied
            'potential':
                Apply shifted Coulomb potential method
                    V_shifted(r) = V_Coulomb(r) - V_Coulomb(r_cutoff)
            'force', 'forces':
                Apply shifted Coulomb force method
                    V_shifted(r) = V_Coulomb(r) - V_Coulomb(r_cutoff)
                        - (dV_Coulomb/dr)|r_cutoff  * (r - r_cutoff)
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: torch.Tensor,
        cuton: torch.Tensor,
        device: str,
        dtype: 'dtype',
        unit_properties: Dict[str, str],
        lambda_value: torch.Tensor,
        switch_fn: Union[str, object] = 'Poly6_range',
        truncation: str = 'potential',
        atomic_dipoles: bool = False,
        **kwargs
    ):

        super(MLMM_electrostatics, self).__init__()

        # Assign variables
        self.cutoff = cutoff
        self.cuton = cuton
        self.lambda_value = lambda_value

        # Assign module variable parameters from configuration
        self.dtype = dtype
        self.device = device

        # Assign switch function
        switch_class = layer.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(
            self.cutoff, self.cuton, device=self.device, dtype=self.dtype)

        # Set property units for parameter scaling
        self.set_unit_properties(unit_properties)
       
        # Assign truncation method
        if truncation is None or truncation.lower() == 'none':
            self.potential_fn = MLMM_electrostatics_NoShift(
                self.cutoff,
                self.ke,
                self.switch_fn,
                atomic_dipoles)
        elif truncation.lower() == 'potential':
            self.potential_fn = MLMM_electrostatics_ShiftedPotential(
                self.cutoff,
                self.ke,
                self.switch_fn,
                atomic_dipoles)
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = MLMM_electrostatics_ShiftedForce(
                self.cutoff,
                self.ke,
                self.switch_fn,
                atomic_dipoles)
        else:
            raise SyntaxError(
                "Truncation method of the Coulomb potential "
                + f"'{truncation:}' is unknown!\n"
                + "Available are 'None', 'potential', 'force'.")

        return

    def __str__(self):
        return "Shielded Point Charge and Atomic Dipole Electrostatics"

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {}

    def set_unit_properties(
        self,
        unit_properties: Dict[str, str],
    ):
        """
        Set unit conversion factors for compatibility between requested
        property units and applied property units (for physical constants)
        of the module.
        
        Parameters
        ----------
        unit_properties: dict
            Dictionary with the units of the model properties to initialize 
            correct conversion factors.
        
        """

        # Get conversion factors
        if unit_properties is None:
            unit_energy = settings._default_units.get('energy')
            unit_positions = settings._default_units.get('positions')
            unit_charge = settings._default_units.get('charge')
            factor_energy, _ = utils.check_units(unit_energy)
            factor_positions, _ = utils.check_units(unit_positions)
            factor_charge, _ = utils.check_units(unit_charge)
        else:
            factor_energy, _ = utils.check_units(
                unit_properties.get('energy'))
            factor_positions, _ = utils.check_units(
                unit_properties.get('positions'))
            factor_charge, _ = utils.check_units(
                unit_properties.get('charge'))
    
        # Convert 1/(4*pi*epsilon) from e**2/eV/Ang to model units
        ke_ase = 14.399645351950548
        ke = torch.tensor(
            [ke_ase*factor_charge**2/factor_energy/factor_positions],
            device=self.device, dtype=self.dtype)
        self.register_buffer('ke', ke)

        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute shielded electrostatic interaction between atom center point 
        charges and dipoles.

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary added by module results
        
        """

        # Assign variables
        positions = batch['positions']
        ml_idx_i = batch['mlmm_idx_i']
        mm_idx_j = batch['mlmm_idx_j']

        # Compute ML-MM atom pair vectors and distances
        if 'mlmm_pbc_offset_ij' in batch:
            batch['mlmm_vectors'] = (
                positions[mm_idx_j] - positions[ml_idx_i]
                + batch['mlmm_pbc_offset_ij'])
        else:
            batch['mlmm_vectors'] = (
                positions[mm_idx_j] - positions[ml_idx_i])
        batch['mlmm_distances'] = torch.norm(batch['mlmm_vectors'], dim=-1)

        # Compute damped ML-MM Coulomb potential
        Eelec_pair = self.potential_fn(batch)

        # Sum up electrostatic atom pair contribution
        Eelec_sys = torch.sum(Eelec_pair)

        # Assign total electrostatic energy contributions
        batch['mlmm_energy'] = Eelec_sys

        return batch


class MLMM_electrostatics_NoShift(torch.nn.Module):
    """
    Torch implementation of a damped point charge and, eventually, atomic
    dipole electrostatic pair interaction without applying any cutoff method

    Parameters
    ----------
    cutoff: float
        Coulomb potential cutoff range between atom pairs
    ke: float
        Coulomb potential factor
    switch_fn: callable
        Switch off function to turn of electrostatics for pair distances
        between cuton and cutoff radius
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation

    """

    def __init__(
        self,
        cutoff: float,
        ke: float,
        switch_fn: Callable,
        atomic_dipoles: bool,
    ):

        super(MLMM_electrostatics_NoShift, self).__init__()

        # Assign parameters
        self.cutoff = cutoff
        self.ke = ke
        self.switch_fn = switch_fn
        self.atomic_dipoles = atomic_dipoles

        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Damped & shifted forces Coulomb interaction

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors

        Returns
        -------
        torch.Tensor
            Damped and force shifted electrostatic atom pair interaction

        """

        # Compute reciprocal distances and cutoff shifts
        distances = batch['mlmm_distances']
        chi = 1.0/distances

        # Gather atomic charge pairs
        atomic_charges_i = batch['atomic_charges'][batch['mlmm_idx_i']]
        atomic_charges_j = batch['mlmm_atomic_charges'][batch['mlmm_idx_j']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_i*atomic_charges_j*chi

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if self.atomic_dipoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2

            # Adjust atom pair vectors
            chi_vectors = batch['mlmm_vectors']/distances.unsqueeze(-1)

            # Gather atomic dipole pairs
            atomic_dipoles_i = batch['atomic_dipoles'][batch['mlmm_idx_i']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_ji = torch.sum(chi_vectors*atomic_dipoles_i, dim=1)

            # Compute damped charge-dipole electrostatics
            Eelec = Eelec + atomic_charges_j*dot_ji*chi2

        # Sum electrostatic contributions
        Eelec = self.ke*Eelec

        # Apply switch off function
        Eelec = Eelec*self.switch_fn(distances)

        return Eelec


class MLMM_electrostatics_ShiftedPotential(torch.nn.Module):
    """
    Torch implementation of a damped point charge and, eventually, atomic
    dipole electrostatic pair interaction applying the shifted potential cutoff
    method

    Parameters
    ----------
    cutoff: float
        Coulomb potential cutoff range between atom pairs
    ke: float
        Coulomb potential factor
    switch_fn: callable
        Switch off function to turn of electrostatics for pair distances
        between cuton and cutoff radius
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation

    """

    def __init__(
        self,
        cutoff: float,
        ke: float,
        switch_fn: Callable,
        atomic_dipoles: bool,
        **kwargs
    ):

        super(MLMM_electrostatics_ShiftedPotential, self).__init__()

        # Assign parameters
        self.cutoff = cutoff
        self.ke = ke
        self.switch_fn = switch_fn
        self.atomic_dipoles = atomic_dipoles

        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Damped & shifted forces Coulomb interaction

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors

        Returns
        -------
        torch.Tensor
            Damped and force shifted electrostatic atom pair interaction

        """

        # Compute damped reciprocal distances and cutoff shifts
        distances = batch['mlmm_distances']
        chi = 1.0/distances
        chi_shift = 1.0/self.cutoff

        # Gather atomic charge pairs
        atomic_charges_i = batch['atomic_charges'][batch['mlmm_idx_i']]
        atomic_charges_j = batch['mlmm_atomic_charges'][batch['mlmm_idx_j']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_i*atomic_charges_j*(chi - chi_shift)

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if self.atomic_dipoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi2_shift = chi_shift**2

            # Adjust atom pair vectors
            chi_vectors = batch['mlmm_vectors']/distances.unsqueeze(-1)

            # Gather atomic dipole pairs
            atomic_dipoles_i = batch['atomic_dipoles'][batch['mlmm_idx_i']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_ji = torch.sum(chi_vectors*atomic_dipoles_i, dim=1)

            # Compute damped charge-dipole electrostatics
            Eelec = Eelec + atomic_charges_j*dot_ji*(chi2 - chi2_shift)

        # Sum electrostatic contributions
        Eelec = self.ke*Eelec

        # Apply switch off function
        Eelec = Eelec*self.switch_fn(distances)

        return Eelec


class MLMM_electrostatics_ShiftedForce(torch.nn.Module):
    """
    Torch implementation of a damped point charge and, eventually, atomic
    dipole electrostatic pair interaction applying the shifted force cutoff
    method

    Parameters
    ----------
    cutoff: float
        Coulomb potential cutoff range between atom pairs
    kehalf: float
        Coulomb potential factor
    switch_fn: callable
        Switch off function to turn of electrostatics for pair distances
        between cuton and cutoff radius
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation

    """

    def __init__(
        self,
        cutoff: float,
        ke: float,
        switch_fn: Callable,
        atomic_dipoles: bool,
    ):

        super(MLMM_electrostatics_ShiftedForce, self).__init__()

        # Assign parameters
        self.cutoff = cutoff
        self.cutoff2 = cutoff**2
        self.cutoff3 = cutoff**3
        self.ke = ke
        self.switch_fn = switch_fn
        self.atomic_dipoles = atomic_dipoles

        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Damped & shifted forces Coulomb interaction

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors

        Returns
        -------
        torch.Tensor
            Damped and force shifted electrostatic atom pair interaction

        """

        # Compute damped reciprocal distances and cutoff shifts
        distances = batch['mlmm_distances']
        chi = 1.0/distances
        chi_shift = 2.0/self.cutoff - distances/self.cutoff2

        # Gather atomic charge pairs
        atomic_charges_i = batch['atomic_charges'][batch['mlmm_idx_i']]
        atomic_charges_j = batch['mlmm_atomic_charges'][batch['mlmm_idx_j']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_i*atomic_charges_j*(chi - chi_shift)

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if self.atomic_dipoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi2_shift = (
                3.0/self.cutoff2 - 2.0*distances/self.cutoff3)

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi2_shift = chi_shift**2

            # Adjust atom pair vectors
            chi_vectors = batch['mlmm_vectors']/distances.unsqueeze(-1)

            # Gather atomic dipole pairs
            atomic_dipoles_i = batch['atomic_dipoles'][batch['mlmm_idx_i']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_ji = torch.sum(chi_vectors*atomic_dipoles_i, dim=1)

            # Compute damped charge-dipole electrostatics
            Eelec = Eelec + atomic_charges_j*dot_ji*(chi2 - chi2_shift)

        # Sum electrostatic contributions
        Eelec = self.ke*Eelec

        # Apply switch off function
        Eelec = Eelec*self.switch_fn(distances)

        return Eelec

