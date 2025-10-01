import sys
import ctypes
import numpy as np
from typing import Optional, List, Dict, Callable, Any, Tuple, Union

import torch

from asparagus import utils
from asparagus import layer

__all__ = ['PyCharmm_Calculator']

CHARMM_calculator_units = {
    'positions':        'Ang',
    'energy':           'kcal/mol',
    'atomic_energies':  'kcal/mol',
    'forces':           'kcal/mol/Ang',
    'hessian':          'kcal/mol/Ang/Ang',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'e*Ang',
    'atomic_dipoles':   'e*Ang',
    }


class PyCharmm_Calculator:
    """
    Calculator for the interface between PyCHARMM and Asparagus.

    Parameters
    ----------
    model_calculator: torch.nn.Module
        Asparagus model calculator object with already loaded parameter set
    ml_atom_indices: list(int)
        List of atom indices referring to the ML treated atoms in the total 
        system loaded in CHARMM
    ml_atomic_numbers: list(int)
        Respective atomic numbers of the ML atom selection
    ml_charge: float
        Total charge of the partial ML atom selection
    ml_fluctuating_charges: bool
        If True, electrostatic interaction contribution between the MM atom
        charges and the model predicted ML atom charges. Else, the ML atom
        charges are considered fixed as defined by the CHARMM psf file.
    mlmm_atomic_charges: list(float)
        List of all atomic charges of the system loaded to CHARMM.
        If 'ml_fluctuating_charges' is True, the atomic charges of the ML
        atoms are ignored (usually set to zero anyways) and their atomic
        charge prediction is used.
    mlmm_cutoff: float
        Interaction cutoff distance for ML/MM electrostatic interactions
    mlmm_cuton: float
        Lower atom pair distance to start interaction switch-off for ML/MM
        electrostatic interactions
    mlmm_lambda: float, optional, default None
        ML/MM electrostatic interactions scaling factor. If None, no scaling
        is applied.
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        model_calculator: Union[torch.nn.Module, List[torch.nn.Module]],
        ml_atom_indices: Optional[List[int]] = None,
        ml_atomic_numbers: Optional[List[int]] = None,
        ml_charge: Optional[float] = None,
        ml_fluctuating_charges: Optional[bool] = None,
        mlmm_atomic_charges: Optional[List[float]] = None,
        mlmm_cutoff: Optional[float] = None,
        mlmm_cuton: Optional[float] = None,
        mlmm_lambda: Optional[float] = None,
        **kwargs
    ):

        # Assign model device and dtype
        self.device = model_calculator.device
        self.dtype = model_calculator.dtype

        ################################
        # # # Set PyCHARMM Options # # #
        ################################
        
        # Number of machine learning (ML) atoms
        self.ml_num_atoms = torch.tensor(
            [len(ml_atom_indices)], device=self.device, dtype=torch.int64)
        self.ml_sysi = torch.zeros(
            self.ml_num_atoms, device=self.device, dtype=torch.int64)

        # ML atom indices
        self.ml_atom_indices = torch.tensor(
            ml_atom_indices, device=self.device, dtype=torch.int64)

        # ML atomic numbers
        self.ml_atomic_numbers = torch.tensor(
            ml_atomic_numbers, device=self.device, dtype=torch.int64)

        # ML atom total charge
        self.ml_charge = torch.tensor(
            ml_charge, device=self.device, dtype=self.dtype)

        # ML fluctuating charges
        self.ml_fluctuating_charges = ml_fluctuating_charges

        # Assign ML-ML model cutoff
        self.ml_cutoff = model_calculator.model_cutoff
        self.ml_cutoff2 = self.ml_cutoff**2

        # ML and MM atom charges
        self.mlmm_atomic_charges = torch.tensor(
            mlmm_atomic_charges, device=self.device, dtype=self.dtype)

        # ML and MM number of atoms
        self.mlmm_num_atoms = len(mlmm_atomic_charges)

        # ML atoms - atom indices pointing from MLMM position to ML position
        # 0, 1, 2 ..., ml_num_atoms: ML atom 1, 2, 3 ... ml_num_atoms + 1
        # ml_num_atoms + 1: MM atoms
        ml_idxp = np.full(self.mlmm_num_atoms, -1)
        for ia, ai in enumerate(ml_atom_indices):
            ml_idxp[ai] = ia
        self.ml_idxp = torch.tensor(
            ml_idxp, device=self.device, dtype=torch.int64)
        
        # Running number list
        self.mlmm_idxa = torch.arange(
            self.mlmm_num_atoms, device=self.device, dtype=torch.int64)

        # Non-bonding interaction range
        self.mlmm_cutoff = torch.tensor(
            mlmm_cutoff, device=self.device, dtype=self.dtype)
        self.mlmm_cuton = torch.tensor(
            mlmm_cuton, device=self.device, dtype=self.dtype)
        
        # Non-bonding electrostatic scaling factor
        if mlmm_lambda is None:
            self.mlmm_lambda = torch.tensor(
                1.0, device=self.device, dtype=self.dtype)
        else:
            self.mlmm_lambda = torch.tensor(
                mlmm_lambda, device=self.device, dtype=self.dtype)

        ################################
        # # # Set Model Calculator # # #
        ################################

        # In case of model calculator is a list of models
        if utils.is_array_like(model_calculator):
            self.model_calculator = None
            self.model_calculator_list = model_calculator
            self.model_calculator_num = len(model_calculator)
            self.model_ensemble = True
        else:
            self.model_calculator = model_calculator
            self.model_calculator_list = None
            self.model_calculator_num = 1
            self.model_ensemble = False

        # Get implemented model properties
        if self.model_ensemble:
            self.implemented_properties = (
                self.model_calculator_list[0].model_properties)
            # Check model properties and set evaluation mode
            for ic, calc in enumerate(self.model_calculator_list):
                for prop in self.implemented_properties:
                    if prop not in calc.model_properties:
                        raise SyntaxError(
                            f"Model calculator {ic:d} does not predict "
                            + f"property {prop:s}!\n"
                            + "Specify 'implemented_properties' with "
                            + "properties all model calculator support.")
        else:
            self.implemented_properties = (
                self.model_calculator.model_properties)

        # Check if model calculator has loaded a checkpoint file or stored
        # if model parameters are stored in a checkpoint file
        if self.model_calculator is None:
            for ic, calc in enumerate(self.model_calculator_list):
                if not calc.checkpoint_loaded:
                    raise SyntaxError(
                        f"Model calculator {ic:d} does not seem to have a "
                        + "proper parameter set loaded from a checkpoint file."
                        + "\nMake sure parameters are loaded otherwise "
                        + "model predictions are random.")
        else:
            if not self.model_calculator.checkpoint_loaded:
                raise SyntaxError(
                    "The model calculator does not seem to have a "
                    + "proper parameter set loaded from a checkpoint file."
                    + "\nMake sure parameters are loaded otherwise "
                    + "model predictions are random.")

        # Set model to evaluation mode
        if self.model_ensemble:
            for calc in self.model_calculator_list:
                calc.eval()
        else:
            self.model_calculator.eval()

        #############################
        # # # Set ML/MM Options # # #
        #############################

        # Get property unit conversions from model units to CHARMM units
        self.model_unit_properties = (
            self.model_calculator.model_unit_properties)
        self.model2charmm_unit_conversion = {}

        # Positions unit conversion
        conversion, _ = utils.check_units(
            CHARMM_calculator_units['positions'],
            self.model_unit_properties.get('positions'))
        self.model2charmm_unit_conversion['positions'] = conversion

        # Implemented property units conversion
        for prop in self.implemented_properties:
            conversion, _ = utils.check_units(
                CHARMM_calculator_units[prop],
                self.model_unit_properties.get(prop))
            self.model2charmm_unit_conversion[prop] = conversion

        # Initialize the non-bonded interaction calculator
        if self.ml_fluctuating_charges:

            self.electrostatics_calc = MLMM_electrostatics(
                self.mlmm_cutoff,
                self.mlmm_cuton,
                self.device,
                self.dtype,
                unit_properties=self.model_unit_properties,
                lambda_value=self.mlmm_lambda,
                atomic_dipoles=self.model_calculator.model_atomic_dipoles,
                )

        else:

            self.electrostatics_calc = None

        return

    def calculate_charmm(
        self,
        Natom: int,
        Ntrans: int,
        Natim: int,
        idxp: List[float],
        x: List[float],
        y: List[float],
        z: List[float],
        dx: List[float],
        dy: List[float],
        dz: List[float],
        Nmlp: int,
        Nmlmmp: int,
        idxi: List[int],
        idxj: List[int],
        idxjp: List[int],
        idxu: List[int],
        idxv: List[int],
        idxup: List[int],
        idxvp: List[int],
    ) -> float:
        """
        This function matches the signature of the corresponding MLPot class in
        PyCHARMM.

        Parameters
        ----------
        Natom: int
            Number of atoms in primary cell
        Ntrans: int
            Number of unit cells (primary + images)
        Natim: int
            Number of atoms in primary and image unit cells
        idxp: list(int)
            List of primary and primary to image atom index pointer
        x: list(float)
            List of x coordinates 
        y: list(float)
            List of y coordinates
        z: list(float)
            List of z coordinates
        dx: list(float)
            List of x derivatives
        dy: list(float)
            List of y derivatives
        dz: list(float)
            List of z derivatives
        Nmlp: int
            Number of ML atom pairs in the system
        Nmlmmp: int
            Number of ML/MM atom pairs in the system
        idxi: list(int)
            List of ML atom pair indices for ML potential
        idxj: list(int)
            List of ML atom pair indices for ML potential
        idxjp: list(int)
            List of image to primary ML atom pair index pointer
        idxu: list(int)
            List of ML atom pair indices for ML-MM embedding potential
        idxv: list(int)
            List of MM atom pair indices for ML-MM embedding potential
        idxup: list(int)
            List of image to primary ML atom pair index pointer
        idxvp: list(int)
            List of image to primary MM atom pair index pointer

        Return
        ------
        float
            ML potential plus ML-MM embedding potential

        """

        # Assign number of atoms
        if Ntrans:
            Nmlmm = Natim
        else:
            Nmlmm = Natom

        # Assign primary and primary to image atom index pointer
        mlmm_idxp = torch.tensor(idxp[:Nmlmm], dtype=torch.int64)
        
        # Assign all positions
        mlmm_R = torch.transpose(
            torch.tensor(
                [x[:Nmlmm], y[:Nmlmm], z[:Nmlmm]],
                device=self.device,
                dtype=self.dtype
            ),
            0, 1)

        # Assign ML-ML pair indices
        ml_idxi = torch.tensor(
            idxi[:Nmlp], device=self.device, dtype=torch.int64)
        ml_idxj = torch.tensor(
            idxj[:Nmlp], device=self.device, dtype=torch.int64)
        ml_idxjp = torch.tensor(
            idxjp[:Nmlp], device=self.device, dtype=torch.int64)

        # Update ML-ML pair indices by cutoff distance
        selection = (
            torch.sum((mlmm_R[ml_idxi] - mlmm_R[ml_idxj])**2, dim=1)
            <= self.ml_cutoff2)
        ml_idxi = ml_idxi[selection]
        ml_idxj = ml_idxj[selection]
        ml_idxjp = ml_idxjp[selection]

        # Set gradient properties
        mlmm_R.requires_grad_(True)

        # Create batch for evaluating the model
        atoms_batch = {}
        atoms_batch['atoms_number'] = self.ml_num_atoms
        atoms_batch['atomic_numbers'] = self.ml_atomic_numbers
        atoms_batch['positions'] = mlmm_R
        atoms_batch['charge'] = self.ml_charge
        atoms_batch['idx_i'] = ml_idxi
        atoms_batch['idx_j'] = ml_idxj
        atoms_batch['sys_i'] = self.ml_sysi
        
        # Add ML/MM and PBC supercluster variables
        atoms_batch['ml_idx'] = self.ml_atom_indices
        atoms_batch['ml_idx_p'] = self.ml_idxp
        atoms_batch['ml_idx_jp'] = ml_idxjp
        
        # Compute model properties
        atoms_batch = self.model_calculator(
            atoms_batch,
            create_graph=self.ml_fluctuating_charges)

        # Unit conversion
        self.results = {}
        for prop in self.implemented_properties:
            self.results[prop] = (
                atoms_batch[prop]*self.model2charmm_unit_conversion[prop])

        # Apply dtype conversion of force components
        ml_Epot = self.results['energy'].cpu().detach().numpy()
        ml_F = self.results['forces'].cpu().detach().numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))

        # Add forces to CHARMM derivative arrays
        for ai in self.ml_atom_indices:
            ii = 3*ai
            dx[ai] -= ml_F[ii]
            dy[ai] -= ml_F[ii+1]
            dz[ai] -= ml_F[ii+2]

        # Calculate electrostatic energy and force contribution
        if self.electrostatics_calc is None:

            self.mlmm_Eele = 0.0

        else:
            
            # Assign ML-MM pair indices and pointer
            mlmm_idxu = torch.tensor(
                idxu[:Nmlmmp], device=self.device, dtype=torch.int64)
            mlmm_idxv = torch.tensor(
                idxv[:Nmlmmp], device=self.device, dtype=torch.int64)
            mlmm_idxup = torch.tensor(
                idxup[:Nmlmmp], device=self.device, dtype=torch.int64)
            mlmm_idxvp = torch.tensor(
                idxvp[:Nmlmmp], device=self.device, dtype=torch.int64)
            
            # Update batch for evaluating the model
            atoms_batch['mlmm_atomic_charges'] = self.mlmm_atomic_charges
            atoms_batch['mlmm_idxu'] = mlmm_idxu
            atoms_batch['mlmm_idxv'] = mlmm_idxv
            atoms_batch['mlmm_idxup'] = mlmm_idxup
            atoms_batch['mlmm_idxvp'] = mlmm_idxvp

            # Compute MLMM electrostatics
            mlmm_Eelec = self.electrostatics_calc(atoms_batch)['mlmm_energy']

            # Compute MLMM electrostatics forces
            mlmm_gradient_elec = torch.autograd.grad(
                torch.sum(mlmm_Eelec),
                mlmm_R,
                retain_graph=False)[0]

            # Add electrostatic interaction potential to ML energy
            self.mlmm_Eelec = (
                mlmm_Eelec*self.mlmm_lambda
                * self.model2charmm_unit_conversion['energy']
                ).cpu().detach().numpy()
            
            # Only add electrostatic when using older CHARMM versions where
            # MLMM electrostatic is not added to the ELEC energy contribution
            # ml_Epot += self.mlmm_Eelec

            # Apply dtype conversion of force components
            mlmm_F = (
                -mlmm_gradient_elec*self.mlmm_lambda
                * self.model2charmm_unit_conversion['forces']
                ).cpu().detach().numpy().ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                    )

            # Add electrostatic forces to CHARMM derivative arrays
            for ia, ai in enumerate(mlmm_idxp):
                ii = 3*ia
                dx[ai] -= mlmm_F[ii]
                dy[ai] -= mlmm_F[ii+1]
                dz[ai] -= mlmm_F[ii+2]

        return ml_Epot

    def mlmm_elec(self):
        return self.mlmm_Eelec


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
        self.cutoff2 = cutoff**2
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
        ml_idxu = batch['mlmm_idxu']
        mm_idxv = batch['mlmm_idxv']
        ml_idxup = batch['mlmm_idxup']
        mm_idxvp = batch['mlmm_idxvp']
        ml_idxtp = batch['mlmm_idxvp']

        # Compute ML-MM atom pair vectors and distances within cutoff
        mlmm_vectors = positions[ml_idxu] - positions[mm_idxv]
        mlmm_distances2 = torch.sum(mlmm_vectors**2, dim=1)
        in_cutoff = mlmm_distances2 < self.cutoff2
        batch['mlmm_vectors'] = mlmm_vectors[in_cutoff]
        batch['mlmm_distances'] = torch.sqrt(mlmm_distances2[in_cutoff])

        # Select indexes to consider only interacting pairs (selection)
        # and point image atom indices from periodic cells to the respective 
        # primary atom indices (idx?p[idx?])
        ml_idxu = ml_idxup[in_cutoff]
        batch['mlmm_idxv'] = mm_idxvp[in_cutoff]

        # Point from CHARMM topology ML atom indices (somewhere between 0 and
        # N_MLMM) to model calculator atom indices (between 0 and N_ML)
        batch['mlmm_idxu'] = batch['ml_idx_p'][ml_idxu]

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
        atomic_charges_i = batch['mlmm_atomic_charges'][batch['mlmm_idxu']]
        atomic_charges_j = batch['mlmm_atomic_charges'][batch['mlmm_idxv']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_i*atomic_charges_j*chi

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if self.atomic_dipoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2

            # Adjust atom pair vectors
            chi_vectors = batch['mlmm_vectors']/distances.unsqueeze(-1)

            # Gather atomic dipole pairs
            atomic_dipoles_i = batch['atomic_dipoles'][batch['mlmm_idxu']]

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
        atomic_charges_i = batch['atomic_charges'][batch['mlmm_idxu']]
        atomic_charges_j = batch['mlmm_atomic_charges'][batch['mlmm_idxv']]

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
            atomic_dipoles_i = batch['atomic_dipoles'][batch['mlmm_idxu']]

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
        atomic_charges_i = batch['mlmm_atomic_charges'][batch['mlmm_idxu']]
        atomic_charges_j = batch['mlmm_atomic_charges'][batch['mlmm_idxv']]

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
            atomic_dipoles_i = batch['atomic_dipoles'][batch['mlmm_idxu']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_ji = torch.sum(chi_vectors*atomic_dipoles_i, dim=1)

            # Compute damped charge-dipole electrostatics
            Eelec = Eelec + atomic_charges_j*dot_ji*(chi2 - chi2_shift)

        # Sum electrostatic contributions
        Eelec = self.ke*Eelec

        # Apply switch off function
        Eelec = Eelec*self.switch_fn(distances)

        return Eelec
