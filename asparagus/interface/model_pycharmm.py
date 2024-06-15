import sys
import ctypes
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils
from .. import layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    dtype: dtype object, optional, default torch.float64
        Data type of the calculator
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
        dtype=torch.float64,
    ):

        # Assign dtype
        self.dtype = model_calculator.dtype #dtype

        ################################
        # # # Set PyCHARMM Options # # #
        ################################
        
        # Number of machine learning (ML) atoms
        self.ml_num_atoms = torch.tensor(
            [len(ml_atom_indices)], dtype=torch.int64)

        # ML atom indices
        self.ml_atom_indices = torch.tensor(ml_atom_indices, dtype=torch.int64)
        self.tt = np.array(ml_atom_indices, dtype=int)
        # ML atomic numbers
        self.ml_atomic_numbers = torch.tensor(
            ml_atomic_numbers, dtype=torch.int64)
        
        # ML atom total charge
        self.ml_charge = torch.tensor(ml_charge, dtype=self.dtype)

        # ML fluctuating charges
        self.ml_fluctuating_charges = ml_fluctuating_charges

        # ML and MM atom charges
        self.mlmm_atomic_charges = torch.tensor(
            mlmm_atomic_charges, dtype=self.dtype)

        # ML and MM number of atoms
        self.mlmm_num_atoms = len(mlmm_atomic_charges)
        
        # ML and MM atom indices
        self.mlmm_atom_indices = torch.zeros(
            self.mlmm_num_atoms, dtype=torch.int64)
        self.mlmm_atom_indices[self.ml_atom_indices] = self.ml_atomic_numbers
        
        # ML atoms - atom indices pointing from MLMM position to ML position
        # 0, 1, 2 ..., ml_num_atoms: ML atom 1, 2, 3 ... ml_num_atoms + 1
        # ml_num_atoms + 1: MM atoms
        ml_idxp = np.full(self.mlmm_num_atoms, -1)
        for ia, ai in enumerate(ml_atom_indices):
            ml_idxp[ai] = ia
        self.ml_idxp = torch.tensor(ml_idxp, dtype=torch.int64)
        
        # Running number list
        self.mlmm_idxa = torch.arange(self.mlmm_num_atoms, dtype=torch.int64)

        # Non-bonding interaction range
        self.mlmm_cutoff = torch.tensor(mlmm_cutoff, dtype=self.dtype)
        self.mlmm_cuton = torch.tensor(mlmm_cuton, dtype=self.dtype)

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

        #############################
        # # # Set ML/MM Options # # #
        #############################

        # Get model interaction cutoff and set evaluation mode
        if self.model_ensemble:
            #self.interaction_cutoff = 0.0
            for calc in self.model_calculator_list:
                #cutoff = calc.model_interaction_cutoff
                #if self.interaction_cutoff < cutoff:
                    #self.interaction_cutoff = cutoff
                calc.eval()
        else:
            #self.interaction_cutoff = (
                #self.model_calculator.model_interaction_cutoff)
            self.model_calculator.eval()

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

            # Convert 1/(2*4*pi*epsilon) from e**2/eV/Ang to CHARMM units
            kehalf_ase = 7.199822675975274
            conversion, _ = utils.check_units(
                self.model_unit_properties.get('energy'))
            self.kehalf = torch.tensor(
                [kehalf_ase*1.**2/conversion/1.],
                dtype=self.dtype)

            self.electrostatics_calc = Electrostatic_shift(
                self.mlmm_cutoff,
                self.mlmm_cuton,
                #self.max_rcut,
                self.ml_idxp,
                self.mlmm_atomic_charges,
                kehalf=self.kehalf,
                )

        else:

            self.electrostatics_calc = None

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
            List of ML atom indices for ML potential
        idxj: list(int)
            List of ML atom indices for ML potential
        idxjp: list(int)
            List of image to primary ML atom index pointer
        idxu: list(int)
            List of ML atom indices for ML-MM embedding potential
        idxv: list(int)
            List of MM atom indices for ML-MM embedding potential
        idxup: list(int)
            List of image to primary ML atom index pointer
        idxvp: list(int)
            List of image to primary MM atom index pointer

        Return
        ------
        float
            ML potential plus ML-MM embedding potential
        """

        # Assign all positions
        if Ntrans:
            mlmm_R = torch.transpose(
                torch.tensor(
                    [x[:Natim], y[:Natim], z[:Natim]], dtype=self.dtype
                ),
                0, 1)
            mlmm_idxp = idxp[:Natim]
        else:
            mlmm_R = torch.transpose(
                torch.tensor(
                    [x[:Natom], y[:Natom], z[:Natom]], dtype=self.dtype
                ),
                0, 1)
            mlmm_idxp = idxp[:Natom]
        mlmm_R.requires_grad_(True)

        # Assign indices
        # ML-ML pair indices
        ml_idxi = torch.tensor(idxi[:Nmlp], dtype=torch.int64)
        ml_idxj = torch.tensor(idxj[:Nmlp], dtype=torch.int64)
        ml_idxjp = torch.tensor(idxjp[:Nmlp], dtype=torch.int64)
        ml_sysi = torch.zeros(self.ml_num_atoms, dtype=torch.int64)
        # ML-MM pair indices and pointer
        mlmm_idxu = torch.tensor(
            idxu[:Nmlmmp], dtype=torch.int64)
        mlmm_idxv = torch.tensor(
            idxv[:Nmlmmp], dtype=torch.int64)
        mlmm_idxup = torch.tensor(
            idxup[:Nmlmmp], dtype=torch.int64)
        mlmm_idxvp = torch.tensor(
            idxvp[:Nmlmmp], dtype=torch.int64)

        # Create batch for evaluating the model
        atoms_batch = {}
        atoms_batch['atoms_number'] = self.ml_num_atoms
        atoms_batch['atomic_numbers'] = self.ml_atomic_numbers
        atoms_batch['positions'] = mlmm_R
        atoms_batch['charge'] = self.ml_charge
        atoms_batch['idx_i'] = ml_idxi
        atoms_batch['idx_j'] = ml_idxj
        atoms_batch['sys_i'] = ml_sysi
        
        # PBC options
        atoms_batch['pbc_offset_ij'] = None
        atoms_batch['pbc_offset_uv'] = None
        atoms_batch['pbc_atoms'] = self.ml_atom_indices
        atoms_batch['pbc_idx'] = self.ml_idxp
        atoms_batch['pbc_idx_j'] = ml_idxjp

        # Compute model properties
        results = {}
        if self.model_ensemble:

            # TODO Test
            for ic, calc in enumerate(self.model_calculator_list):
                results[ic] = calc(atoms_batch)
            for prop in self.implemented_properties:
                prop_std = f"std_{prop:s}"
                results[prop_std], self.results[prop] = torch.std_mean(
                    torch.cat(
                        [
                            results[ic][prop]
                            for ic in range(self.model_calculator_num)
                        ],
                        dim=0),
                    dim=0)

        else:

            results = self.model_calculator(atoms_batch)

        # Unit conversion
        self.results = {}
        for prop in self.implemented_properties:
            self.results[prop] = (
                results[prop]*self.model2charmm_unit_conversion[prop])

        # Apply dtype conversion
        E = self.results['energy'].detach().numpy()
        ml_F = self.results['forces'].detach().numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))

        # Add forces to CHARMM derivative arrays
        for ai in self.ml_atom_indices:
            ii = 3*ai
            dx[ai] -= ml_F[ii]
            dy[ai] -= ml_F[ii+1]
            dz[ai] -= ml_F[ii+2]
        # Calculate electrostatic energy and force contribution
        if self.electrostatics_calc is not None:
            
            mlmm_Eele, mlmm_gradient = self.electrostatics_calc.run(
                mlmm_R,
                self.results['atomic_charges'],
                mlmm_idxu,
                mlmm_idxv,
                mlmm_idxup,
                mlmm_idxvp)
            
            # Add electrostatic interaction potential to ML energy
            E += (
                mlmm_Eele*self.model2charmm_unit_conversion['energy']
                ).detach().numpy()

            # Apply dtype conversion
            mlmm_F = (
                -mlmm_gradient*self.model2charmm_unit_conversion['forces']
                ).detach().numpy().ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                    )

            # Add electrostatic forces to CHARMM derivative arrays
            for ia, ai in enumerate(mlmm_idxp):
                ii = 3*ia
                dx[ai] -= mlmm_F[ii]
                dy[ai] -= mlmm_F[ii+1]
                dz[ai] -= mlmm_F[ii+2]

        return E


class Electrostatic_shift:
    """
    Coulomb potential calculator between fluctuating ML atomic charges and
    static MM atomic charges.

    Parameters
    ----------
    mlmm_cutoff: torch.Tensor
        Interaction cutoff distance for ML/MM electrostatic interactions
    mlmm_cuton: torch.Tensor
        Lower atom pair distance to start interaction switch-off for ML/MM
        electrostatic interactions
    ml_idxp: torch.Tensor
        ML atoms ti atom indices pointing from MLMM position to ML position.
    mlmm_atomic_charges: torch.Tensor
        List of all atomic charges of the system loaded to CHARMM.
    kehalf: torch.Tensor
        Coulomb factor with respective unit
    switch_fn: str, optional, default 'Poly6_range'
        Type of switch off function
    
    """

    def __init__(
        self,
        mlmm_cutoff: torch.Tensor,
        mlmm_cuton: torch.Tensor,
        ml_idxp: torch.Tensor,
        mlmm_atomic_charges: torch.Tensor,
        kehalf: torch.Tensor,
        switch_fn='Poly6_range',
    ):

        self.mlmm_cutoff = mlmm_cutoff
        self.mlmm_cutoff2 = mlmm_cutoff**2
        self.mlmm_cuton = mlmm_cuton
        self.ml_idxp = ml_idxp
        self.mlmm_atomic_charges = mlmm_atomic_charges

        # Initialize the class for cutoff
        self.switch_fn = layer.get_cutoff_fn(switch_fn)(
            self.mlmm_cutoff, self.mlmm_cuton)
        self.kehalf = kehalf

    def calculate_mlmm_interatomic_distances(
        self,
        R: torch.Tensor,
        idxu: torch.Tensor,
        idxv: torch.Tensor,
        idxup: torch.Tensor,
        idxvp: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate ML-MM atom distances.

        Parameters
        ----------
        R: torch.tensor
            ML and MM atom positions
        idxu: torch.tensor
            List of ML atom indices for ML-MM embedding potential
        idxv: torch.tensor
            List of MM atom indices for ML-MM embedding potential
        idxup: torch.tensor
            List of image to primary ML atom index pointer
        idxvp: torch.tensor
            List of image to primary MM atom index pointer

        Return
        ------
        torch.tensor
            ML-MM atom pair distances
        torch.tensor
            ML atom pair indices
        torch.tensor
            MM atom pair indices

        """

        # Gather positions
        Ru = R[idxu]
        Rv = R[idxv]
        
        # Interacting atom pair distances within cutoff
        sum_distances = torch.sum((Ru - Rv)**2, dim=1)
        selection = sum_distances < self.mlmm_cutoff2
        Duv = torch.sqrt(sum_distances[selection])
        
        # Reduce the indexes to consider only interacting pairs (selection)
        # and point image atom indices to primary atom indices (idx?p)
        idxur = idxup[selection]
        idxvr = idxvp[selection]
        
        return Duv, idxur, idxvr

    def electrostatic_energy_per_atom_to_point_charge(
        self,
        Duv: torch.Tensor,
        Qau: torch.Tensor,
        Qav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate electrostatic interaction between ML atom charge and MM point
        charge based on shifted Coulomb potential scheme
        
        Parameters
        ----------
        Duv: torch.tensor
            ML-MM atom pair distances
        Qau: torch.tensor
            List of ML atomic charges
        Qav: torch.tensor
            List of MM atomic charges

        Return
        ------
        torch.tensor
            ML-MM atom pair electrostatic Coulomb interaction
        
        """

        # Cutoff weighted reciprocal distance
        switchoff = self.switch_fn(Duv)

        # Shifted Coulomb energy
        qq = 2.0*self.kehalf*Qau*Qav
        Eele = qq/Duv - qq/self.mlmm_cutoff*(2.0 - Duv/self.mlmm_cutoff)

        return torch.sum(switchoff*Eele)

    def run(
        self,
        mlmm_R: torch.Tensor,
        ml_Qa: torch.Tensor,
        mlmm_idxu: torch.Tensor,
        mlmm_idxv: torch.Tensor,
        mlmm_idxup: torch.Tensor,
        mlmm_idxvp: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Calculates the electrostatic interaction between ML atoms in the 
        primary cell with all MM atoms in the primary or imaginary non-bonded
        lists.
        
        Parameters
        ----------
        mlmm_R: torch.tensor
            ML and MM atom positions
        ml_Qa: torch.tensor
            ML atomic charges
        mlmm_idxu: torch.tensor
            List of ML atom indices for ML-MM embedding potential
        mlmm_idxv: torch.tensor
            List of MM atom indices for ML-MM embedding potential
        mlmm_idxvp: torch.tensor
            List of image to primary ML atom index pointer
        mlmm_idxvp: torch.tensor
            List of image to primary MM atom index pointer

        Return
        ------
        torch.tensor
            Total ML-MM electrostatic Coulomb interaction
        torch.tensor
            ML and MM atom electrostatic Coulomb forces
        
        """
        
        # Calculate ML-MM atom distances
        mlmm_Duv, mlmm_idxur, mlmm_idxvr = (
            self.calculate_mlmm_interatomic_distances(
                mlmm_R, mlmm_idxu, mlmm_idxv, mlmm_idxup, mlmm_idxvp)
            )

        # Point from PyCHARMM ML atom indices to model calculator atom indices
        ml_idxur = self.ml_idxp[mlmm_idxur]
        
        # Get ML and MM charges
        ml_Qau = ml_Qa[ml_idxur]
        ml_Qav = self.mlmm_atomic_charges[mlmm_idxvr]

        # Calculate electrostatic energy and gradients
        Eele = self.electrostatic_energy_per_atom_to_point_charge(
            mlmm_Duv, ml_Qau, ml_Qav)
        Eele_gradient = torch.autograd.grad(
                torch.sum(Eele),
                mlmm_R,
                retain_graph=True)[0]

        return Eele, Eele_gradient
