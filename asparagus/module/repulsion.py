import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from asparagus import utils
from asparagus import settings

__all__ = ["ZBL_repulsion"]

# ======================================
#  Nuclear Repulsion
# ======================================


class ZBL_repulsion(torch.nn.Module):
    """
    Torch implementation of a Ziegler-Biersack-Littmark style nuclear 
    repulsion model.

    Parameters
    ----------
    trainable: bool
        If True, repulsion parameter are trainable. Else, default parameter
        values are fixed.
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type
    unit_properties: dict, optional, default {}
        Dictionary with the units of the model properties to initialize correct
        conversion factors.
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        trainable: bool,
        device: str,
        dtype: 'dtype',
        unit_properties: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize Ziegler-Biersack-Littmark style nuclear repulsion model.
        
        """
        
        super(ZBL_repulsion, self).__init__()
        
        # Assign variables
        self.dtype = dtype
        self.device = device
        
        # Initialize repulsion model parameters
        a_coefficient = 0.8854 # Angstrom
        a_exponent = 0.23
        phi_coefficients = [0.18175, 0.50986, 0.28022, 0.02817]
        phi_exponents = [3.19980, 0.94229, 0.40290, 0.20162] # 1/Angstrom
        
        if trainable:
            self.a_coefficient = torch.nn.Parameter(
                torch.tensor([a_coefficient], device=device, dtype=dtype))
            self.a_exponent = torch.nn.Parameter(
                torch.tensor([a_exponent], device=device, dtype=dtype))
            self.phi_coefficients = torch.nn.Parameter(
                torch.tensor(phi_coefficients, device=device, dtype=dtype))
            self.phi_exponents = torch.nn.Parameter(
                torch.tensor(phi_exponents, device=device, dtype=dtype))
        else:
            self.register_buffer(
                "a_coefficient",
                torch.tensor([a_coefficient], dtype=dtype))
            self.register_buffer(
                "a_exponent",
                torch.tensor([a_exponent], dtype=dtype))
            self.register_buffer(
                "phi_coefficients",
                torch.tensor(phi_coefficients, dtype=dtype))
            self.register_buffer(
                "phi_exponents",
                torch.tensor(phi_exponents, dtype=dtype))

        # Unit conversion factors
        self.set_unit_properties(unit_properties)

        return

    def __str__(self):
        return "Ziegler-Biersack-Littmark style nuclear repulsion model"

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
            factor_energy, _ = utils.check_units(unit_energy, 'Hartree')
            factor_positions, _ = utils.check_units('Ang', unit_positions)
        else:
            factor_energy, _ = utils.check_units(
                unit_properties.get('energy'), 'Hartree')
            factor_positions, _ = utils.check_units(
                'Ang', unit_properties.get('positions'))

        # Convert
        # Distances: model to Bohr
        # Energies: Hartree to model
        self.register_buffer(
            "distances_model2Ang", 
            torch.tensor([factor_positions], dtype=self.dtype))
        self.register_buffer(
            "energies_Hatree2model", 
            torch.tensor([factor_energy], dtype=self.dtype))

        # Convert e**2/(4*pi*epsilon) from 1/eV/Ang to model units
        ke_ase = 1./(4.*np.pi*5.526349406e-3)
        ke = torch.tensor(
            [ke_ase/factor_energy/factor_positions],
            device=self.device, dtype=self.dtype)
        self.register_buffer('ke', ke)

        return

    def forward(
        self,
        atomic_numbers: torch.Tensor, 
        distances: torch.Tensor, 
        cutoffs: torch.Tensor, 
        idx_i: torch.Tensor, 
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Ziegler-Biersack-Littmark style nuclear repulsion potential
        in Hartree with atom pair distances in Angstrom.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            Atomic numbers of all atoms in the batch.
        distances : torch.Tensor
            Distances between all atom pairs in the batch.
        idx_i : torch.Tensor
            Indices of the first atom of each pair.
        idx_j : torch.Tensor
            Indices of the second atom of each pair.

        Returns
        -------
        torch.Tensor
            Nuclear repulsion atom energy contribution
        
        """
        
        # Convert distances from model unit to Angstrom
        distances_ang = distances*self.distances_model2Ang
        
        # Compute atomic number dependent function
        za = atomic_numbers**torch.abs(self.a_exponent)
        a_ij = torch.abs(self.a_coefficient)/(za[idx_i] + za[idx_j])
        
        # Compute screening function
        arguments = distances/a_ij
        coefficients = torch.nn.functional.normalize(
            torch.abs(self.phi_coefficients), p=1.0, dim=0)
        exponents = torch.abs(self.phi_exponents)
        phi = torch.sum(
            coefficients[None, ...]*torch.exp(
                -exponents[None, ...]*arguments[..., None]), 
            dim=1)

        # Compute nuclear repulsion potential
        repulsion = (
            self.ke
            * atomic_numbers[idx_i]*atomic_numbers[idx_j]/distances_ang
            * phi
            * cutoffs)

        # Summarize and convert repulsion potential
        Erep = self.energies_Hatree2model*utils.segment_sum(
            repulsion, idx_i, device=self.device)

        return Erep

        
