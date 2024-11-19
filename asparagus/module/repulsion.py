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
    cutoff: float
        Upper cutoff distance
    cuton: float
        Lower cutoff distance starting switch-off function
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

    """

    def __init__(
        self,
        cutoff: float,
        cuton: float,
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

        # Assign cutoff radii and prepare switch-off parameters
        self.cutoff = torch.tensor(
            [cutoff], device=self.device, dtype=self.dtype)
        if cuton is None:
            self.cuton = torch.tensor(
                [0.0], device=self.device, dtype=self.dtype)
            self.switchoff_range = torch.tensor(
                [cutoff], device=self.device, dtype=self.dtype)
            self.use_switch = True
        elif cuton < cutoff:
            self.cuton = torch.tensor(
                [cuton], device=self.device, dtype=self.dtype)
            self.switchoff_range = torch.tensor(
                [cutoff - cuton], device=self.device, dtype=self.dtype)
            self.use_switch = True
        else:
            self.cuton = None
            self.switchoff_range = None
            self.use_switch = False

        # Initialize repulsion model parameters
        a_coefficient = 0.8854 # Bohr
        a_exponent = 0.23
        phi_coefficients = [0.18175, 0.50986, 0.28022, 0.02817]
        phi_exponents = [3.19980, 0.94229, 0.40290, 0.20162]

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
            factor_positions, _ = utils.check_units('Bohr', unit_positions)
        else:
            factor_energy, _ = utils.check_units(
                unit_properties.get('energy'), 'Hartree')
            factor_positions, _ = utils.check_units(
                'Bohr', unit_properties.get('positions'))

        # Convert
        # Distances: model to Bohr
        # Energies: Hartree to model
        self.register_buffer(
            "distances_model2Bohr",
            torch.tensor([factor_positions], dtype=self.dtype))
        self.register_buffer(
            "energies_Hatree2model",
            torch.tensor([factor_energy], dtype=self.dtype))

        # Convert e**2/(4*pi*epsilon) = 1 from 1/Hartree/Bohr to model units
        ke_au = 1.
        ke = torch.tensor(
            [ke_au*self.energies_Hatree2model/self.distances_model2Bohr],
            device=self.device, dtype=self.dtype)
        self.register_buffer('ke', ke)

        return

    def switch_fn(
        self,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a smooth switch factors from 1 to 0 in the range from 'cuton'
        to 'cutoff'.

        """

        x = (self.cutoff - distances) / self.switchoff_range

        return torch.where(
            distances < self.cuton,
            torch.ones_like(x),
            torch.where(
                distances >= self.cutoff,
                torch.zeros_like(x),
                ((6.0*x - 15.0)*x + 10.0)*x**3
                )
            )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        distances: torch.Tensor,
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

        # Compute switch-off function
        if self.use_switch:
            switch_off = self.switch_fn(distances)
        else:
            switch_off = torch.where(
                distances < self.cutoff,
                torch.ones_like(distances),
                torch.zeros_like(distances),
            )

        # Compute atomic number dependent function
        za = atomic_numbers**torch.abs(self.a_exponent)
        a_ij = (
            torch.abs(self.a_coefficient/self.distances_model2Bohr)
            / (za[idx_i] + za[idx_j]))

        # Compute screening function phi
        arguments = distances/a_ij
        coefficients = torch.nn.functional.normalize(
            torch.abs(self.phi_coefficients), p=1.0, dim=0)
        exponents = torch.abs(self.phi_exponents)
        phi = torch.sum(
            coefficients[None, ...]*torch.exp(
                -exponents[None, ...]*arguments[..., None]),
            dim=1)

        # Compute nuclear repulsion potential in model energy unit
        repulsion = (
            0.5*self.ke
            * atomic_numbers[idx_i]*atomic_numbers[idx_j]/distances
            * phi
            * switch_off)

        # Summarize and convert repulsion potential
        Erep = utils.scatter_sum(
            repulsion, idx_i, dim=0, shape=atomic_numbers.shape)

        return Erep
