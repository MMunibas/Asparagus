import os
import logging
from typing import Optional, Dict, Union, Any

import numpy as np

import asparagus
import tad_dftd3

import torch

from asparagus import utils
from asparagus import settings

from ase import units

import time


class D3_dispersion(torch.nn.Module):
    """
    Torch implementation of Grimme's D3 method (only Becke-Johnson damping is
    implemented)

    Grimme, Stefan, et al. "A consistent and accurate ab initio parametrization
    of density functional dispersion correction (DFT-D) for the 94 elements
    H-Pu." The Journal of Chemical Physics 132, 15 (2010): 154104.

    Update of the implementation according with respect to the tad-dftd3 module
    on git: https://github.com/dftd3/tad-dftd3 (15.11.2024)

    Parameters
    ----------
    cutoff: float
        Upper cutoff distance
    cuton: float
        Lower cutoff distance starting switch-off function
    trainable: bool, optional, default True
        If True the dispersion parameters are trainable
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type
    unit_properties: dict, optional, default {}
        Dictionary with the units of the model properties to initialize correct
        conversion factors.
    truncation: str, optional, default 'force'
        Truncation method of the Dispersion potential at the cutoff range:
            None, 'None': 
                No Dispersion potential shift applied
            'potential':
                Apply shifted Dispersion potential method
                    V_shifted(r) = V_Coulomb(r) - V_Coulomb(r_cutoff)
            'force', 'forces':
                Apply shifted Dispersion force method
                    V_shifted(r) = V_Dispersion(r) - V_Dispersion(r_cutoff)
                        - (dV_Dispersion/dr)|r_cutoff  * (r - r_cutoff)
    d3_s6: float, optional, default 1.0000
        d3_s6 dispersion parameter
    d3_s8: float, optional, default 0.9171
        d3_s8 dispersion parameter
    d3_a1: float, optional, default 0.3385
        d3_a1 dispersion parameter
    d3_a2: float, optional, default 2.8830
        d3_a2 dispersion parameter

    """

    def __init__(
        self,
        cutoff: float,
        cuton: float,
        trainable: bool,
        device: str,
        dtype: 'dtype',
        unit_properties: Optional[Dict[str, str]] = None,
        truncation: Optional[str] = 'force',
        d3_s6: Optional[float] = None,
        d3_s8: Optional[float] = None,
        d3_a1: Optional[float] = None,
        d3_a2: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize Grimme D3 dispersion model.
        
        """

        super(D3_dispersion, self).__init__()

        # Relative filepath to package folder
        package_directory = os.path.dirname(os.path.abspath(__file__))

        # Assign variables
        self.dtype = dtype
        self.device = device

        # Load tables with reference values
        self.d3_rcov = torch.from_numpy(
            np.load(os.path.join(package_directory, "grimme_d3", "rcov.npy"))
            ).to(dtype).to(device)
        self.d3_rcn = torch.from_numpy(
            np.genfromtxt(
                os.path.join(package_directory, "grimme_d3", "refcn.csv"),
                delimiter=',')
            ).to(dtype).to(device)
        self.d3_rcn_max = torch.max(self.d3_rcn, dim=-1, keepdim=True)[0]
        self.d3_rc6 = torch.from_numpy(
            np.load(os.path.join(package_directory, "grimme_d3", "rc6.npy"))
            ).to(dtype).to(device)
        self.d3_r2r4 = torch.from_numpy(
            np.load(os.path.join(package_directory, "grimme_d3", "r2r4.npy"))
            ).to(dtype).to(device)
        
        # Assign truncation method
        if truncation is None or truncation.lower() == 'none':
            self.potential_fn = self.dispersion_fn
        elif truncation.lower() == 'potential':
            self.potential_fn = self.dispersion_sp_fn
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = self.dispersion_sf_fn
        else:
            raise SyntaxError(
                "Truncation method of the Dispersion potential "
                + f"'{truncation:}' is unknown!\n"
                + "Available are 'None', 'potential', 'force'.")

        # Initialize global dispersion correction parameters 
        # (default values for HF)
        if d3_s6 is None:
            d3_s6 = 1.0000
        if d3_s8 is None:
            d3_s8 = 0.9171
        if d3_a1 is None:
            d3_a1 = 0.3385
        if d3_a2 is None:
            d3_a2 = 2.8830
        
        if trainable:
            self.d3_s6 = torch.nn.Parameter(
                torch.tensor([d3_s6], device=device, dtype=dtype))
            self.d3_s8 = torch.nn.Parameter(
                torch.tensor([d3_s8], device=device, dtype=dtype))
            self.d3_a1 = torch.nn.Parameter(
                torch.tensor([d3_a1], device=device, dtype=dtype))
            self.d3_a2 = torch.nn.Parameter(
                torch.tensor([d3_a2], device=device, dtype=dtype))
        else:
            self.register_buffer(
                "d3_s6", torch.tensor([d3_s6], device=device, dtype=dtype))
            self.register_buffer(
                "d3_s8", torch.tensor([d3_s8], device=device, dtype=dtype))
            self.register_buffer(
                "d3_a1", torch.tensor([d3_a1], device=device, dtype=dtype))
            self.register_buffer(
                "d3_a2", torch.tensor([d3_a2], device=device, dtype=dtype))
        self.d3_k1 = torch.tensor([16.000], device=device, dtype=dtype)
        self.d3_k2 = torch.tensor([4./3.], device=device, dtype=dtype)
        self.d3_k3 = torch.tensor([-4.000], device=device, dtype=dtype)

        # Assign cutoff radii
        self.cutoff = cutoff
        self.cuton = cuton

        # Unit conversion factors
        self.set_unit_properties(unit_properties)

        # Prepare interaction switch-off range
        self.set_switch_of_range(cutoff, cuton)

        # Auxiliary parameter
        self.zero_dtype = torch.tensor(0.0, device=device, dtype=dtype)
        self.zero_double = torch.tensor(
            0.0, device=device, dtype=torch.float64)
        self.small_double = torch.tensor(
            1e-300, device=device, dtype=torch.float64)
        self.one_dtype = torch.tensor(1.0, device=device, dtype=dtype)
        self.max_dtype = torch.tensor(
            torch.finfo(dtype).max, device=device, dtype=dtype)

        return
        
    def __str__(self):
        return "D3 Dispersion"

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
            torch.tensor(
                [factor_positions], device=self.device, dtype=self.dtype))
        self.register_buffer(
            "energies_Hatree2model", 
            torch.tensor(
                [factor_energy], device=self.device, dtype=self.dtype))

        # Update interaction switch-off range units
        self.set_switch_of_range(self.cutoff, self.cuton)

        return

    def set_switch_of_range(
        self,
        cutoff: float,
        cuton: float,
    ):
        """
        Prepare switch-off parameters

        """
        
        self.cutoff = (
            torch.tensor([cutoff], device=self.device, dtype=self.dtype)
            * self.distances_model2Bohr)
        if cuton is None or cuton == cutoff:
            self.cuton = None
            self.switchoff_range = None
            self.use_switch = False
        else:
            self.cuton = (
                torch.tensor([cuton], device=self.device, dtype=self.dtype)
                * self.distances_model2Bohr)
            self.switchoff_range = (
                torch.tensor(
                    [cutoff - cuton], device=self.device, dtype=self.dtype)
                * self.distances_model2Bohr)
            self.use_switch = True

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

    def get_cn(
        self,
        atomic_numbers: torch.Tensor,
        atomic_numbers_i: torch.Tensor,
        atomic_numbers_j: torch.Tensor,
        distances: torch.Tensor,
        switch_off: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coordination numbers by adding an inverse damping function.
        
        """

        # Compute atom pairs covalent radii
        rcov_ij = (
            torch.gather(self.d3_rcov, 0, atomic_numbers_i) 
            + torch.gather(self.d3_rcov, 0, atomic_numbers_j))
        
        cn_ij = (
            1.0/(1.0 + torch.exp(-self.d3_k1 * (rcov_ij/distances - 1.0))))
        if self.use_switch:
            cn_ij = cn_ij*switch_off

        return utils.scatter_sum(
            cn_ij, idx_i, dim=0, shape=atomic_numbers.shape)

    def get_weights(
        self,
        atomic_numbers: torch.Tensor,
        cn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dispersion weights
        
        """

        # Get reference atomic coordination numbers of atom pairs ij
        rcn = self.d3_rcn[atomic_numbers]

        # Selection of non-zero reference coordination numbers
        mask_rcn = rcn >= 0

        # Compute deviation between reference coordination number and actual
        # coordination number
        dcn = (rcn - cn.unsqueeze(-1)).type(torch.double)

        # Compute and normalize coordination number Gaussian weights in double 
        # precision and convert back to dtype
        gaussian_weights = torch.where(
            mask_rcn,
            torch.exp(self.d3_k3*dcn**2),
            self.zero_double)
        norm = torch.where(
            mask_rcn,
            torch.sum(gaussian_weights, dim=-1, keepdim=True),
            self.small_double)
        mask_norm = norm == 0
        norm = torch.where(
            mask_norm,
            self.small_double,
            norm)
        gaussian_weights = (gaussian_weights/norm).type(self.dtype)

        # Prevent exceptional values in the gaussian weights, either because
        # the norm was zero or the weight is to large.
        exceptional = torch.logical_or(
            mask_norm, gaussian_weights > self.max_dtype)
        if torch.any(exceptional):
            rcn_max = self.d3_rcn_max[atomic_numbers]
            gaussian_weights = torch.where(
                exceptional,
                torch.where(rcn == rcn_max, self.one_dtype, self.zero_dtype),
                gaussian_weights)
        gaussian_weights = torch.where(
            mask_rcn,
            gaussian_weights,
            self.zero_dtype)

        return gaussian_weights

    def get_c6(
        self,
        atomic_numbers_i: torch.Tensor,
        atomic_numbers_j: torch.Tensor,
        weigths: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute atomic c6 dispersion coefficients
        
        """

        # Collect reference c6 dispersion coefficients of atom pairs ij
        rc6 = self.d3_rc6[atomic_numbers_i, atomic_numbers_j]

        # Collect atomic weights of atom pairs ij
        weights_i = weigths[idx_i]
        weights_j = weigths[idx_j]
        weights_ij = weights_i.unsqueeze(-1)*weights_j.unsqueeze(-2)

        # Compute atomic c6 dispersion coefficients
        c6 = torch.sum(torch.sum(torch.mul(weights_ij, rc6), dim=-1), dim=-1)

        return c6

    def dispersion_fn(
        self,
        distances: torch.Tensor,
        distances6: torch.Tensor,
        distances8: torch.Tensor,
        switch_off: torch.Tensor,
        c6: torch.Tensor,
        c8: torch.Tensor,
        fct6: torch.Tensor,
        fct8: torch.Tensor,
        damp_c6: torch.Tensor,
        damp_c8: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute atomic D3 dispersion energy

        """
        
        # Compute atomic dispersion energy contributions
        e6 = -0.5*self.d3_s6*c6*damp_c6
        e8 = -0.5*self.d3_s8*c8*damp_c8

        # Apply switch-off function
        edisp = switch_off*(e6 + e8)
        
        return edisp

    def dispersion_sp_fn(
        self,
        distances: torch.Tensor,
        distances6: torch.Tensor,
        distances8: torch.Tensor,
        switch_off: torch.Tensor,
        c6: torch.Tensor,
        c8: torch.Tensor,
        fct6: torch.Tensor,
        fct8: torch.Tensor,
        damp_c6: torch.Tensor,
        damp_c8: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute shifted potential atomic D3 dispersion energy

        """

        # Compute all required powers of the cutoff distance
        cutoff2 = self.cutoff**2
        cutoff6 = cutoff2**3
        cutoff8 = cutoff6*cutoff2
        denominator6 = cutoff6 + fct6
        denominator8 = cutoff8 + fct8
        
        # Compute force shifted atomic dispersion energy contributions
        e6 = -0.5*self.d3_s6*c6*(damp_c6 - 1.0/denominator6)
        e8 = -0.5*self.d3_s8*c8*(damp_c8 - 1.0/denominator8)

        # Apply switch-off function
        edisp = switch_off*(e6 + e8)
        
        return edisp

    def dispersion_sf_fn(
        self,
        distances: torch.Tensor,
        distances6: torch.Tensor,
        distances8: torch.Tensor,
        switch_off: torch.Tensor,
        c6: torch.Tensor,
        c8: torch.Tensor,
        fct6: torch.Tensor,
        fct8: torch.Tensor,
        damp_c6: torch.Tensor,
        damp_c8: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute shifted force  atomic D3 dispersion energy

        """

        # Compute all required powers of the cutoff distance
        cutoff2 = self.cutoff**2
        cutoff6 = cutoff2**3
        cutoff8 = cutoff6*cutoff2
        denominator6 = cutoff6 + fct6
        denominator8 = cutoff8 + fct8
        
        # Compute force shifted atomic dispersion energy contributions
        e6 = -0.5*self.d3_s6*c6*(
            damp_c6 - 1.0/denominator6 
            + 6.0*cutoff6/denominator6**2*(distances/self.cutoff - 1.0))
        e8 = -0.5*self.d3_s8*c8*(
            damp_c8 - 1.0/denominator8 
            + 8.0*cutoff8/denominator8**2*(distances/self.cutoff - 1.0))

        # Apply switch-off function
        edisp = switch_off*(e6 + e8)
        
        return edisp

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        distances: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Grimme's D3 dispersion energy in Hartree with atom pair 
        distances in Bohr.

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
            Dispersion atom energy contribution
        
        """
        
        # Convert distances from model unit to Bohr
        distances_d3 = distances*self.distances_model2Bohr

        # Compute switch-off function
        if self.use_switch:
            switch_off = self.switch_fn(distances_d3)
        else:
            switch_off = torch.where(
                distances_d3 < self.cutoff,
                torch.ones_like(distances_d3),
                torch.zeros_like(distances_d3),
            )

        # Gather atomic numbers of atom pairs ij
        atomic_numbers_i = torch.gather(atomic_numbers, 0, idx_i)
        atomic_numbers_j = torch.gather(atomic_numbers, 0, idx_j)

        # Compute coordination numbers and of atom pairs ij
        cn = self.get_cn(
            atomic_numbers,
            atomic_numbers_i,
            atomic_numbers_j,
            distances_d3,
            switch_off,
            idx_i,
            idx_j)
        
        # Compute atomic weights
        weights = self.get_weights(
            atomic_numbers,
            cn)
        
        # Compute atomic C6 and C8 coefficients
        c6 = self.get_c6(
            atomic_numbers_i,
            atomic_numbers_j,
            weights,
            idx_i,
            idx_j)
        qq = (
            3.0
            * torch.gather(self.d3_r2r4, 0, atomic_numbers_i)
            * torch.gather(self.d3_r2r4, 0, atomic_numbers_j))
        c8 = qq*c6

        # Compute the powers of the atom pair distances
        distances2 = distances_d3**2
        distances6 = distances2**3
        distances8 = distances6*distances2

        # Apply rational Becke-Johnson damping.
        fct = self.d3_a1*torch.sqrt(qq) + self.d3_a2
        fct2 = fct**2
        fct6 = fct2**3
        fct8 = fct6*fct2
        damp_c6 = 1.0/(distances6 + fct6)
        damp_c8 = 1.0/(distances8 + fct8)

        # Compute atomic dispersion energy contributions
        print("")
        Edisp = self.dispersion_fn(
            distances_d3,
            distances6,
            distances8,
            switch_off,
            c6,
            c8,
            fct6,
            fct8,
            damp_c6,
            damp_c8)
        print("Switch-off: ", Edisp.detach().numpy())
        Edisp = self.dispersion_sp_fn(
            distances_d3,
            distances6,
            distances8,
            switch_off,
            c6,
            c8,
            fct6,
            fct8,
            damp_c6,
            damp_c8)
        print("Potential shifted switch-off: ", Edisp.detach().numpy())
        Edisp = self.dispersion_sf_fn(
            distances_d3,
            distances6,
            distances8,
            switch_off,
            c6,
            c8,
            fct6,
            fct8,
            damp_c6,
            damp_c8)
        print("Force shifted switch-off: ", Edisp.detach().numpy())
        Edisp = self.potential_fn(
            distances_d3,
            distances6,
            distances8,
            switch_off,
            c6,
            c8,
            fct6,
            fct8,
            damp_c6,
            damp_c8)

        # Return system dispersion energies and convert to model energy unit
        return self.energies_Hatree2model*utils.scatter_sum(
            Edisp, idx_i, dim=0, shape=atomic_numbers.shape)





atomic_numbers = torch.tensor([6, 8, 6, 8, 6, 8])
positions = torch.tensor(
    [
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.4],
        [50.0, 0.0, 0.0], [50.0, 0.0, 10.4],
        [100.0, 0.0, 0.0], [100.0, 0.0, 11.999999]
     ],
    dtype=torch.float32)
positions.requires_grad_(True)

param = {
    "s6": torch.tensor(1.0000, dtype=torch.float32),
    "s8": torch.tensor(0.9171, dtype=torch.float32),
    "a1": torch.tensor(0.3385, dtype=torch.float32),
    "a2": torch.tensor(2.8830, dtype=torch.float32),
}
edisp = tad_dftd3.dftd3(atomic_numbers, positions/units.Bohr, param)
edisp *= units.Hartree
print(edisp)

gradient = torch.autograd.grad(
    torch.sum(edisp),
    positions)
print(gradient)


idx_i = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
idx_j = torch.tensor([1, 0, 3, 2, 5, 4], dtype=torch.int64)
distances = torch.sqrt(
    torch.sum(
        (positions[idx_i] - positions[idx_j])**2,
        dim=1)
    )

d3disp = D3_dispersion(12.0, 10.0, False, 'cpu', torch.float32, truncation='none')

edisp = d3disp(atomic_numbers, distances, idx_i, idx_j)
print(edisp)

gradient = torch.autograd.grad(
    torch.sum(edisp),
    positions)
print(gradient)

