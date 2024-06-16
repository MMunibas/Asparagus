import os
import logging
from typing import Optional, Dict, Union, Any

import numpy as np

import torch

from .. import utils
from .. import settings

__all__ = ["D3_dispersion"]

#======================================
# Grimme Dispersion Correction
#======================================


class D3_dispersion(torch.nn.Module):
    """
    Torch implementation of Grimme's D3 method (only Becke-Johnson damping is
    implemented)

    Grimme, Stefan, et al. "A consistent and accurate ab initio parametrization
    of density functional dispersion correction (DFT-D) for the 94 elements
    H-Pu." The Journal of Chemical Physics 132, 15 (2010): 154104.

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
        dtype: object,
        unit_properties: Optional[Dict[str, str]] = None,
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
        self.d3_c6ab = torch.from_numpy(
            np.load(os.path.join(package_directory, "grimme_d3", "c6ab.npy"))
            ).to(dtype).to(device)
        self.d3_r0ab = torch.from_numpy(
            np.load(os.path.join(package_directory, "grimme_d3", "r0ab.npy"))
            ).to(dtype).to(device)
        self.d3_rcov = torch.from_numpy(
            np.load(os.path.join(package_directory, "grimme_d3", "rcov.npy"))
            ).to(dtype).to(device)
        self.d3_r2r4 = torch.from_numpy(
            np.load(os.path.join(package_directory, "grimme_d3", "r2r4.npy"))
            ).to(dtype).to(device)
        
        # Maximum number of coordination complexes
        self.d3_maxc = 5
        
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
                torch.tensor([d3_s6], dtype=dtype)).to(device)
            self.d3_s8 = torch.nn.Parameter(
                torch.tensor([d3_s8], dtype=dtype)).to(device)
            self.d3_a1 = torch.nn.Parameter(
                torch.tensor([d3_a1], dtype=dtype)).to(device)
            self.d3_a2 = torch.nn.Parameter(
                torch.tensor([d3_a2], dtype=dtype)).to(device)
        else:
            self.register_buffer(
                "d3_s6", torch.tensor([d3_s6], device=device, dtype=dtype))
            self.register_buffer(
                "d3_s8", torch.tensor([d3_s8], device=device, dtype=dtype))
            self.register_buffer(
                "d3_a1", torch.tensor([d3_a1], device=device, dtype=dtype))
            self.register_buffer(
                "d3_a2", torch.tensor([d3_a2], device=device, dtype=dtype))
        self.d3_k1 = torch.tensor([16.000], dtype=dtype).to(device)
        self.d3_k2 = torch.tensor([4./3.], dtype=dtype).to(device)
        self.d3_k3 = torch.tensor([-4.000], dtype=dtype).to(device)
        
        # Prepare interaction switch-off range
        self.cutoff = torch.tensor([cutoff], dtype=dtype).to(device)
        if cuton is None or cuton == cutoff:
            self.cuton = None
            self.switchoff_range = None
            self.use_switch = False
        else:
            self.cuton = torch.tensor([cuton], dtype=dtype).to(device)
            self.switchoff_range = torch.tensor(
                [cutoff - cuton], dtype=dtype).to(device)
            self.use_switch = True

        # Unit conversion factors
        self.set_unit_properties(unit_properties)

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

        return

    def _smootherstep(
        self,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a smooth step from 1 to 0 in the range from 'cuton to 
        'cutoff'.
        
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

    def _ncoord(
        self,
        atomic_numbers_i: torch.Tensor,
        atomic_numbers_j: torch.Tensor, 
        distances: torch.Tensor, 
        idx_i: torch.Tensor,
        idx_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coordination numbers by adding an inverse damping function.
        
        """
        
        rco = (
            torch.gather(self.d3_rcov, 0, atomic_numbers_i) 
            + torch.gather(self.d3_rcov, 0, atomic_numbers_j))

        damp = 1.0/(1.0 + torch.exp(-self.d3_k1 * (rco/distances - 1.0)))
        if self.use_switch:
            damp = damp*self._smootherstep(distances)

        return utils.segment_sum(damp, idx_i, device=self.device)

    def _getc6(
        self,
        atomic_pair_numbers: torch.Tensor, 
        nci: torch.Tensor,  
        ncj: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate the c6 coefficient.

        """
        
        # Gather the relevant table entries
        c6ab_ = utils.gather_nd(self.d3_c6ab, atomic_pair_numbers)
        #.type(nci.dtype)
        
        # Calculate c6 coefficients
        c6mem = -1.0e99 * torch.ones_like(nci, device=self.device)
        r_save = 1.0e99 * torch.ones_like(nci, device=self.device)
        
        rsum = torch.zeros_like(nci, device=self.device)
        csum = torch.zeros_like(nci, device=self.device)
        
        for i in range(self.d3_maxc):
            for j in range(self.d3_maxc):
                
                cn0 = c6ab_[:, i, j, 0]
                cn1 = c6ab_[:, i, j, 1]
                cn2 = c6ab_[:, i, j, 2]

                r = (cn1 - nci) ** 2 + (cn2 - ncj) ** 2
                r_save = torch.where(r < r_save, r, r_save)

                c6mem = torch.where(r < r_save, cn0, c6mem)

                tmp1 = torch.exp(self.d3_k3 * r)
                rsum = rsum + torch.where(
                    cn0 > 0.0, 
                    tmp1, 
                    torch.zeros_like(tmp1, device=self.device))
                csum = csum + torch.where(
                    cn0 > 0.0, 
                    tmp1*cn0, 
                    torch.zeros_like(tmp1, device=self.device))
                
        c6 = torch.where(rsum > 0.0, csum/rsum, c6mem)
        
        return c6

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

        # Compute all necessary quantities
        atomic_numbers_i = atomic_numbers[idx_i]
        atomic_numbers_j = atomic_numbers[idx_j]
        atomic_pair_numbers = torch.stack(
            [atomic_numbers_i, atomic_numbers_j], axis=1)

        # Compute coordination numbers
        nc = self._ncoord(
            atomic_numbers_i, atomic_numbers_j, distances_d3, idx_i, idx_j)
        nci = torch.gather(nc, 0, idx_i)
        ncj = torch.gather(nc, 0, idx_j)
        
        # Compute C6 and C8 coefficients
        c6 = self._getc6(atomic_pair_numbers, nci, ncj)
        c8 = (
            3.0*c6
            *torch.gather(self.d3_r2r4, 0, atomic_numbers_i)
            *torch.gather(self.d3_r2r4, 0, atomic_numbers_j))

        # Compute all required powers of the distance
        distances2 = distances_d3**2
        distances6 = distances2**3
        distances8 = distances6*distances2

        # Becke-Johnson damping only, because
        # zero-damping introduces spurious repulsion and is therefore not 
        # implemented.
        tmp = self.d3_a1*torch.sqrt(c8/c6) + self.d3_a2
        tmp2 = tmp**2
        tmp6 = tmp2**3
        tmp8 = tmp6*tmp2
        
        cut2 = self.cutoff**2
        cut6 = cut2**3
        cut8 = cut6*cut2
        
        cut6tmp6 = cut6 + tmp6
        cut8tmp8 = cut8 + tmp8
        
        # Compute dispersion energy
        e6 = (
            1.0/(distances6 + tmp6) - 1.0/cut6tmp6 
            + 6.0*cut6/cut6tmp6**2*(distances_d3/self.cutoff - 1.0))
        e8 = (
            1.0/(distances8 + tmp8) - 1.0/cut8tmp8 
            + 8.0*cut8/cut8tmp8**2*(distances_d3/self.cutoff - 1.0))

        e6 = torch.where(distances_d3 < self.cutoff, e6, torch.zeros_like(e6))
        e8 = torch.where(distances_d3 < self.cutoff, e8, torch.zeros_like(e8))

        e6 = -0.5*self.d3_s6*c6*e6
        e8 = -0.5*self.d3_s8*c8*e8

        # Summarize and convert dispersion energy
        Edisp = self.energies_Hatree2model*utils.segment_sum(
            e6 + e8, idx_i, device=self.device)

        return Edisp
