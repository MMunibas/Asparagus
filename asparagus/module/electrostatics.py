import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Callable, Any

import torch

from asparagus import layer
from asparagus import utils
from asparagus import settings

__all__ = [
    'Damped_electrostatics',
    'MLMM_electrostatics',
    ]

# ======================================
#  Damped Multipole Electrostatics
# ======================================


class Damped_electrostatics(torch.nn.Module):
    """
    Torch implementation of a damped point charge and, eventually, atomic
    dipole and quadrupoles electrostatic model that avoids singularities at
    very close atom pair distances.

    Parameters
    ----------
    cutoff: float
        Electrostatic interaction cutoff distance.
    cutoff_short_range: float
        Damping range electrostatic cutoff distance.
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type
    unit_properties: dict, optional, default None
        Dictionary with the units of the model properties to initialize correct
        conversion factors.
    switch_fn: (str, callable), optional, default None
        Switch function for the short range cutoff.
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
    atomic_quadrupoles: bool, optional, default False
        Flag if atomic quadrupoles are predicted and to include in the
        electrostatic interaction potential computation
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: float,
        cutoff_short_range: float,
        device: str,
        dtype: 'dtype',
        unit_properties: Dict[str, str] = None,
        switch_fn: Union[str, Callable] = 'Poly6',
        truncation: str = 'force',
        atomic_dipoles: bool = False,
        atomic_quadrupoles: bool = False,
        **kwargs
    ):

        super(Damped_electrostatics, self).__init__()

        # Assign module variable parameters from configuration
        self.dtype = dtype
        self.device = device

        # Assign variables
        self.cutoff = torch.tensor(
            cutoff, device=self.device, dtype=self.dtype)
        if cutoff_short_range is None or cutoff == cutoff_short_range:
            self.cutoff_short_range = torch.tensor(
                cutoff, device=self.device, dtype=self.dtype)
        else:
            self.cutoff_short_range = torch.tensor(
            cutoff_short_range, device=self.device, dtype=self.dtype)

        # Assign switch function
        switch_class = layer.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(
            self.cutoff_short_range, device=self.device, dtype=self.dtype)

        # Set property units for parameter scaling
        self.set_unit_properties(unit_properties)

        # Assign truncation method
        if truncation is None or truncation.lower() == 'none':
            self.potential_fn = Damped_electrostatics_NoShift(
                self.cutoff,
                self.cutoff_short_range,
                self.kehalf,
                self.switch_fn,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() == 'potential':
            self.potential_fn = Damped_electrostatics_ShiftedPotential(
                self.cutoff,
                self.cutoff_short_range,
                self.kehalf,
                self.switch_fn,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = Damped_electrostatics_ShiftedForce(
                self.cutoff,
                self.cutoff_short_range,
                self.kehalf,
                self.switch_fn,
                atomic_dipoles,
                atomic_quadrupoles)
        else:
            raise SyntaxError(
                "Truncation method of the Coulomb potential "
                + f"'{truncation:}' is unknown!\n"
                + "Available are 'None', 'potential', 'force'.")

        return

    def __str__(self):
        return "Damped Atomic Multipole Electrostatics"

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
    
        # Convert 1/(2*4*pi*epsilon) from e**2/eV/Ang to model units
        kehalf_ase = 7.199822675975274
        kehalf = torch.tensor(
            kehalf_ase*factor_charge**2/factor_energy/factor_positions,
            device=self.device, dtype=self.dtype)
        self.register_buffer(
            'kehalf', kehalf)

        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute shielded electrostatic interaction between atom center point 
        charges and dipoles.

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary added by module results
        
        """

        # Compute damped Coulomb potential
        Eelec_pair = self.potential_fn(batch)

        # Sum up electrostatic atom pair contribution of each atom
        Eelec_atom = torch.zeros_like(batch['atomic_energies']).scatter_add_(
            0,
            batch['idx_u'],
            Eelec_pair)

        # Add electrostatic atomic energy contributions
        batch['atomic_energies'] = batch['atomic_energies'] + Eelec_atom

        # If verbose, store a copy of the electrostatic atomic energy 
        # contributions
        if verbose:
            batch['electrostatic_atomic_energies'] = Eelec_atom.detach()

        return batch


class Damped_electrostatics_NoShift(torch.nn.Module):
    """
    Torch implementation of a damped point charge and, eventually, atomic
    dipole electrostatic pair interaction without applying any cutoff method

    Parameters
    ----------
    cutoff: float
        Electrostatic interaction cutoff distance
    cutoff_short_range: float
        Damping range electrostatic cutoff distance
    kehalf: float
        Half of the Coulomb potential factor
    switch_fn: (str, callable), optional, default None
        Switch function for the short range cutoff.
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation
    atomic_quadrupoles: bool, optional, default False
        Flag if atomic quadrupoles are predicted and to include in the
        electrostatic interaction potential computation
    **kwargs
        Additional keyword arguments

    """

    def __init__(
        self,
        cutoff: float,
        cutoff_short_range: float,
        kehalf: float,
        switch_fn: Callable,
        atomic_dipoles: bool,
        atomic_quadrupoles: bool,
        **kwargs
    ):

        super(Damped_electrostatics_NoShift, self).__init__()

        # Assign parameters
        self.cutoff = cutoff
        self.cutoff_short_range = cutoff_short_range
        self.kehalf = kehalf
        self.switch_fn = switch_fn
        self.atomic_dipoles = atomic_dipoles
        self.atomic_quadrupoles = atomic_quadrupoles
        
        # Atomic quadrupoles can only be included if atomic dipolesa are 
        # included as well
        if not self.atomic_dipoles and self.atomic_quadrupoles:
            raise SyntaxError(
                "Electrostatic interaction including atomic quadrupoles also "
                "requires the inclusion of atomic dipoles!")

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
        distances = batch['distances_uv']
        distances_damped = torch.sqrt(distances**2 + 1.0)
        switch_damped = self.switch_fn(distances)
        switch_ordinary = 1.0 - switch_damped
        chi = (switch_damped/distances_damped + switch_ordinary/distances)

        # Gather atomic charge pairs
        atomic_charges_u = batch['atomic_charges'][batch['idx_u']]
        atomic_charges_v = batch['atomic_charges'][batch['idx_v']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_u*atomic_charges_v*chi

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if self.atomic_dipoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi

            # Normalize atom pair vectors
            chi_vectors = batch['vectors_uv']/distances.unsqueeze(-1)

            # Gather atomic dipole pairs
            atomic_dipoles_u = batch['atomic_dipoles'][batch['idx_u']]
            atomic_dipoles_v = batch['atomic_dipoles'][batch['idx_v']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_uv = torch.sum(chi_vectors*atomic_dipoles_v, dim=1)
            dot_vu = torch.sum(chi_vectors*atomic_dipoles_u, dim=1)

            # Compute damped charge-dipole electrostatics (times 2 to counter
            # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
            # Usually 1/r**3 but here one 1/r is already included in the
            # normalization of the atom pair connection vector.
            Eelec = Eelec + 2.0*atomic_charges_u*dot_uv*chi2

            # Compute damped dipole-dipole electrostatics
            # The 1/r**2 in the second term is included in the
            # normalization of both atom pair connection vectors.
            Eelec = Eelec + (
                torch.sum(atomic_dipoles_u*atomic_dipoles_v, dim=1)
                - 3*dot_uv*dot_vu
                )*chi3

        if self.atomic_quadrupoles:
            
            # Here, only the charge-quadrupole interaction is included!

            # Gather atomic quadrupole pairs
            # atomic_quadrupoles_u = batch['atomic_quadrupoles'][batch['idx_u']]
            atomic_quadrupoles_v = batch['atomic_quadrupoles'][batch['idx_v']]

            # Normalize traceless outer product
            # Here, traceless outer procuct already divided by 3.
            traceless_outer_product = (
                batch['outer_product_uv']
                - torch.diag_embed(
                    torch.tile(
                        batch['outer_product_uv'].diagonal(
                            dim1=-2, dim2=-1).mean(
                                dim=-1, keepdim=True),
                        (1, 3)
                    )
                )
            )
            chi_traceless_outer_product = (
                traceless_outer_product
                / torch.square(distances).unsqueeze(-1).unsqueeze(-1)
            )

            # Sum up the product of quadrupole tensor and outer product 
            # components
            sum_uv = torch.sum(
                chi_traceless_outer_product*atomic_quadrupoles_v, dim=(1, 2))

            # Compute damped charge-quadrupole electrostatics (by physics times
            # 0.5 but also times 2 (= 1) to counter kehalf = ke/2, as 
            # charge(i)-quadrupole(j) != charge(j)-quadrupole(i)).
            # Usually 1/r**5 but here one 1/r**2 is already included in the
            # normalization of the outer product.
            Eelec = Eelec + atomic_charges_u*sum_uv*chi3

        # Sum electrostatic contributions
        Eelec = self.kehalf*Eelec

        # Apply interaction cutoff
        Eelec = torch.where(
            distances <= self.cutoff,
            Eelec,
            torch.zeros_like(Eelec))

        return Eelec


class Damped_electrostatics_ShiftedPotential(torch.nn.Module):
    """
    Torch implementation of a damped point charge and, eventually, atomic
    dipole electrostatic pair interaction applying the shifted potential cutoff
    method

    Parameters
    ----------
    cutoff: float
        Electrostatic interaction cutoff distance.
    cutoff_short_range: float
        Damping range electrostatic cutoff distance.
    kehalf: float
        Half of the Coulomb potential factor
    switch_fn: (str, callable), optional, default None
        Switch function for the short range cutoff.
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation
    atomic_quadrupoles: bool, optional, default False
        Flag if atomic quadrupoles are predicted and to include in the
        electrostatic interaction potential computation
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: float,
        cutoff_short_range: float,
        kehalf: float,
        switch_fn: Callable,
        atomic_dipoles: bool,
        atomic_quadrupoles: bool,
        **kwargs
    ):

        super(Damped_electrostatics_ShiftedPotential, self).__init__()

        # Assign parameters
        self.cutoff = cutoff
        self.cutoff_short_range = cutoff_short_range
        self.kehalf = kehalf
        self.switch_fn = switch_fn
        self.atomic_dipoles = atomic_dipoles
        self.atomic_quadrupoles = atomic_quadrupoles

        # Atomic quadrupoles can only be included if atomic dipolesa are 
        # included as well
        if not self.atomic_dipoles and self.atomic_quadrupoles:
            raise SyntaxError(
                "Electrostatic interaction including atomic quadrupoles also "
                "requires the inclusion of atomic dipoles!")

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
        distances = batch['distances_uv']
        distances_damped = torch.sqrt(distances**2 + 1.0)
        switch_damped = self.switch_fn(distances)
        switch_ordinary = 1.0 - switch_damped
        chi = (switch_damped/distances_damped + switch_ordinary/distances)
        chi_shift = 1.0/self.cutoff

        # Gather atomic charge pairs
        atomic_charges_u = batch['atomic_charges'][batch['idx_u']]
        atomic_charges_v = batch['atomic_charges'][batch['idx_v']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_u*atomic_charges_v*(chi - chi_shift)

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if self.atomic_dipoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi
            chi2_shift = chi_shift**2
            chi3_shift = chi2_shift*chi_shift

            # Normalize atom pair vectors
            chi_vectors = batch['vectors_uv']/distances.unsqueeze(-1)

            # Gather atomic dipole pairs
            atomic_dipoles_u = batch['atomic_dipoles'][batch['idx_u']]
            atomic_dipoles_v = batch['atomic_dipoles'][batch['idx_v']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_uv = torch.sum(chi_vectors*atomic_dipoles_v, dim=1)
            dot_vu = torch.sum(chi_vectors*atomic_dipoles_u, dim=1)

            # Compute damped charge-dipole electrostatics (times 2 to counter
            # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
            Eelec = Eelec + 2.0*atomic_charges_u*dot_uv*(chi2 - chi2_shift)

            # Compute damped dipole-dipole electrostatics
            Eelec = Eelec + (
                torch.sum(atomic_dipoles_u*atomic_dipoles_v, dim=1)
                - 3*dot_uv*dot_vu
                )*(chi3 - chi3_shift)

        if self.atomic_quadrupoles:
            
            # Here, only the charge-quadrupole interaction is included!

            # Gather atomic quadrupole pairs
            # atomic_quadrupoles_u = batch['atomic_quadrupoles'][batch['idx_u']]
            atomic_quadrupoles_v = batch['atomic_quadrupoles'][batch['idx_v']]

            # Normalize traceless outer product
            # Here, traceless outer procuct already divided by 3.
            traceless_outer_product = (
                batch['outer_product_uv']
                - torch.diag_embed(
                    torch.tile(
                        batch['outer_product_uv'].diagonal(
                            dim1=-2, dim2=-1).mean(
                                dim=-1, keepdim=True),
                        (1, 3)
                    )
                )
            )
            chi_traceless_outer_product = (
                traceless_outer_product
                / torch.square(distances).unsqueeze(-1).unsqueeze(-1)
            )

            # Sum up the product of quadrupole tensor and outer product 
            # components
            sum_uv = torch.sum(
                chi_traceless_outer_product*atomic_quadrupoles_v, dim=(1, 2))

            # Compute damped charge-quadrupole electrostatics (by physics times
            # 0.5 but also times 2 (= 1) to counter kehalf = ke/2, as 
            # charge(i)-quadrupole(j) != charge(j)-quadrupole(i)).
            # Usually 1/r**5 but here one 1/r**2 is already included in the
            # normalization of the outer product.
            Eelec = Eelec + atomic_charges_u*sum_uv*(chi3 - chi3_shift)

        # Sum electrostatic contributions
        Eelec = self.kehalf*Eelec

        # Apply interaction cutoff
        Eelec = torch.where(
            distances <= self.cutoff,
            Eelec,
            torch.zeros_like(Eelec))

        return Eelec


class Damped_electrostatics_ShiftedForce(torch.nn.Module):
    """
    Torch implementation of a damped point charge and, eventually, atomic
    dipole electrostatic pair interaction applying the shifted force cutoff
    method

    Parameters
    ----------
    cutoff: float
        Electrostatic interaction cutoff distance.
    cutoff_short_range: float
        Damping range electrostatic cutoff distance.
    kehalf: float
        Half of the Coulomb potential factor
    switch_fn: (str, callable), optional, default None
        Switch function for the short range cutoff.
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation
    atomic_quadrupoles: bool, optional, default False
        Flag if atomic quadrupoles are predicted and to include in the
        electrostatic interaction potential computation
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: float,
        cutoff_short_range: float,
        kehalf: float,
        switch_fn: Callable,
        atomic_dipoles: bool,
        atomic_quadrupoles: bool,
        **kwargs
    ):

        super(Damped_electrostatics_ShiftedForce, self).__init__()

        # Assign parameters
        self.cutoff = cutoff
        self.cutoff_short_range = cutoff_short_range
        self.kehalf = kehalf
        self.switch_fn = switch_fn
        self.atomic_dipoles = atomic_dipoles
        self.atomic_quadrupoles = atomic_quadrupoles

        # Atomic quadrupoles can only be included if atomic dipolesa are 
        # included as well
        if not self.atomic_dipoles and self.atomic_quadrupoles:
            raise SyntaxError(
                "Electrostatic interaction including atomic quadrupoles also "
                "requires the inclusion of atomic dipoles!")

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
        distances = batch['distances_uv']
        distances_damped = torch.sqrt(distances**2 + 1.0)
        switch_damped = self.switch_fn(distances)
        switch_ordinary = 1.0 - switch_damped
        chi = (switch_damped/distances_damped + switch_ordinary/distances)
        cutoff2 = self.cutoff**2
        chi_shift = 2.0/self.cutoff - distances/cutoff2

        # Gather atomic charge pairs
        atomic_charges_u = batch['atomic_charges'][batch['idx_u']]
        atomic_charges_v = batch['atomic_charges'][batch['idx_v']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_u*atomic_charges_v*(chi - chi_shift)

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if self.atomic_dipoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi
            cutoff3 = cutoff2*self.cutoff
            chi2_shift = (
                3.0/cutoff2 - 2.0*distances/cutoff3)
            cutoff4 = cutoff3*self.cutoff
            chi3_shift = (
                4.0/cutoff3 - 3.0*distances/cutoff4)

            # Normalize atom pair vectors
            chi_vectors = batch['vectors_uv']/distances.unsqueeze(-1)

            # Gather atomic dipole pairs
            atomic_dipoles_u = batch['atomic_dipoles'][batch['idx_u']]
            atomic_dipoles_v = batch['atomic_dipoles'][batch['idx_v']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_uv = torch.sum(chi_vectors*atomic_dipoles_v, dim=1)
            dot_vu = torch.sum(chi_vectors*atomic_dipoles_u, dim=1)

            # Compute damped charge-dipole electrostatics (times 2 to counter
            # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
            Eelec = Eelec + 2.0*atomic_charges_u*dot_uv*(chi2 - chi2_shift)

            # Compute damped dipole-dipole electrostatics
            Eelec = Eelec + (
                torch.sum(atomic_dipoles_u*atomic_dipoles_v, dim=1)
                - 3*dot_uv*dot_vu
                )*(chi3 - chi3_shift)

        if self.atomic_quadrupoles:
            
            # Here, only the charge-quadrupole interaction is included!

            # Gather atomic quadrupole pairs
            # atomic_quadrupoles_u = batch['atomic_quadrupoles'][batch['idx_u']]
            atomic_quadrupoles_v = batch['atomic_quadrupoles'][batch['idx_v']]

            # Normalize traceless outer product
            # Here, traceless outer procuct already divided by 3.
            traceless_outer_product = (
                batch['outer_product_uv']
                - torch.diag_embed(
                    torch.tile(
                        batch['outer_product_uv'].diagonal(
                            dim1=-2, dim2=-1).mean(
                                dim=-1, keepdim=True),
                        (1, 3)
                    )
                )
            )
            chi_traceless_outer_product = (
                traceless_outer_product
                / torch.square(distances).unsqueeze(-1).unsqueeze(-1)
            )

            # Sum up the product of quadrupole tensor and outer product 
            # components
            sum_uv = torch.sum(
                chi_traceless_outer_product*atomic_quadrupoles_v, dim=(1, 2))

            # Compute damped charge-quadrupole electrostatics (by physics times
            # 0.5 but also times 2 (= 1) to counter kehalf = ke/2, as 
            # charge(i)-quadrupole(j) != charge(j)-quadrupole(i)).
            # Usually 1/r**5 but here one 1/r**2 is already included in the
            # normalization of the outer product.
            Eelec = Eelec + atomic_charges_u*sum_uv*(chi3 - chi3_shift)

        # Sum electrostatic contributions
        Eelec = self.kehalf*Eelec

        # Apply interaction cutoff
        Eelec = torch.where(
            distances <= self.cutoff,
            Eelec,
            torch.zeros_like(Eelec))

        return Eelec


class MLMM_electrostatics(torch.nn.Module):
    """
    Torch implementation of MM point charge interaction with ML atomic
    charges and, eventually, dipoles and quadrupoles.

    Parameters
    ----------
    cutoff: float
        Electrostatic interaction cutoff distance.
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type
    unit_properties: dict, optional, default None
        Dictionary with the units of the model properties to initialize correct
        conversion factors.
    truncation: str, optional, default 'None'
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
    atomic_quadrupoles: bool, optional, default False
        Flag if atomic quadrupoles are predicted and to include in the
        electrostatic interaction potential computation
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: float,
        device: str,
        dtype: 'dtype',
        unit_properties: Dict[str, str] = None,
        truncation: str = 'None',
        atomic_dipoles: bool = False,
        atomic_quadrupoles: bool = False,
        **kwargs
    ):

        super(MLMM_electrostatics, self).__init__()

        # Assign module variable parameters from configuration
        self.dtype = dtype
        self.device = device

        # Assign variables
        self.cutoff = torch.tensor(
            cutoff, device=self.device, dtype=self.dtype)

        # Set property units for parameter scaling
        self.set_unit_properties(unit_properties)

        # Assign truncation method
        if truncation is None or truncation.lower() == 'none':
            self.potential_fn = MLMM_electrostatics_NoShift(
                self.cutoff,
                self.ke,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() == 'potential':
            self.potential_fn = MLMM_electrostatics_NoShift(
                self.cutoff,
                self.ke,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = MLMM_electrostatics_NoShift(
                self.cutoff,
                self.ke,
                atomic_dipoles,
                atomic_quadrupoles)
        else:
            raise SyntaxError(
                "Truncation method of the Coulomb potential "
                + f"'{truncation:}' is unknown!\n"
                + "Available are 'None', 'potential', 'force'.")

        return

    def __str__(self):
        return "MM Atomic Charge to ML Atomic Multipole Electrostatics"

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
    
        # Convert 1/(2*4*pi*epsilon) from e**2/eV/Ang to model units
        ke_ase = 14.399645351950548
        ke = torch.tensor(
            ke_ase*factor_charge**2/factor_energy/factor_positions,
            device=self.device, dtype=self.dtype)
        self.register_buffer(
            'ke', ke)

        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute shielded electrostatic interaction between atom center point 
        charges and dipoles.

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of data tensors
        verbose: bool, optional, default False
            If True, store extended model property contributions in the data
            dictionary.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary added by module results
        
        """

        # Compute damped Coulomb potential
        Eelec_pair = self.potential_fn(batch)

        # Sum up electrostatic atom pair contribution of each atom
        Eelec_atom = torch.zeros_like(batch['atomic_energies']).scatter_add_(
            0,
            batch['mlmm_idx_u'],
            Eelec_pair)

        # Add electrostatic atomic energy contributions
        batch['atomic_energies'] = batch['atomic_energies'] + Eelec_atom

        # If verbose, store a copy of the electrostatic atomic energy 
        # contributions
        if verbose:
            batch['electrostatic_atomic_energies'] = Eelec_atom.detach()

        return batch


class MLMM_electrostatics_NoShift(torch.nn.Module):
    """
    Torch implementation of MM point charge interaction with ML atomic
    charges and, eventually, dipoles and quadrupoles without applying 
    any cutoff method

    Parameters
    ----------
    cutoff: float
        Electrostatic interaction cutoff distance
    ke: float
        Coulomb potential factor
    atomic_dipoles: bool, optional, default False
        Flag if atomic dipoles are predicted and to include in the
        electrostatic interaction potential computation
    atomic_quadrupoles: bool, optional, default False
        Flag if atomic quadrupoles are predicted and to include in the
        electrostatic interaction potential computation
    **kwargs
        Additional keyword arguments

    """

    def __init__(
        self,
        cutoff: float,
        ke: float,
        atomic_dipoles: bool,
        atomic_quadrupoles: bool,
        **kwargs
    ):

        super(MLMM_electrostatics_NoShift, self).__init__()

        # Assign parameters
        self.cutoff = cutoff
        self.ke = ke
        self.atomic_dipoles = atomic_dipoles
        self.atomic_quadrupoles = atomic_quadrupoles
        
        # Atomic quadrupoles can only be included if atomic dipolesa are 
        # included as well
        if not self.atomic_dipoles and self.atomic_quadrupoles:
            raise SyntaxError(
                "Electrostatic interaction including atomic quadrupoles also "
                "requires the inclusion of atomic dipoles!")

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

        # Compute reciprocal distances
        distances = batch['mlmm_distances_uv']
        chi = 1.0/distances

        # Gather atomic charge pairs
        ml_atomic_charges_u = batch['atomic_charges'][batch['mlmm_idx_u']]
        mm_atomic_charges_v = (
            batch['reference']['atomic_charges'][batch['mlmm_idx_v']])

        # Compute ML/MM charge-charge electrostatics
        Eelec = ml_atomic_charges_u*mm_atomic_charges_v*chi

        # Compute ML/MM dipole-charge electrostatics
        if self.atomic_dipoles:

            # Compute powers of reciprocal distances
            chi2 = chi**2

            # Normalize atom pair vectors
            chi_vectors = batch['mlmm_vectors_uv']/distances.unsqueeze(-1)

            # Gather ML atomic dipoles
            ml_atomic_dipoles_u = batch['atomic_dipoles'][batch['mlmm_idx_u']]

            # Compute dot products of atom pair vector and atomic dipole
            dot_vu = torch.sum(chi_vectors*ml_atomic_dipoles_u, dim=1)

            # Compute ML/MM dipole-charge electrostatics
            # Usually 1/r**3 but here one 1/r is already included in the
            # normalization of the atom pair connection vector.
            Eelec = Eelec - mm_atomic_charges_v*dot_vu*chi2

        # Compute ML/MM quadrupole-charge electrostatics
        if self.atomic_quadrupoles:
            
            # Compute powers of reciprocal distances
            chi3 = chi2*chi

            # Gather atomic quadrupole pairs
            ml_atomic_quadrupoles_u = (
                batch['atomic_quadrupoles'][batch['mlmm_idx_u']])

            # Normalize traceless outer product
            # Here, traceless outer procuct already divided by 3.
            traceless_outer_product = (
                batch['mlmm_outer_product_uv']
                - torch.diag_embed(
                    torch.tile(
                        batch['mlmm_outer_product_uv'].diagonal(
                            dim1=-2, dim2=-1).mean(
                                dim=-1, keepdim=True),
                        (1, 3)
                    )
                )
            )
            chi_traceless_outer_product = (
                traceless_outer_product
                / torch.square(distances).unsqueeze(-1).unsqueeze(-1)
            )

            # Sum up the product of quadrupole tensor and outer product 
            # components
            sum_vu = torch.sum(
                chi_traceless_outer_product*ml_atomic_quadrupoles_u,
                dim=(1, 2))

            # Compute ML/MM quadrupole-charge electrostatics 
            # Usually 1/r**5 but here one 1/r**2 is already included in the
            # normalization of the outer product.
            Eelec = Eelec + mm_atomic_charges_v*sum_vu*chi3

        # Sum electrostatic contributions
        Eelec = self.ke*Eelec

        # Apply interaction cutoff
        Eelec = torch.where(
            distances <= self.cutoff,
            Eelec,
            torch.zeros_like(Eelec))

        return Eelec
