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
                self.offset2,
                self.switch_fn,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() == 'potential':
            self.potential_fn = Damped_electrostatics_ShiftedPotential(
                self.cutoff,
                self.cutoff_short_range,
                self.kehalf,
                self.offset2,
                self.switch_fn,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = Damped_electrostatics_ShiftedForce(
                self.cutoff,
                self.cutoff_short_range,
                self.kehalf,
                self.offset2,
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
    
        # Convert 1/(2*4*pi*epsilon) from eV*Ang/e**2 to model units
        kehalf_ase = 7.199822675975274
        kehalf = torch.tensor(
            kehalf_ase*factor_energy*factor_positions/factor_charge**2,
            device=self.device, dtype=self.dtype)
        self.register_buffer('kehalf', kehalf)

        # Convert damping offset from Ang**2 to model unit
        offset2_ase = 1.0**2
        offset2 = torch.tensor(
            offset2_ase*factor_positions**2,
            device=self.device, dtype=self.dtype)
        # self.register_buffer('offset2', offset2)
        self.offset2 = offset2

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
    offset2: float
        Damping squared damping distance offset that the atom pair distance
        will converge when going to zero.
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
        offset2: float,
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
        self.offset2 = offset2
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
        distances_damped = torch.sqrt(distances**2 + self.offset2)
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

            # Compute traceless outer product
            outer_product = (
                batch['vectors_uv'].unsqueeze(-1)
                * batch['vectors_uv'].unsqueeze(-2))
            traceless_outer_product = (
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
    offset2: float
        Damping squared damping distance offset that the atom pair distance
        will converge when going to zero.
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
        offset2: float,
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
        self.offset2 = offset2
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
        distances_damped = torch.sqrt(distances**2 + self.offset2)
        switch_damped = self.switch_fn(distances)
        switch_ordinary = 1.0 - switch_damped
        chi = (switch_damped/distances_damped + switch_ordinary/distances)
        chi_shift = 1.0/self.cutoff

        # Gather atomic charge pairs
        atomic_charges_u = batch['atomic_charges'][batch['idx_u']]
        atomic_charges_v = batch['atomic_charges'][batch['idx_v']]

        # Compute damped charge-charge electrostatics
        Eelec = atomic_charges_u*atomic_charges_v*(chi - chi_shift)

        # Compute damped electrostatics between atomic charges, dipoles and,
        # eventually, quadrupoles
        if self.atomic_dipoles or self.atomic_quadrupoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi
            chi2_shift = chi_shift**2
            chi3_shift = chi2_shift*chi_shift

            # Compute damped charge-dipole and dipole-dipole electrostatics
            if self.atomic_dipoles:
                
                Eelec = Eelec + self.get_atomic_dipole_contributions(
                    atomic_charges_u,
                    batch['atomic_dipoles'][batch['idx_u']],
                    batch['atomic_dipoles'][batch['idx_v']],
                    chi2,
                    chi3,
                    chi2_shift,
                    chi3_shift,
                    batch['vectors_uv']/distances.unsqueeze(-1)
                )

            # Here, only the charge-quadrupole interaction is included!
            if self.atomic_quadrupoles:
            
                # Compute traceless outer product
                outer_product_uv = (
                    batch['vectors_uv'].unsqueeze(-1)
                    * batch['vectors_uv'].unsqueeze(-2))
                traceless_outer_product_uv = (
                    outer_product_uv
                    - torch.diag_embed(
                        torch.tile(
                            outer_product_uv.diagonal(
                                dim1=-2, dim2=-1).mean(
                                    dim=-1, keepdim=True),
                            (1, 3)
                        )
                    )
                )
                traceless_outer_product_uv = (
                    traceless_outer_product_uv
                    / torch.square(distances).unsqueeze(-1).unsqueeze(-1)
                )
                
                # Compute atomic quadrupole contributions
                Eelec = Eelec + self.get_atomic_quadrupole_contributions(
                    atomic_charges_u,
                    batch['atomic_quadrupoles'][batch['idx_v']],
                    chi3,
                    chi3_shift,
                    traceless_outer_product_uv
                )

        # Sum electrostatic contributions
        Eelec = self.kehalf*Eelec

        # Apply interaction cutoff
        Eelec = torch.where(
            distances <= self.cutoff,
            Eelec,
            torch.zeros_like(Eelec))

        return Eelec

    def get_atomic_dipole_contributions(
        self,
        atomic_charges_u: torch.Tensor,
        atomic_dipoles_u: torch.Tensor,
        atomic_dipoles_v: torch.Tensor,
        chi2: torch.Tensor,
        chi3: torch.Tensor,
        chi2_shift: torch.Tensor,
        chi3_shift: torch.Tensor,
        normalized_vectors_uv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute damped electrostatic charge and atomic dipole contributions

        Parameters
        ----------
        atomic_charges_u: torch.Tensor
            Atomic charges of pair atom u
        atomic_dipoles_u: torch.Tensor
            Atomic dipoles of pair atom u
        atomic_dipoles_v: torch.Tensor
            Atomic dipoles of pair atom v
        chi2: torch.Tensor
            Squared reciprocal distance vector
        chi3: torch.Tensor
            Cubed reciprocal distance vector
        chi2_shift: torch.Tensor
            Squared reciprocal shift distance vector for shifting method
        chi3_shift: torch.Tensor
            Cubed reciprocal shift distance vector for shifting method
        normalized_vectors_uv: torch.Tensor
            Normalized connection vectors of atom pair u and v

        Returns
        -------
        torch.Tensor
            Damped and shifted electrostatic atomic dipole contributions

        """

        # Compute dot products of atom pair vector and atomic dipole
        dot_uv = torch.sum(normalized_vectors_uv*atomic_dipoles_v, dim=1)
        dot_vu = torch.sum(normalized_vectors_uv*atomic_dipoles_u, dim=1)

        # Compute damped charge-dipole electrostatics (times 2 to counter
        # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
        Eelec_cd = 2.0*atomic_charges_u*dot_uv*(chi2 - chi2_shift)

        # Compute damped dipole-dipole electrostatics
        Eelec_dd = (
            torch.sum(atomic_dipoles_u*atomic_dipoles_v, dim=1)
            - 3*dot_uv*dot_vu
            )*(chi3 - chi3_shift)
        
        return Eelec_cd + Eelec_dd

    def get_atomic_quadrupole_contributions(
        self,
        atomic_charges_u: torch.Tensor,
        atomic_quadrupoles_v: torch.Tensor,
        chi3: torch.Tensor,
        chi3_shift: torch.Tensor,
        traceless_outer_product_uv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute damped electrostatic charge and atomic dipole contributions

        Parameters
        ----------
        atomic_charges_u: torch.Tensor
            Atomic charges of pair atom u
        atomic_quadrupoles_v: torch.Tensor
            Atomic quadrupoles of pair atom v
        chi3: torch.Tensor
            Cubed reciprocal distance vector
        chi3_shift: torch.Tensor
            Cubed reciprocal shift distance vector for shifting method
        traceless_outer_product_uv: torch.Tensor
            Detraced outer product of normalized connection vectors of atom
            pair u and v

        Returns
        -------
        torch.Tensor
            Damped and shifted electrostatic atomic quadrupole contributions

        """

        # Sum up the product of quadrupole tensor and outer product 
        # components
        sum_uv = torch.sum(
            traceless_outer_product_uv*atomic_quadrupoles_v, dim=(1, 2))

        # Compute damped charge-quadrupole electrostatics (by physics times
        # 0.5 but also times 2 (= 1) to counter kehalf = ke/2, as 
        # charge(i)-quadrupole(j) != charge(j)-quadrupole(i)).
        # Usually 1/r**5 but here one 1/r**2 is already included in the
        # normalization of the outer product.
        Eelec_cq = atomic_charges_u*sum_uv*(chi3 - chi3_shift)

        return Eelec_cq


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
    offset2: float
        Damping squared damping distance offset that the atom pair distance
        will converge when going to zero.
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
        offset2: float,
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
        self.offset2 = offset2
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
        distances_damped = torch.sqrt(distances**2 + self.offset2)
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

        # Compute damped electrostatics between atomic charges, dipoles and,
        # eventually, quadrupoles
        if self.atomic_dipoles or self.atomic_quadrupoles:

            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi
            cutoff3 = cutoff2*self.cutoff
            chi2_shift = (
                3.0/cutoff2 - 2.0*distances/cutoff3)
            cutoff4 = cutoff3*self.cutoff
            chi3_shift = (
                4.0/cutoff3 - 3.0*distances/cutoff4)

            # Compute damped charge-dipole and dipole-dipole electrostatics
            if self.atomic_dipoles:
                
                Eelec = Eelec + self.get_atomic_dipole_contributions(
                    atomic_charges_u,
                    batch['atomic_dipoles'][batch['idx_u']],
                    batch['atomic_dipoles'][batch['idx_v']],
                    chi2,
                    chi3,
                    chi2_shift,
                    chi3_shift,
                    batch['vectors_uv']/distances.unsqueeze(-1)
                )

            # Here, only the charge-quadrupole interaction is included!
            if self.atomic_quadrupoles:
            
                # Compute traceless outer product
                outer_product_uv = (
                    batch['vectors_uv'].unsqueeze(-1)
                    * batch['vectors_uv'].unsqueeze(-2))
                traceless_outer_product_uv = (
                    outer_product_uv
                    - torch.diag_embed(
                        torch.tile(
                            outer_product_uv.diagonal(
                                dim1=-2, dim2=-1).mean(
                                    dim=-1, keepdim=True),
                            (1, 3)
                        )
                    )
                )
                traceless_outer_product_uv = (
                    traceless_outer_product_uv
                    / torch.square(distances).unsqueeze(-1).unsqueeze(-1)
                )
                
                # Compute atomic quadrupole contributions
                Eelec = Eelec + self.get_atomic_quadrupole_contributions(
                    atomic_charges_u,
                    batch['atomic_quadrupoles'][batch['idx_v']],
                    chi3,
                    chi3_shift,
                    traceless_outer_product_uv
                )

        # Sum electrostatic contributions
        Eelec = self.kehalf*Eelec

        # Apply interaction cutoff
        Eelec = torch.where(
            distances <= self.cutoff,
            Eelec,
            torch.zeros_like(Eelec))

        return Eelec

    def get_atomic_dipole_contributions(
        self,
        atomic_charges_u: torch.Tensor,
        atomic_dipoles_u: torch.Tensor,
        atomic_dipoles_v: torch.Tensor,
        chi2: torch.Tensor,
        chi3: torch.Tensor,
        chi2_shift: torch.Tensor,
        chi3_shift: torch.Tensor,
        normalized_vectors_uv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute damped electrostatic charge and atomic dipole contributions

        Parameters
        ----------
        atomic_charges_u: torch.Tensor
            Atomic charges of pair atom u
        atomic_dipoles_u: torch.Tensor
            Atomic dipoles of pair atom u
        atomic_dipoles_v: torch.Tensor
            Atomic dipoles of pair atom v
        chi2: torch.Tensor
            Squared reciprocal distance vector
        chi3: torch.Tensor
            Cubed reciprocal distance vector
        chi2_shift: torch.Tensor
            Squared reciprocal shift distance vector for shifting method
        chi3_shift: torch.Tensor
            Cubed reciprocal shift distance vector for shifting method
        normalized_vectors_uv: torch.Tensor
            Normalized connection vectors of atom pair u and v

        Returns
        -------
        torch.Tensor
            Damped and shifted electrostatic atomic dipole contributions

        """

        # Compute dot products of atom pair vector and atomic dipole
        dot_uv = torch.sum(normalized_vectors_uv*atomic_dipoles_v, dim=1)
        dot_vu = torch.sum(normalized_vectors_uv*atomic_dipoles_u, dim=1)

        # Compute damped charge-dipole electrostatics (times 2 to counter
        # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
        Eelec_cd = 2.0*atomic_charges_u*dot_uv*(chi2 - chi2_shift)

        # Compute damped dipole-dipole electrostatics
        Eelec_dd = (
            torch.sum(atomic_dipoles_u*atomic_dipoles_v, dim=1)
            - 3*dot_uv*dot_vu
            )*(chi3 - chi3_shift)
        
        return Eelec_cd + Eelec_dd

    def get_atomic_quadrupole_contributions(
        self,
        atomic_charges_u: torch.Tensor,
        atomic_quadrupoles_v: torch.Tensor,
        chi3: torch.Tensor,
        chi3_shift: torch.Tensor,
        traceless_outer_product_uv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute damped electrostatic charge and atomic dipole contributions

        Parameters
        ----------
        atomic_charges_u: torch.Tensor
            Atomic charges of pair atom u
        atomic_quadrupoles_v: torch.Tensor
            Atomic quadrupoles of pair atom v
        chi3: torch.Tensor
            Cubed reciprocal distance vector
        chi3_shift: torch.Tensor
            Cubed reciprocal shift distance vector for shifting method
        traceless_outer_product_uv: torch.Tensor
            Detraced outer product of normalized connection vectors of atom
            pair u and v

        Returns
        -------
        torch.Tensor
            Damped and shifted electrostatic atomic quadrupole contributions

        """

        # Sum up the product of quadrupole tensor and outer product 
        # components
        sum_uv = torch.sum(
            traceless_outer_product_uv*atomic_quadrupoles_v, dim=(1, 2))

        # Compute damped charge-quadrupole electrostatics (by physics times
        # 0.5 but also times 2 (= 1) to counter kehalf = ke/2, as 
        # charge(i)-quadrupole(j) != charge(j)-quadrupole(i)).
        # Usually 1/r**5 but here one 1/r**2 is already included in the
        # normalization of the outer product.
        Eelec_cq = atomic_charges_u*sum_uv*(chi3 - chi3_shift)

        return Eelec_cq


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
                self.ke,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() == 'potential':
            self.potential_fn = MLMM_electrostatics_NoShift(
                self.ke,
                atomic_dipoles,
                atomic_quadrupoles)
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = MLMM_electrostatics_NoShift(
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

        # Convert 1/(4*pi*epsilon) from eV*Ang/e**2 to model units
        ke_ase = 14.399645351950548
        ke = torch.tensor(
            ke_ase*factor_energy*factor_positions/factor_charge**2,
            device=self.device, dtype=self.dtype)

        self.register_buffer('ke', ke)

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
            batch['mlmm_electrostatic_atomic_energies'] = Eelec_atom.detach()

        return batch


class MLMM_electrostatics_NoShift(torch.nn.Module):
    """
    Torch implementation of MM point charge interaction with ML atomic
    charges and, eventually, dipoles and quadrupoles without applying 
    any cutoff method, except the one applied to the neighbor list.

    Parameters
    ----------
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
        ke: float,
        atomic_dipoles: bool,
        atomic_quadrupoles: bool,
        **kwargs
    ):

        super(MLMM_electrostatics_NoShift, self).__init__()

        # Assign parameters
        self.ke = ke
        self.atomic_dipoles = atomic_dipoles
        self.atomic_quadrupoles = atomic_quadrupoles
        
        # Atomic quadrupoles can only be included if atomic dipoles are 
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
        mlmm_distances = batch['mlmm_distances_uv']
        mlmm_distances2 = torch.square(mlmm_distances)

        # Gather atomic pair charges
        ml_atomic_charges_u = batch['atomic_charges'][batch['mlmm_idx_u']]
        mm_atomic_charges_v = batch['mlmm_atomic_charges'][batch['mlmm_idx_v']]

        # Compute B terms, G terms and MLMM electrostatic interaction potential
        # according to expressions in https://doi.org/10.3390/ijms21010277 
        # and from implementation of AMP (https://doi.org/10.1021/jacs.4c17015)
        B0 = 1./mlmm_distances
        G0 = ml_atomic_charges_u*mm_atomic_charges_v
        Eelec = B0*G0

        if self.atomic_dipoles or self.atomic_quadrupoles:

            B1 = B0/mlmm_distances2
            G1 = torch.sum(
                (
                    batch['atomic_dipoles'][batch['mlmm_idx_u']]
                    * batch['mlmm_vectors_uv']
                ),
                dim=1,
                keepdim=False
            )*mm_atomic_charges_v
                
            Eelec = Eelec + B1*G1

            if self.atomic_quadrupoles:

                mlmm_outer_product = (
                    batch['mlmm_vectors_uv'].unsqueeze(-1)
                    * batch['mlmm_vectors_uv'].unsqueeze(-2)
                )
                mlmm_traceless_outer_product = (
                    mlmm_outer_product
                    - torch.diag_embed(
                        torch.tile(
                            mlmm_outer_product.diagonal(
                                dim1=-2, dim2=-1).mean(
                                    dim=-1, keepdim=True),
                            (1, 3)
                        )
                    )
                )
                B2 = 3.*B1/mlmm_distances2
                G2 = torch.sum(
                    (
                        batch['atomic_quadrupoles'][batch['mlmm_idx_u']]
                        * mlmm_traceless_outer_product
                    ),
                    dim=(1, 2)
                )*mm_atomic_charges_v
                Eelec = Eelec - B2*G2

        # Weight electrostatic contributions by prefactor
        Eelec = self.ke*Eelec

        return Eelec

