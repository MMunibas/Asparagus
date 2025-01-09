import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from asparagus import layer
from asparagus import utils
from asparagus import settings

__all__ = [
    'PC_shielded_electrostatics',
    'PC_damped_electrostatics',
    'PC_Dipole_damped_electrostatics'
    ]

# ======================================
#  Point Charge Electrostatics
# ======================================


class PC_shielded_electrostatics(torch.nn.Module):
    """
    Torch implementation of a shielded point charge electrostatic model that
    avoids singularities at very close atom pair distances.

    Parameters
    ----------
    cutoff: float
        interaction cutoff distance.
    cutoff_short_range: float
        Short range cutoff distance.
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type
    unit_properties: dict, optional, default None
        Dictionary with the units of the model properties to initialize correct
        conversion factors.
    switch_fn: (str, callable), optional, default None
        Switch function for the short range cutoff.
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: float,
        cutoff_short_range: float,
        device: str,
        dtype: 'dtype',
        unit_properties: Optional[Dict[str, str]] = None,
        switch_fn: Optional[Union[str, object]] = 'Poly6',
        **kwargs
    ):

        super(PC_shielded_electrostatics, self).__init__()

        # Assign variables
        self.cutoff = cutoff
        if cutoff_short_range is None or cutoff == cutoff_short_range:
            self.cutoff_short_range = cutoff
            self.split_distance = False
        else:
            self.cutoff_short_range = cutoff_short_range
            self.split_distance = True
        self.cutoff_squared = cutoff**2
        
        # Assign module variable parameters from configuration
        self.dtype = dtype
        self.device = device

        # Assign switch function
        switch_class = layer.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(
            self.cutoff_short_range, device=self.device, dtype=self.dtype)

        # Set property units for parameter scaling
        self.set_unit_properties(unit_properties)

        return

    def __str__(self):
        return "Shielded Point Charge Electrostatics"

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
            [kehalf_ase*factor_charge**2/factor_energy/factor_positions],
            device=self.device, dtype=self.dtype)
        self.register_buffer(
            'kehalf', kehalf)

        return

    def forward(
        self,
        properties: Dict[str, torch.Tensor],
        distances: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute shielded electrostatic interaction between atom center point 
        charges.

        Parameters
        ----------
        properties: dict
            system properties including atomic charges
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

        # Grep atomic charges
        atomic_charges = properties['atomic_charges']

        # Gather atomic charge pairs
        atomic_charges_i = torch.gather(atomic_charges, 0, idx_i)
        atomic_charges_j = torch.gather(atomic_charges, 0, idx_j)

        # Compute shielded distances
        distances_shielded = torch.sqrt(distances**2 + 1.0)

        # Compute switch weights
        switch_off_weights = self.switch_fn(distances)
        switch_on_weights = 1.0 - switch_off_weights

        # Compute electrostatic potential
        if self.split_distance:

            # Shifted Force Coulomb potential method
            # Compute ordinary (unshielded) and shielded contributions
            E_ordinary = (
                1.0/distances
                + distances/self.cutoff_squared
                - 2.0/self.cutoff)
            E_shielded = (
                1.0/distances_shielded
                + distances_shielded/self.cutoff_squared
                - 2.0/self.cutoff)

            # Combine electrostatic contributions
            E = (
                self.kehalf*atomic_charges_i*atomic_charges_j*(
                    switch_off_weights*E_shielded
                    + switch_on_weights*E_ordinary))
            
        else:

            # Compute ordinary (unshielded) and shielded contributions
            E_ordinary = 1.0/distances
            E_shielded = 1.0/distances_shielded

            # Combine electrostatic contributions
            E = (
                self.kehalf*atomic_charges_i*atomic_charges_j
                * (
                    switch_off_weights*E_shielded 
                    + switch_on_weights*E_ordinary)
                )

        # Apply interaction cutoff
        E = torch.where(
            distances <= self.cutoff,
            E,                      # distance <= cutoff
            torch.zeros_like(E))    # distance > cutoff

        # Sum up electrostatic atom pair contribution of each atom
        return utils.scatter_sum(
             E, idx_i, dim=0, shape=atomic_charges.shape)


class PC_damped_electrostatics(torch.nn.Module):
    """
    Torch implementation of a damped point charge electrostatic model that
    avoids singularities at very close atom pair distances.

    Parameters
    ----------
    cutoff: float
        interaction cutoff distance.
    cutoff_short_range: float
        Short range cutoff distance.
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
                    
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: float,
        cutoff_short_range: float,
        device: str,
        dtype: 'dtype',
        unit_properties: Optional[Dict[str, str]] = None,
        switch_fn: Optional[Union[str, object]] = 'Poly6',
        truncation: Optional[str] = 'force',
        **kwargs
    ):

        super(PC_damped_electrostatics, self).__init__()

        # Assign variables
        self.cutoff = cutoff
        if cutoff_short_range is None or cutoff == cutoff_short_range:
            self.cutoff_short_range = cutoff
        else:
            self.cutoff_short_range = cutoff_short_range
        self.cutoff_squared = cutoff**2
        
        # Assign module variable parameters from configuration
        self.dtype = dtype
        self.device = device

        # Assign switch function
        switch_class = layer.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(
            self.cutoff_short_range, device=self.device, dtype=self.dtype)

        # Assign truncation method
        if truncation is None or truncation.lower() == 'none':
            self.potential_fn = self.damped_coulomb_fn
        elif truncation.lower() == 'potential':
            self.potential_fn = self.damped_coulomb_sp_fn
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = self.damped_coulomb_sf_fn
        else:
            raise SyntaxError(
                "Truncation method of the Coulomb potential "
                + f"'{truncation:}' is unknown!\n"
                + "Available are 'None', 'potential', 'force'.")

        # Set property units for parameter scaling
        self.set_unit_properties(unit_properties)

        return

    def __str__(self):
        return "Shielded Point Charge Electrostatics"

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {}

    def damped_coulomb_fn(
        self,
        atomic_charges_i: torch.Tensor,
        atomic_charges_j: torch.Tensor,
        distances: torch.Tensor,
        distances_damped: torch.Tensor,
        switch_damped: torch.Tensor,
        switch_ordinary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Damped Coulomb potential

        """
        
        # Compute ordinary and damped contributions
        E_ordinary = 1.0/distances
        E_damped = 1.0/distances_damped

        # Compute damped electrostatics
        E = (
            self.kehalf*atomic_charges_i*atomic_charges_j
            * (switch_damped*E_damped + switch_ordinary*E_ordinary)
            )

        return E

    def damped_coulomb_sp_fn(
        self,
        atomic_charges_i: torch.Tensor,
        atomic_charges_j: torch.Tensor,
        distances: torch.Tensor,
        distances_damped: torch.Tensor,
        switch_damped: torch.Tensor,
        switch_ordinary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Damped & shifted potential Coulomb interaction

        """
        
        # Compute ordinary and damped contributions
        E_ordinary = 1.0/distances
        E_damped = 1.0/distances_damped
        E_shift = -1.0/self.cutoff

        # Compute damped electrostatics
        E = (
            self.kehalf*atomic_charges_i*atomic_charges_j
            * (switch_damped*E_damped + switch_ordinary*E_ordinary + E_shift)
            )

        return E

    def damped_coulomb_sf_fn(
        self,
        atomic_charges_i: torch.Tensor,
        atomic_charges_j: torch.Tensor,
        distances: torch.Tensor,
        distances_damped: torch.Tensor,
        switch_damped: torch.Tensor,
        switch_ordinary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Damped & shifted forces Coulomb interaction

        """
        
        # Compute ordinary and damped contributions
        E_ordinary = 1.0/distances
        E_damped = 1.0/distances_damped
        E_shift = distances/self.cutoff_squared - 2.0/self.cutoff

        # Compute damped electrostatics
        E = (
            self.kehalf*atomic_charges_i*atomic_charges_j
            * (switch_damped*E_damped + switch_ordinary*E_ordinary + E_shift)
            )

        return E

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
        # Coulomb factor ke is halfed to avoid double counting as atom pairs
        # are handled in both order i<->j and j<->i.
        kehalf_ase = 7.199822675975274
        kehalf = torch.tensor(
            [kehalf_ase*factor_charge**2/factor_energy/factor_positions],
            device=self.device, dtype=self.dtype)
        self.register_buffer(
            'kehalf', kehalf)

        return

    def forward(
        self,
        properties: Dict[str, torch.Tensor],
        distances: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute shielded electrostatic interaction between atom center point 
        charges.

        Parameters
        ----------
        properties: dict
            system properties including atomic charges
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

        # Grep atomic charges
        atomic_charges = properties['atomic_charges']

        # Gather atomic charge pairs
        atomic_charges_i = torch.gather(atomic_charges, 0, idx_i)
        atomic_charges_j = torch.gather(atomic_charges, 0, idx_j)

        # Compute shielded distances
        distances_damped = torch.sqrt(distances**2 + 1.0)

        # Compute switch weights
        switch_damped = self.switch_fn(distances)
        switch_ordinary = 1.0 - switch_damped

        # Compute damped electrostatic Coulomb potential
        E = self.potential_fn(
            atomic_charges_i,
            atomic_charges_j,
            distances,
            distances_damped,
            switch_damped,
            switch_ordinary)

        # Apply interaction cutoff
        E = torch.where(
            distances <= self.cutoff,
            E,                      # distance <= cutoff
            torch.zeros_like(E))    # distance > cutoff

        # Sum up electrostatic atom pair contribution of each atom
        return utils.scatter_sum(
             E, idx_i, dim=0, shape=atomic_charges.shape)


class PC_Dipole_damped_electrostatics(torch.nn.Module):
    """
    Torch implementation of a damped point charge and atomic dipole 
    electrostatic model that avoids singularities at very close atom pair
    distances.

    Parameters
    ----------
    cutoff: float
        interaction cutoff distance.
    cutoff_short_range: float
        Short range cutoff distance.
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
                    
    **kwargs
        Additional keyword arguments.

    """

    def __init__(
        self,
        cutoff: float,
        cutoff_short_range: float,
        device: str,
        dtype: 'dtype',
        unit_properties: Optional[Dict[str, str]] = None,
        switch_fn: Optional[Union[str, object]] = 'Poly6',
        truncation: Optional[str] = 'force',
        **kwargs
    ):

        super(PC_Dipole_damped_electrostatics, self).__init__()

        # Assign variables
        self.cutoff = cutoff
        if cutoff_short_range is None or cutoff == cutoff_short_range:
            self.cutoff_short_range = cutoff
        else:
            self.cutoff_short_range = cutoff_short_range
        self.cutoff_squared = cutoff**2
        self.cutoff_cubed = self.cutoff_squared*cutoff
        self.cutoff_quartic = self.cutoff_squared*self.cutoff_squared

        # Assign module variable parameters from configuration
        self.dtype = dtype
        self.device = device

        # Assign switch function
        switch_class = layer.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(
            self.cutoff_short_range, device=self.device, dtype=self.dtype)

        # Assign truncation method
        if truncation is None or truncation.lower() == 'none':
            self.potential_fn = self.damped_coulomb_fn
        elif truncation.lower() == 'potential':
            self.potential_fn = self.damped_coulomb_sp_fn
        elif truncation.lower() in ['force', 'forces']:
            self.potential_fn = self.damped_coulomb_sf_fn
        else:
            raise SyntaxError(
                "Truncation method of the Coulomb potential "
                + f"'{truncation:}' is unknown!\n"
                + "Available are 'None', 'potential', 'force'.")

        # Set property units for parameter scaling
        self.set_unit_properties(unit_properties)

        return

    def __str__(self):
        return "Shielded Point Charge and Atomic Dipole Electrostatics"

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {}

    def damped_coulomb_fn(
        self,
        atomic_charges_i: torch.Tensor,
        atomic_charges_j: torch.Tensor,
        atomic_dipoles_i: torch.Tensor,
        atomic_dipoles_j: torch.Tensor,
        distances: torch.Tensor,
        distances_damped: torch.Tensor,
        switch_damped: torch.Tensor,
        switch_ordinary: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Damped Coulomb potential

        """

        # Compute ordinary and damped contributions
        E_ordinary = 1.0/distances
        E_damped = 1.0/distances_damped

        # Compute damped reciprocal distances
        chi = (switch_damped*E_damped + switch_ordinary*E_ordinary)

        # Compute damped charge-charge electrostatics
        E = atomic_charges_i*atomic_charges_j*chi

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if vectors is not None:
            
            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi

            # Normalize atom pair vectors (vec/chi is canceled within formula)
            chi_vectors = vectors/distances[:, None]

            # Compute dot products of atom pair vector and atomic dipole
            dot_ij = torch.sum(chi_vectors*atomic_dipoles_j, axis=1)
            dot_ji = torch.sum(chi_vectors*atomic_dipoles_i, axis=1)

            # Compute damped charge-dipole electrostatics (times 2 to counter
            # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
            E = E + 2.0*atomic_charges_i*dot_ij*chi2

            # Compute damped dipole-dipole electrostatics
            E = E + (
                torch.sum(atomic_dipoles_i*atomic_dipoles_j, axis=1)
                - 3*dot_ij*dot_ji
                )*chi3

        # Sum electrostatic contributions
        E = self.kehalf*E

        return E

    def damped_coulomb_sp_fn(
        self,
        atomic_charges_i: torch.Tensor,
        atomic_charges_j: torch.Tensor,
        atomic_dipoles_i: torch.Tensor,
        atomic_dipoles_j: torch.Tensor,
        distances: torch.Tensor,
        distances_damped: torch.Tensor,
        switch_damped: torch.Tensor,
        switch_ordinary: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Damped & shifted potential Coulomb interaction

        """
        
        # Compute ordinary and damped contributions
        E_ordinary = 1.0/distances
        E_damped = 1.0/distances_damped

        # Compute damped reciprocal distances
        chi = (switch_damped*E_damped + switch_ordinary*E_ordinary)
        chi_shift = 1.0/self.cutoff

        # Compute damped charge-charge electrostatics
        E = atomic_charges_i*atomic_charges_j*(chi - chi_shift)

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if vectors is not None:
            
            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi
            chi2_shift = chi_shift**2
            chi3_shift = chi2_shift*chi_shift

            # Adjust atom pair vectors
            chi_vectors = vectors/distances[:, None]

            # Compute dot products of atom pair vector and atomic dipole
            dot_ij = torch.sum(chi_vectors*atomic_dipoles_j, axis=1)
            dot_ji = torch.sum(chi_vectors*atomic_dipoles_i, axis=1)

            # Compute damped charge-dipole electrostatics (times 2 to counter
            # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
            E = E + 2.0*atomic_charges_i*dot_ij*(chi2 - chi2_shift)

            # Compute damped dipole-dipole electrostatics
            E = E + (
                torch.sum(atomic_dipoles_i*atomic_dipoles_j, axis=1)
                - 3*dot_ij*dot_ji
                )*(chi3 - chi3_shift)

        # Sum electrostatic contributions
        E = self.kehalf*E

        return E

    def damped_coulomb_sf_fn(
        self,
        atomic_charges_i: torch.Tensor,
        atomic_charges_j: torch.Tensor,
        atomic_dipoles_i: torch.Tensor,
        atomic_dipoles_j: torch.Tensor,
        distances: torch.Tensor,
        distances_damped: torch.Tensor,
        switch_damped: torch.Tensor,
        switch_ordinary: torch.Tensor,
        vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Damped & shifted forces Coulomb interaction

        """
        
        # Compute ordinary and damped contributions
        E_ordinary = 1.0/distances
        E_damped = 1.0/distances_damped

        # Compute damped reciprocal distances
        chi = (switch_damped*E_damped + switch_ordinary*E_ordinary)
        chi_shift = 2.0/self.cutoff - distances/self.cutoff_squared

        # Compute damped charge-charge electrostatics
        E = atomic_charges_i*atomic_charges_j*(chi - chi_shift)

        # Compute damped charge-dipole and dipole-dipole electrostatics
        if vectors is not None:
            
            # Compute powers of damped reciprocal distances
            chi2 = chi**2
            chi3 = chi2*chi
            chi2_shift = (
                3.0/self.cutoff_squared - 2.0*distances/self.cutoff_cubed)
            chi3_shift = (
                4.0/self.cutoff_cubed - 3.0*distances/self.cutoff_quartic)

            # Adjust atom pair vectors
            chi_vectors = vectors/distances[:, None]

            # Compute dot products of atom pair vector and atomic dipole
            dot_ij = torch.sum(chi_vectors*atomic_dipoles_j, axis=1)
            dot_ji = torch.sum(chi_vectors*atomic_dipoles_i, axis=1)

            # Compute damped charge-dipole electrostatics (times 2 to counter
            # kehalf = ke/2, as charge(i)-dipole(j) != charge(j)-dipole(i))
            E = E + 2.0*atomic_charges_i*dot_ij*(chi2 - chi2_shift)

            # Compute damped dipole-dipole electrostatics
            E = E + (
                torch.sum(atomic_dipoles_i*atomic_dipoles_j, axis=1)
                - 3*dot_ij*dot_ji
                )*(chi3 - chi3_shift)

        # Sum electrostatic contributions
        E = self.kehalf*E

        return E

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
            [kehalf_ase*factor_charge**2/factor_energy/factor_positions],
            device=self.device, dtype=self.dtype)
        self.register_buffer(
            'kehalf', kehalf)

        return

    def forward(
        self,
        properties: Dict[str, torch.Tensor],
        distances: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        vectors: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute shielded electrostatic interaction between atom center point 
        charges and dipoles.

        Parameters
        ----------
        properties: dict
            system properties including atomic charges
        distances : torch.Tensor
            Distances between all atom pairs in the batch.
        idx_i : torch.Tensor
            Indices of the first atom of each pair.
        idx_j : torch.Tensor
            Indices of the second atom of each pair.
        vectors: torch.Tensor, optional, default None
            Connection vectors i->j between all atom pairs in the batch.
            If None, charge-dipole and dipole-dipole interaction are ignored.

        Returns
        -------
        torch.Tensor
            Dispersion atom energy contribution
        
        """

        # Grep atomic charges and dipoles
        atomic_charges = properties.get('atomic_charges')
        atomic_dipoles = properties.get('atomic_dipoles')

        # Gather atomic charge and dipole pairs
        atomic_charges_i = torch.gather(atomic_charges, 0, idx_i)
        atomic_charges_j = torch.gather(atomic_charges, 0, idx_j)
        if atomic_dipoles is None or vectors is None:
            atomic_dipoles_i = None
            atomic_dipoles_j = None
        else:
            atomic_dipoles_i = atomic_dipoles[idx_i]
            atomic_dipoles_j = atomic_dipoles[idx_j]

        # Compute shielded distances
        distances_damped = torch.sqrt(distances**2 + 1.0)

        # Compute switch weights
        switch_damped = self.switch_fn(distances)
        switch_ordinary = 1.0 - switch_damped

        # Compute damped electrostatic Coulomb potential
        E = self.potential_fn(
            atomic_charges_i,
            atomic_charges_j,
            atomic_dipoles_i,
            atomic_dipoles_j,
            distances,
            distances_damped,
            switch_damped,
            switch_ordinary,
            vectors)

        # Apply interaction cutoff
        E = torch.where(
            distances <= self.cutoff,
            E,                      # distance <= cutoff
            torch.zeros_like(E))    # distance > cutoff

        # Sum up electrostatic atom pair contribution of each atom
        return utils.scatter_sum(
             E, idx_i, dim=0, shape=atomic_charges.shape)
