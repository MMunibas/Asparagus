from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from asparagus import utils

__all__ = [
    "TorchNeighborListRangeSeparated",
    "TorchNeighborListRangeSeparatedMLMM",
    ]

class TorchNeighborListRangeSeparated(torch.nn.Module):
    """
    Environment provider making use of neighbor lists as implemented in
    TorchAni. Modified to provide neighbor lists for a set of cutoff radii.

    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    References:
        https://github.com/aiqm/torchani/blob/master/torchani/aev.py

    Parameters
    ----------
    cutoffs: list(float)
        List of cutoff distances
    fragment: int, optional, default 0
        Atomic fragment number for which atom pairs are computed

    """

    def __init__(
        self,
        cutoffs: List[float],
        device: str,
        dtype: object,
        fragment: int = 0,
    ):
        """
        Initialize neighbor list computation class

        """

        super().__init__()

        # Assign module variable parameters
        self.device = device
        self.dtype = dtype
        self.fragment = fragment

        # Check and set cutoffs
        self.set_cutoffs(cutoffs)

        return

    def set_cutoffs(
        self,
        cutoffs: List[float],
    ):
        """
        Set cutoff values,
        
        Parameters
        ----------
        cutoffs: list(float)
            List of cutoff distances

        """
        
        # Check and set cutoffs
        if utils.is_torch_tensor(cutoffs):
            if cutoffs.dim():
                self.cutoffs = cutoffs.clone().detach().to(
                    device=self.device, dtype=self.dtype)
            else:
                self.cutoffs = cutoffs.clone().detach().unsqueeze(0).to(
                    device=self.device, dtype=self.dtype)
        elif utils.is_numeric(cutoffs):
            self.cutoffs = torch.tensor(
                [cutoffs], device=self.device, dtype=self.dtype)
        else:
            self.cutoffs = torch.tensor(
                cutoffs, device=self.device, dtype=self.dtype)

        self.max_cutoff = torch.max(self.cutoffs)
        
        # Check max cutoff and, if infinite, disable periodic boundary 
        # conditions forcefully
        if torch.isinf(self.max_cutoff):
            self.no_pbc = True
        else:
            self.no_pbc = False
        
        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Build neighbor list for a batch of systems.

        Parameters
        ----------
        batch: dict
            System property batch

        Returns
        -------
        dict(str, torch.Tensor)
            Updated system batch with atom pair information

        """

        # Assign batch data
        atoms_number = batch["atoms_number"]
        positions = batch['positions']
        cell = batch['cell']
        pbc = batch['pbc']

        # Check system indices
        if 'sys_i' in batch:
            sys_i = batch['sys_i']
        else:
            sys_i = torch.zeros(
                (positions.shape[0], ),
                device=self.device,
                dtype=atoms_number.dtype)

        # Check for system batch or single system input
        # System batch:
        if atoms_number.dim():

            # Compute cumulative atoms number list
            atoms_numbers_cumsum = torch.cat(
                [
                    torch.zeros(
                        (1, ), device=self.device, dtype=sys_i.dtype),
                    torch.cumsum(atoms_number[:-1], dim=0)
                ],
                dim=0)
            
        # Single system
        else:

            # Assign cumulative atoms number list and system index
            atoms_numbers_cumsum = torch.zeros(
                (1, ), device=self.device, dtype=sys_i.dtype)

            # Extend periodic system data size
            cell = cell.unsqueeze(0)
            pbc = pbc.unsqueeze(0)

        # Disable periodic boundary conditions for infinite cutoffs
        if self.no_pbc:
            pbc[:] = False

        # Check fragment indices and if fragments are defined get selected
        # fragment mask
        if 'fragment_numbers' in batch:
            
            fragment_numbers = batch['fragment_numbers']
            
            # If multiple fragments are defined, create index pointer from
            # full system to fragment system (e.g. atom with index 42 in the
            # full system has only index 2 in the fragment subsystem).
            if torch.unique(fragment_numbers).shape[0] > 1:
                ml_idx = (
                    torch.arange(
                        fragment_numbers.shape[0],
                        device=self.device,
                        dtype=fragment_numbers.dtype)
                    )[fragment_numbers == self.fragment]
                ml_idx_p = torch.full_like(
                    fragment_numbers,
                    -1,
                    device=self.device,
                    dtype=fragment_numbers.dtype)
                for ia, ai in enumerate(ml_idx):
                    ml_idx_p[ai] = ia
                batch['ml_idx'] = ml_idx.detach()
                batch['ml_idx_p'] = ml_idx_p.detach()

        else:
            
            fragment_numbers = torch.full_like(sys_i, self.fragment)

        # Compute atom pair neighbor list
        idcs_i, idcs_j, pbc_offsets = self._build_neighbor_list(
            self.cutoffs,
            positions,
            cell,
            pbc,
            sys_i,
            fragment_numbers,
            atoms_numbers_cumsum)

        # Add neighbor lists to batch data
        # 1: Neighbor list of first cutoff (usually short range)
        batch['idx_i'] = idcs_i[0].detach()
        batch['idx_j'] = idcs_j[0].detach()
        batch['pbc_offset_ij'] = pbc_offsets[0].detach()
        # 2: If demanded, neighbor list of second cutoff (usually long range)
        if len(self.cutoffs) > 1:
            batch['idx_u'] = idcs_i[1].detach()
            batch['idx_v'] = idcs_j[1].detach()
            batch['pbc_offset_uv'] = pbc_offsets[1].detach()
        # 3+: If demanded, list of neighbor lists of further cutoffs
        if len(self.cutoffs) > 2:
            for icut, (idx_i, idx_j) in enumerate(zip(idcs_i[2:], idcs_j[2:])):
                batch['idcs_k:{:d}'.format(icut + 2)] = idx_i.detach()
                batch['idcs_l:{:d}'.format(icut + 2)] = idx_j.detach()
                batch['pbc_offsets_kl:{:d}'.format(icut + 2)] = (
                    pbc_offsets[icut].detach())

        return batch

    def _build_neighbor_list(
        self,
        cutoffs: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        sys_i: torch.Tensor,
        fragment_numbers: torch.Tensor,
        atoms_numbers_cumsum: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Execute neighbor list generation

        Parameters
        ----------
        cutoffs: torch.Tensor
            Cutoff ranges
        positions: torch.Tensor
            Atom positions
        cell: torch.Tensor
            Systm cell parameter
        pbc: torch.Tensor
            Systm periodic boundary conditions
        fragment_numbers: torch.Tensor
            Atomic fragment indices of the atoms in the positions list to 
            build neighbor list only within a defined fragment atoms.
        atoms_numbers_cumsum: torch.Tensor
            Cumulative atoms number sum serving as starting index for atom
            length system data lists.

        Returns
        -------
        torch.Tensor
            List of atom pair indices i for every cutoff distance
        torch.Tensor
            List of atom pair indices j for every cutoff distance
        torch.Tensor
            List of atom pair position offsets

        """
        # Initialize result lists
        idcs_i = [[] for _ in cutoffs]
        idcs_j = [[] for _ in cutoffs]
        offsets = [[] for _ in cutoffs]

        # Iterate over system segments
        for iseg, idx_off in enumerate(atoms_numbers_cumsum):

            # Atom system selection
            select = sys_i == iseg

            # Check if shifts are needed for periodic boundary conditions
            if cell[iseg].dim() == 1:
                if cell[iseg].shape[0] == 3:
                    cell_seg = cell[iseg].diag()
                else:
                    cell_seg = cell[iseg].reshape(3, 3)
            else:
                cell_seg = cell[iseg]

            if torch.any(pbc[iseg]):
                seg_offsets = self._get_shifts(
                    cell_seg, pbc[iseg], self.max_cutoff)
            else:
                seg_offsets = torch.zeros(
                    0, 3, device=self.device, dtype=positions.dtype)

            # Compute pair indices
            sys_idcs_i, sys_idcs_j, seg_offsets = self._get_neighbor_pairs(
                positions[select],
                cell_seg,
                seg_offsets,
                cutoffs,
                fragment_numbers[select])

            # Create bidirectional id arrays, similar to what the ASE
            # neighbor list returns
            bi_idcs_i = [
                torch.cat((sys_idx_i, sys_idx_j), dim=0)
                for sys_idx_i, sys_idx_j in zip(sys_idcs_i, sys_idcs_j)]
            bi_idcs_j = [
                torch.cat((sys_idx_j, sys_idx_i), dim=0)
                for sys_idx_j, sys_idx_i in zip(sys_idcs_j, sys_idcs_i)]

            # Sort along first dimension (necessary for atom-wise pooling)
            for ic, (bi_idx_i, bi_idx_j, seg_offset) in enumerate(
                zip(bi_idcs_i, bi_idcs_j, seg_offsets)
            ):
                sorted_idx = torch.argsort(bi_idx_i)
                sys_idx_i = bi_idx_i[sorted_idx]
                sys_idx_j = bi_idx_j[sorted_idx]

                bi_offset = torch.cat((-seg_offset, seg_offset), dim=0)
                seg_offset = bi_offset[sorted_idx]
                seg_offset = torch.mm(seg_offset.to(cell.dtype), cell_seg)

                # Append pair indices and position offsets
                idcs_i[ic].append(sys_idx_i + idx_off)
                idcs_j[ic].append(sys_idx_j + idx_off)
                offsets[ic].append(seg_offset)

        idcs_i = [
            torch.cat(idx_i, dim=0).to(dtype=sys_i.dtype)
            for idx_i in idcs_i]
        idcs_j = [
            torch.cat(idx_j, dim=0).to(dtype=sys_i.dtype)
            for idx_j in idcs_j]
        offsets = [
            torch.cat(offset, dim=0).to(dtype=positions.dtype)
            for offset in offsets]

        return idcs_i, idcs_j, offsets

    def _get_neighbor_pairs(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        shifts: torch.Tensor,
        cutoffs: torch.Tensor,
        fragment_numbers: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute pairs of atoms that are neighbors.

        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Parameters
        ----------
        positions: torch.Tensor
            Atom positions
        cell: torch.Tensor
            System cell parameter
        shifts: torch.Tensor
            System cell shift parameter
        cutoffs: torch.Tensor
            Cutoff ranges
        fragment_numbers: torch.Tensor
            Atomic fragment indices

        Returns
        -------
        torch.Tensor
            List of atom pair indices i for every cutoff distance
        torch.Tensor
            List of atom pair indices j for every cutoff distance
        torch.Tensor
            List of atom pair position offsets

        """

        num_atoms = positions.shape[0]
        fragment_atoms = (
            torch.arange(num_atoms, device=cell.device
                )[fragment_numbers == self.fragment]
        )

        # 1) Central cell
        pi_center, pj_center = torch.combinations(fragment_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, fragment_atoms, fragment_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute distances, and find all pairs within cutoffs
        # torch.norm(Rij_all, dim=1)
        distances2 = torch.sum(Rij_all**2, dim=1)
        in_cutoffs = [
            torch.nonzero(distances2 < cutoff_i**2).flatten()
            for cutoff_i in cutoffs]

        # 6) Reduce tensors to relevant components
        atom_indices_i, atom_indices_j, offsets = [], [], []
        for in_cutoff_i in in_cutoffs:
            atom_indices_i.append(pi_all[in_cutoff_i])
            atom_indices_j.append(pj_all[in_cutoff_i])
            offsets.append(shifts_all[in_cutoff_i])

        return atom_indices_i, atom_indices_j, offsets

    def _get_shifts(
        self,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        max_cutoff: float,
    ) -> torch.Tensor:
        """
        Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.

        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Parameters
        ----------
        cell: torch.Tensor
            Systm cell parameter
        pbc: torch.Tensor
            Systm periodic boundary conditions
        max_cutoff: float
            Maximum cutoff range
        ml_atom_indices: torch.Tensor
            Indices of the ML atoms in the positions list

        Returns
        -------
        torch.Tensor 
            Tensor of shifts. the center cell and symmetric cells are not
            included.

        """

        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)

        num_repeats = torch.ceil(max_cutoff*inverse_lengths).to(cell.dtype)
        num_repeats = torch.where(
            pbc.flatten(),
            num_repeats,
            torch.tensor([0.0], device=cell.device, dtype=cell.dtype)
        )

        r1 = torch.arange(
            1, num_repeats[0] + 1, dtype=cell.dtype, device=cell.device)
        r2 = torch.arange(
            1, num_repeats[1] + 1, dtype=cell.dtype, device=cell.device)
        r3 = torch.arange(
            1, num_repeats[2] + 1, dtype=cell.dtype, device=cell.device)
        o = torch.zeros(1, dtype=cell.dtype, device=cell.device)

        return torch.cat(
            [
                torch.cartesian_prod(r1, r2, r3),
                torch.cartesian_prod(r1, r2, o),
                torch.cartesian_prod(r1, r2, -r3),
                torch.cartesian_prod(r1, o, r3),
                torch.cartesian_prod(r1, o, o),
                torch.cartesian_prod(r1, o, -r3),
                torch.cartesian_prod(r1, -r2, r3),
                torch.cartesian_prod(r1, -r2, o),
                torch.cartesian_prod(r1, -r2, -r3),
                torch.cartesian_prod(o, r2, r3),
                torch.cartesian_prod(o, r2, o),
                torch.cartesian_prod(o, r2, -r3),
                torch.cartesian_prod(o, o, r3),
            ]
        )


class TorchNeighborListRangeSeparatedMLMM(torch.nn.Module):
    """
    Environment provider making use of neighbor lists between two sets of atom
    positions adopted from the TorchAni implementation.
    Modified to provide neighbor lists for a set of cutoff radii.

    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    References:
        https://github.com/aiqm/torchani/blob/master/torchani/aev.py

    Parameters
    ----------
    cutoffs: list(float)
        List of Cutoff distances
    device: str, optional, default global setting
        Device type for model variable allocation
    dtype: dtype object, optional, default global setting
        Model variables data type

    """

    def __init__(
        self,
        cutoffs: Union[float, List[float]],
        device: str,
        dtype: object,
        ml_fragment: int = 0,
        mm_fragment: int = 1,
    ):
        """
        Initialize neighbor list computation class
        """

        super().__init__()

        # Assign module variable parameters
        self.device = device
        self.dtype = dtype
        self.ml_fragment = ml_fragment
        self.mm_fragment = mm_fragment

        # Check and set cutoffs
        self.set_cutoffs(cutoffs)

        return

    def set_cutoffs(
        self,
        cutoffs: List[float],
    ):
        """
        Set cutoff values,
        
        Parameters
        ----------
        cutoffs: list(float)
            List of cutoff distances

        """
        
        # Check and set cutoffs
        if utils.is_numeric(cutoffs):
            self.cutoffs = torch.tensor(
                [cutoffs], device=self.device, dtype=self.dtype)
        else:
            self.cutoffs = torch.tensor(
                cutoffs, device=self.device, dtype=self.dtype)

        self.max_cutoff = torch.max(self.cutoffs)

        # Check max cutoff and, if infinite, disable periodic boundary 
        # conditions forcefully
        if torch.isinf(self.max_cutoff):
            self.no_pbc = True
        else:
            self.no_pbc = False

        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Generate neighbor list for a batch of systems.

        Parameters
        ----------
        batch: dict
            System property batch

        Returns
        -------
        dict(str, torch.Tensor)
            Updated system batch with atom pair information

        """

        # Assign batch data
        atoms_number = batch["atoms_number"]
        positions = batch['positions']
        cell = batch['cell']
        pbc = batch['pbc']

        # Check ML and MM system indices
        if 'sys_i' in batch:
            sys_i = batch['sys_i']
        else:
            sys_i = torch.zeros(
                (positions.shape[0], ),
                device=self.device,
                dtype=atoms_number.dtype)

        # Check for system batch or single system input
        # System batch:
        if atoms_number.dim():

            # Compute cumulative atoms number list
            atoms_numbers_cumsum = torch.cat(
                [
                    torch.zeros(
                        (1,), device=self.device, dtype=sys_i.dtype),
                    torch.cumsum(atoms_number[:-1], dim=0)
                ],
                dim=0)

        # Single system
        else:

            # Assign cumulative atoms number list and system index
            atoms_numbers_cumsum = torch.zeros(
                (1,), device=self.device, dtype=sys_i.dtype)

            # Extend periodic system data
            cell = cell.unsqueeze(0)
            pbc = pbc.unsqueeze(0)

        # Disable periodic boundary conditions for infinite cutoffs
        if self.no_pbc:
            pbc[:] = False

        # Check fragment indices
        if 'fragment_numbers' in batch:

            fragment_numbers = batch['fragment_numbers']

            # If multiple fragments are defined, create index pointer from
            # full system to MM fragment system
            if torch.unique(fragment_numbers).shape[0] > 1:
                mm_idx = (
                    torch.arange(
                        fragment_numbers.shape[0],
                        device=self.device,
                        dtype=fragment_numbers.dtype)
                    )[fragment_numbers == self.mm_fragment]
                batch['mm_idx'] = mm_idx.detach()

        else:
            fragment_numbers = torch.full_like(sys_i, self.ml_fragment)

        # Compute atom pair neighbor list
        idcs_i, idcs_j, pbc_offsets = self._build_neighbor_list(
            self.cutoffs,
            positions,
            cell,
            pbc,
            sys_i,
            fragment_numbers,
            atoms_numbers_cumsum)

        # Add neighbor lists to batch data
        # 1: Neighbor list of first cutoff (usually short range)
        batch['mlmm_idx_i'] = idcs_i[0].detach()
        batch['mlmm_idx_j'] = idcs_j[0].detach()
        batch['mlmm_pbc_offset_ij'] = pbc_offsets[0].detach()
        # 2: If demanded, neighbor list of second cutoff (usually long range)
        if len(self.cutoffs) > 1:
            batch['mlmm_idx_u'] = idcs_i[1].detach()
            batch['mlmm_idx_v'] = idcs_j[1].detach()
            batch['mlmm_pbc_offset_uv'] = pbc_offsets[1].detach()
        # 3+: If demanded, list of neighbor lists of further cutoffs
        if len(self.cutoffs) > 2:
            for icut, (idx_i, idx_j) in enumerate(zip(idcs_i[2:], idcs_j[2:])):
                batch['idcs_k:{:d}'.format(icut + 2)] = idx_i.detach()
                batch['idcs_l:{:d}'.format(icut + 2)] = idx_j.detach()
                batch['pbc_offsets_kl:{:d}'.format(icut + 2)] = (
                    pbc_offsets[icut].detach())

        return batch

    def _build_neighbor_list(
        self,
        cutoffs: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        sys_i: torch.Tensor,
        fragment_numbers: torch.Tensor,
        atoms_numbers_cumsum: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Execute neighbor list generation

        Parameters
        ----------
        cutoffs: torch.Tensor
            Cutoff ranges
        positions: torch.Tensor
            Atom positions
        cell: torch.Tensor
            Systm cell parameter
        pbc: torch.Tensor
            Systm periodic boundary conditions
        sys_i: torch.Tensor
            System atom indices
        fragment_numbers: torch.Tensor
            Atomic fragment indices of the atoms in the positions list to 
            build neighbor list between two sets of fragment atoms.
        atoms_numbers_cumsum: torch.Tensor
            Cumulative atoms number sum serving as starting index for atom
            length system data lists.

        Returns
        -------
        torch.Tensor
            List of atom pair indices i for every cutoff distance
        torch.Tensor
            List of atom pair indices j for every cutoff distance
        torch.Tensor
            List of atom pair position offsets

        """

        # Initialize result lists
        idcs_i = [[] for _ in cutoffs]
        idcs_j = [[] for _ in cutoffs]
        offsets = [[] for _ in cutoffs]

        # Iterate over system segments
        for iseg, idx_off in enumerate(atoms_numbers_cumsum):

            # Atom system selection
            select = sys_i == iseg

            # Check if shifts are needed for periodic boundary conditions
            if cell[iseg].dim() == 1:
                if cell[iseg].shape[0] == 3:
                    cell_seg = cell[iseg].diag()
                else:
                    cell_seg = cell[iseg].reshape(3, 3)
            else:
                cell_seg = cell[iseg]

            if torch.any(pbc[iseg]):
                seg_offsets = self._get_shifts(
                    cell_seg, pbc[iseg], self.max_cutoff)
            else:
                seg_offsets = torch.zeros(
                    0, 3, device=self.device, dtype=positions.dtype)

            # Compute pair indices
            sys_idcs_i, sys_idcs_j, seg_offsets = self._get_neighbor_pairs(
                positions[select],
                cell_seg,
                seg_offsets,
                cutoffs,
                fragment_numbers[select])

            # Sort along first dimension (necessary for atom-wise pooling)
            for ic, (sys_idx_i, sys_idx_j, seg_offset) in enumerate(
                zip(sys_idcs_i, sys_idcs_j, seg_offsets)
            ):
                sorted_idx = torch.argsort(sys_idx_i)
                sys_idx_i = sys_idx_i[sorted_idx]
                sys_idx_j = sys_idx_j[sorted_idx]

                bi_offset = torch.cat((-seg_offset, seg_offset), dim=0)
                seg_offset = bi_offset[sorted_idx]
                seg_offset = torch.mm(seg_offset.to(cell.dtype), cell_seg)

                # Append pair indices and position offsets
                idcs_i[ic].append(sys_idx_i + idx_off)
                idcs_j[ic].append(sys_idx_j + idx_off)
                offsets[ic].append(seg_offset)

        idcs_i = [
            torch.cat(idx_i, dim=0).to(dtype=sys_i.dtype)
            for idx_i in idcs_i]
        idcs_j = [
            torch.cat(idx_j, dim=0).to(dtype=sys_i.dtype)
            for idx_j in idcs_j]
        offsets = [
            torch.cat(offset, dim=0).to(dtype=positions.dtype)
            for offset in offsets]

        return idcs_i, idcs_j, offsets

    def _get_neighbor_pairs(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        shifts: torch.Tensor,
        cutoffs: torch.Tensor,
        fragment_numbers: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute pairs of atoms that are neighbors.

        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Parameters
        ----------
        positions: torch.Tensor
            Atom positions
        cell: torch.Tensor
            System cell parameter
        shifts: torch.Tensor
            System cell shift parameter
        cutoffs: torch.Tensor
            Cutoff ranges
        fragment_numbers: torch.Tensor
            Atomic fragment indices

        Returns
        -------
        torch.Tensor
            List of atom pair indices i for every cutoff distance
        torch.Tensor
            List of atom pair indices j for every cutoff distance
        torch.Tensor
            List of atom pair position offsets

        """

        # Get ML and MM atom information
        num_atoms = fragment_numbers.shape[0]
        ml_fragment_atoms = (
            torch.arange(num_atoms, device=cell.device
                )[fragment_numbers == self.ml_fragment]
            )
        mm_fragment_atoms = (
            torch.arange(num_atoms, device=cell.device
                )[fragment_numbers == self.mm_fragment]
            )

        # Return empty list if either ML or MM atoms are missing
        if not ml_fragment_atoms.dim() or not mm_fragment_atoms.dim():
            atom_indices_i, atom_indices_j, offsets = [], [], []
            for _ in cutoffs:
                atom_indices_i.append(
                    torch.empty(0, device=cell.device, dtype=torch.int64))
                atom_indices_j.append(
                    torch.empty(0, device=cell.device, dtype=torch.int64))
                offsets.append(
                    torch.empty((0, 3,), device=cell.device, dtype=cell.dtype))
            return atom_indices_i, atom_indices_j, offsets

        # 1) Central cell
        pi_center, pj_center = torch.cartesian_prod(
            ml_fragment_atoms, mm_fragment_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, ml_fragment_atoms, mm_fragment_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute squared distances, and find all pairs within cutoff
        distances2 = torch.sum(Rij_all**2, dim=1)
        in_cutoffs = [
            torch.nonzero(distances2 < cutoff_i**2).flatten()
            for cutoff_i in cutoffs]

        # 6) Reduce tensors to relevant components
        atom_indices_i, atom_indices_j, offsets = [], [], []
        for in_cutoff_i in in_cutoffs:
            atom_indices_i.append(pi_all[in_cutoff_i])
            atom_indices_j.append(pj_all[in_cutoff_i])
            offsets.append(shifts_all[in_cutoff_i])

        return atom_indices_i, atom_indices_j, offsets

    def _get_shifts(
        self,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        max_cutoff: float,
    ) -> torch.Tensor:
        """
        Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.

        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Parameters
        ----------
        cell: torch.Tensor
            Systm cell parameter
        pbc: torch.Tensor
            Systm periodic boundary conditions
        max_cutoff: float
            Maximum cutoff range

        Returns
        -------
        torch.Tensor 
            Tensor of shifts. the center cell and symmetric cells are not
            included.

        """

        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)

        num_repeats = torch.ceil(max_cutoff*inverse_lengths).to(cell.dtype)
        num_repeats = torch.where(
            pbc.flatten(),
            num_repeats,
            torch.tensor([0.0], device=cell.device, dtype=cell.dtype)
        )

        r1 = torch.arange(
            1, num_repeats[0] + 1, dtype=cell.dtype, device=cell.device)
        r2 = torch.arange(
            1, num_repeats[1] + 1, dtype=cell.dtype, device=cell.device)
        r3 = torch.arange(
            1, num_repeats[2] + 1, dtype=cell.dtype, device=cell.device)
        o = torch.zeros(1, dtype=cell.dtype, device=cell.device)

        return torch.cat(
            [
                torch.cartesian_prod(r1, r2, r3),
                torch.cartesian_prod(r1, r2, o),
                torch.cartesian_prod(r1, r2, -r3),
                torch.cartesian_prod(r1, o, r3),
                torch.cartesian_prod(r1, o, o),
                torch.cartesian_prod(r1, o, -r3),
                torch.cartesian_prod(r1, -r2, r3),
                torch.cartesian_prod(r1, -r2, o),
                torch.cartesian_prod(r1, -r2, -r3),
                torch.cartesian_prod(o, r2, r3),
                torch.cartesian_prod(o, r2, o),
                torch.cartesian_prod(o, r2, -r3),
                torch.cartesian_prod(o, o, r3),
            ]
        )
