import os
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list

import torch

from .. import utils

__all__ = ["TorchNeighborListRangeSeparated"]

class TorchNeighborListRangeSeparated(torch.nn.Module):
    """
    Environment provider making use of neighbor lists as implemented in
    TorchAni. 
    Modified to provide neighbor lists for a set of cutoff radii.

    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    References:
        https://github.com/aiqm/torchani/blob/master/torchani/aev.py
    """

    def __init__(
        self,
        cutoff: List[float],
        device: str,
        dtype: object,
    ):
        """
        Parameters
        ----------
        cutoff: list(float)
            List of Cutoff distances

        """
        
        super().__init__()
        
        # Assign module variable parameters
        self.device = device
        self.dtype = dtype
        
        # Check cutoffs
        if utils.is_numeric(cutoff):
            self.cutoff = torch.tensor(
                [cutoff], device=self.device, dtype=self.dtype)
        else:
            self.cutoff = torch.tensor(
                cutoff, device=self.device, dtype=self.dtype)
        self.max_cutoff = torch.max(self.cutoff)

        return

    def forward(
        self,
        coll_batch: Dict[str, torch.Tensor],
        atomic_numbers_cumsum: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Build neighbor list for a batch of systems.
        Parameters
        ----------
        coll_batch: dict
            System property batch
        atomic_numbers_cumsum: torch.Tensor, optional, default None
            Cumulative atomic number sum serving as starting index for atom
            length system data lists.

        Returns
        -------
        dict(str, torch.Tensor)
            Updated system batch with atom pair information

        """

        # Extract system data
        atomic_numbers = coll_batch['atomic_numbers']
        positions = coll_batch['positions']
        cell = coll_batch['cell']
        pbc = coll_batch['pbc']

        # Check system indices
        if coll_batch.get("sys_i") is None:
            sys_i = torch.zeros_like(atomic_numbers)
        else:
            sys_i = coll_batch["sys_i"]
        
        # Check for system batch or single system input
        # System batch:
        if coll_batch["atoms_number"].dim():
            
            # Compute, eventually, cumulative atomic number list
            if atomic_numbers_cumsum is None:
                atomic_numbers_cumsum = torch.cat(
                    [
                        torch.zeros((1,), dtype=sys_i.dtype),
                        torch.cumsum(coll_batch["atoms_number"][:-1], dim=0)
                    ],
                    dim=0)

        # Single system
        else:
            
            # Assign cumulative atomic number list and system index
            atomic_numbers_cumsum = torch.zeros((1,), dtype=sys_i.dtype)
            sys_i = torch.zeros_like(atomic_numbers)
            
            # Extend periodic system data
            cell = cell[None, ...]
            pbc = pbc[None, ...]

        # Compute atom pair neighbor list
        idcs_i, idcs_j, pbc_offsets = self._build_neighbor_list(
            self.cutoff,
            atomic_numbers, 
            positions, 
            cell, 
            pbc,
            sys_i, 
            atomic_numbers_cumsum)

        # Add neighbor lists to batch data
        # 1: Neighbor list of first cutoff (usually short range)
        coll_batch['idx_i'] = idcs_i[0].detach()
        coll_batch['idx_j'] = idcs_j[0].detach()
        if pbc_offsets is not None:
            coll_batch['pbc_offset_ij'] = pbc_offsets[0].detach()
        # 2: If demanded, neighbor list of second cutoff (usually long range)
        if len(idcs_i) > 1:
            coll_batch['idx_u'] = idcs_i[1].detach()
            coll_batch['idx_v'] = idcs_j[1].detach()
            if pbc_offsets is not None:
                coll_batch['pbc_offset_uv'] = pbc_offsets[1].detach()
        # 3+: If demanded, list of neighbor lists of further cutoffs
        if len(idcs_i) > 2:
            coll_batch['idcs_k'] = [idx_i.detach() for idx_i in idcs_i]
            coll_batch['idcs_l'] = [idx_j.detach() for idx_j in idcs_j]
            if pbc_offsets is not None:
                coll_batch['pbc_offsets_l'] = [
                    pbc_offset.detach() for pbc_offset in pbc_offsets]

        return coll_batch

    def _build_neighbor_list(
        self,
        cutoff: List[float],
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        sys_i: torch.Tensor,
        atomic_numbers_cumsum: torch.Tensor,
    ) -> (List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]):

        # Initialize result lists
        idcs_i = [[] for _ in cutoff]
        idcs_j = [[] for _ in cutoff]
        offsets = [[] for _ in cutoff]

        # Iterate over system segments
        for iseg, idx_off in enumerate(atomic_numbers_cumsum):

            # Atom system selection
            select = sys_i == iseg

            # Check if shifts are needed for periodic boundary conditions
            if cell[iseg].dim() == 1:
                if cell[iseg].shape[0] == 3:
                    cell_seg = cell[iseg].diag()
                else:
                    cell_seg = cell[iseg].reshape(3,3)
            else:
                cell_seg = cell[iseg]

            if torch.any(pbc[iseg]):
                seg_offsets = self._get_shifts(
                    cell_seg, pbc[iseg], self.max_cutoff)
            else:
                seg_offsets = torch.zeros(
                    0, 3, device=positions.device, dtype=positions.dtype)

            # Compute pair indices
            sys_idcs_i, sys_idcs_j, seg_offsets = self._get_neighbor_pairs(
                positions[select], cell_seg, seg_offsets, cutoff)

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
                #syss_ij[ic].append(torch.full_like(sys_idx_i, iseg))

        idcs_i = [
            torch.cat(idx_i, dim=0).to(dtype=atomic_numbers.dtype)
            for idx_i in idcs_i]
        idcs_j = [
            torch.cat(idx_j, dim=0).to(dtype=atomic_numbers.dtype)
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
        cutoff: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Compute pairs of atoms that are neighbors.

        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Arguments:
            positions (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the
                three vectors defining unit cell:
                tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing
                shifts
            cutoff (:class:`torch.Tensor`): tensor of shape (?) storing
                cutoff radii
        """

        num_atoms = positions.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # 1) Central cell
        pi_center, pj_center = torch.combinations(all_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, all_atoms, all_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute distances, and find all pairs within cutoff
        # torch.norm(Rij_all, dim=1)
        distances2 = torch.sum(Rij_all**2, dim=1)
        in_cutoffs = [
            torch.nonzero(distances2 < cutoff_i**2, as_tuple=True)
            for cutoff_i in cutoff]

        # 6) Reduce tensors to relevant components
        atom_indices_i, atom_indices_j, offsets = [], [], []
        for in_cutoff in in_cutoffs:
            pair_index = in_cutoff
            atom_indices_i.append(pi_all[pair_index])
            atom_indices_j.append(pj_all[pair_index])
            offsets.append(shifts_all[pair_index])

        return atom_indices_i, atom_indices_j, offsets

    def _get_shifts(
        self, 
        cell, 
        pbc, 
        cutoff
    ) -> torch.Tensor:
        """
        Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.

        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3)
                of the three vectors defining unit cell:
                    tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.
            cutoff (:class:`torch.Tensor`): tensor of shape (1) storing
                cutoff radius

        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)
        
        num_repeats = torch.ceil(cutoff*inverse_lengths).to(cell.dtype)
        num_repeats = torch.where(
            pbc.flatten(),
            num_repeats,
            torch.Tensor([0], device=cell.device).to(cell.dtype)
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
