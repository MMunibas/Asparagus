import os
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple, Union

import torch

from asparagus import data
from asparagus import utils
from asparagus import module

__all__ = ['DataLoader']

class DataLoader(torch.utils.data.DataLoader):
    """
    Data loader class from a dataset
    
    Parameters
    ----------
    dataset: (data.DataSet, data.DataSubSet)
        DataSet or DataSubSet instance of reference data
    batch_size: int
        Number of atomic systems per batch
    reference_properties: list
        Reference properties to add to the data batch
    data_shuffle: bool
        Shuffle batch compilation after each epoch
    num_workers: int
        Number of parallel workers for collecting data
    data_collate_fn: callable, optional, default None
        Callable function that prepare and return batch data
    data_pin_memory: bool, optional, default False
        If True data are loaded to GPU
    data_atomic_energies_shift: list(float), optional, default None
        Atom type specific energy shift terms to shift the system energies.
    device: str, optional, default 'cpu'
        Device type for data allocation
    dtype: dtype object, optional, default 'torch.float64'
        Reference data type to convert to

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    def __init__(
        self,
        dataset: Union[data.DataSet, data.DataSubSet],
        batch_size: int,
        reference_properties: List[str],
        data_shuffle: bool,
        num_workers: int,
        device: str,
        dtype: object,
        data_collate_fn: Optional[object] = None,
        data_pin_memory: Optional[bool] = False,
        data_atomic_energies_shift: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize data loader.
        """

        # Check collate function
        if data_collate_fn is None:
            data_collate_fn = self.data_collate_fn

        # Assign class variables
        self.dataset = dataset
        self.batch_size = batch_size
        self.reference_properties = reference_properties

        if num_workers is None:
            self.num_workers = 0
        else:
            self.num_workers = num_workers

        # Initiate DataLoader
        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=data_shuffle,
            num_workers=self.num_workers,
            collate_fn=self.data_collate_fn,
            pin_memory=data_pin_memory,
            **kwargs
        )

        # Initialize neighbor list function class parameter
        self.neighbor_list = None
        self.mlmm_neighbor_list = None

        # Assign reference data conversion parameter
        self.device = device
        self.dtype = dtype

        # Assign atomic energies shift
        self.set_data_atomic_energies_shift(
            data_atomic_energies_shift,
            self.dataset.get_data_properties_dtype())

        return

    def set_data_atomic_energies_shift(
        self,
        atomic_energies_shift: List[float],
        atomic_energies_dtype: 'dtype',
    ):
        """
        Assign atomic energies shift list per atom type
        
        Parameters
        ----------
        atomic_energies_shift: list(float)
            Atom type specific energy shift terms to shift the system energies.
        atomic_energies_dtype: dtype
            Database properties dtype
        """

        if atomic_energies_shift is None:
            self.atomic_energies_shift = None
        else:
            self.atomic_energies_shift = torch.tensor(
                atomic_energies_shift, dtype=atomic_energies_dtype)

        return

    def set_reference_properties(
        self,
        reference_properties: List[str],
    ):
        """
        Update reference properties list.
        
        Parameters
        ----------
        reference_properties: list
            Reference properties to add to the data batch

        """
        
        # Assign new reference properties list
        self.reference_properties = reference_properties
        
        return

    def init_neighbor_list(
        self,
        cutoff: Optional[Union[float, List[float]]] = np.inf,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
    ):
        """
        Initialize neighbor list function

        Parameters
        ----------
        cutoff: float, optional, default infinity
            Neighbor list cutoff range equal to max interaction range

        """

        # Check input parameter
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        # Initialize neighbor list creator
        self.neighbor_list = module.TorchNeighborListRangeSeparated(
            cutoff, device, dtype)

        return

    def init_mlmm_neighbor_list(
        self,
        cutoff: Optional[Union[float, List[float]]] = np.inf,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
    ):
        """
        Initialize MLM/M neighbor list function

        Parameters
        ----------
        cutoff: float, optional, default infinity
            Neighbor list cutoff range equal to max interaction range

        """

        # Check input parameter
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        # Initialize neighbor list creator
        self.mlmm_neighbor_list = module.TorchNeighborListRangeSeparatedMLMM(
            cutoff, device, dtype)

        return

    def data_collate_fn(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch properties from a dataset such as pair indices and
        return with system properties

        Parameters
        ----------
        batch: dict
            Data batch

        Returns
        -------
        dict
            Collated data batch with essential properties, optional structure
            and reference properties.
            Essential Properties:
                atoms_number: (Nsystem,) torch.Tensor
                atomic_numbers: (Natoms,) torch.Tensor
                positions: (Natoms, 3) torch.Tensor
                charge: (Nsystem,) torch.Tensor
                cell: (Nsystem, 3) torch.Tensor
                pbc: (Nsystem, 3,) torch.Tensor
                idx_i: (Npairs,) torch.Tensor
                idx_j: (Npairs,) torch.Tensor
                pbc_offset_ij: (Npairs, 3) torch.Tensor
                idx_u: (Npairs,) torch.Tensor
                idx_v: (Npairs,) torch.Tensor
                pbc_offset_uv: (Npairs, 3) torch.Tensor
                sys_i: (Natoms,) torch.Tensor
            Optional structure properties:
                fragment_numbers: (Natoms,) torch.Tensor
            Reference properties, e.g.:
                energy: (Nsystem,) torch.Tensor
                forces: (Natoms, 3) torch.Tensor
                dipole: (Nsystem,) torch.Tensor


        """

        # Collected batch system properties
        coll_batch = {}

        # Get batch size a.k.a. number of systems
        Nsys = len(batch)

        # Get atoms number per system segment
        coll_batch['atoms_number'] = torch.tensor(
            [b['atoms_number'] for b in batch],
            device=self.device, dtype=torch.int64)

        # System segment index
        coll_batch['sys_i'] = torch.repeat_interleave(
            torch.arange(Nsys, device=self.device, dtype=torch.int64),
            repeats=coll_batch['atoms_number'], dim=0).to(
                device=self.device, dtype=torch.int64)

        # Atomic numbers properties
        coll_batch['atomic_numbers'] = torch.cat(
            [b['atomic_numbers'] for b in batch], 0).to(
                device=self.device, dtype=torch.int64)

        # Atom positions
        coll_batch['positions'] = torch.cat(
            [b['positions'] for b in batch], 0).to(
                device=self.device, dtype=self.dtype)

        # System charge
        coll_batch['charge'] = torch.cat(
            [b['charge'] for b in batch], 0).to(
                device=self.device, dtype=self.dtype)

        # Periodic boundary conditions
        coll_batch['pbc'] = torch.cat(
            [b['pbc'] for b in batch], 0).to(
                device=self.device, dtype=torch.bool
                ).reshape(Nsys, 3)

        # Unit cell sizes
        coll_batch['cell'] = torch.cat(
            [b['cell'] for b in batch], 0).to(
                device=self.device, dtype=self.dtype
                ).reshape(Nsys, -1)

        # System atomic fragment indices
        # Only add to batch, if multiple fragments are defined
        if 'fragment_numbers' in batch[0]:
            fragment_numbers = torch.cat(
                [b['fragment_numbers'] for b in batch], 0).to(
                    device=self.device, dtype=torch.int64)
            if torch.unique(fragment_numbers).shape[0] > 1:
                coll_batch['fragment_numbers'] = fragment_numbers

            # Compute ML/MM pair indices and position offsets
            if self.mlmm_neighbor_list is None:
                self.init_mlmm_neighbor_list()
            coll_batch = self.mlmm_neighbor_list(coll_batch)

        # Compute pair indices and position offsets
        if self.neighbor_list is None:
            self.init_neighbor_list()
        coll_batch = self.neighbor_list(coll_batch)

        # Collect reference properties
        coll_batch['reference'] = {}
        for prop in self.reference_properties:

            # Apply energy shift for property 'energy' and 'atomic_energies'
            # if defined
            if self.atomic_energies_shift is not None and prop == 'energy':

                coll_batch['reference'][prop] = torch.tensor(
                    [
                        b[prop]
                        - torch.sum(
                            self.atomic_energies_shift[b['atomic_numbers']]
                        )
                        for b in batch
                    ],
                    device=self.device,
                    dtype=self.dtype
                )
        
            elif (
                self.atomic_energies_shift is not None
                and prop == 'atomic_energies'
            ):

                coll_batch['reference'][prop] = torch.cat(
                    [
                        b[prop]
                        - self.atomic_energies_shift[b['atomic_numbers']]
                        for b in batch
                    ],
                    0).to(device=self.device, dtype=self.dtype)

            else:

                # Concatenate tensor data
                if batch[0][prop].size():
                    coll_batch['reference'][prop] = torch.cat(
                        [b[prop] for b in batch], 0).to(
                            device=self.device, dtype=self.dtype)

                # Concatenate numeric data
                else:
                    coll_batch['reference'][prop] = torch.tensor(
                        [b[prop] for b in batch],
                        device=self.device, dtype=self.dtype)

        return coll_batch

    @property
    def data_properties(self):
        return self.dataset.data_properties

