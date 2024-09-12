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

        # Assign reference dataset as class parameter for additions by the
        # neighbor list functions
        self.dataset = dataset
        self.batch_size = batch_size

        #if device == 'cpu':
            #self.data_num_workers = 0
        #else:
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

        # Assign reference data conversion parameter
        #self.device = utils.check_device_option(device, config)
        #self.dtype = utils.check_dtype_option(dtype, config)
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

    def init_neighbor_list(
        self,
        cutoff: Optional[Union[float, List[float]]] = np.inf,
        store: Optional[bool] = False,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
    ):
        """
        Initialize neighbor list function

        Parameters
        ----------
        cutoff: float, optional, default infinity
            Neighbor list cutoff range equal to max interaction range
        store: bool, optional, default False
            Pre-compute neighbor list and store in the reference dataset

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
            Collated data batch with additional properties
            Properties:
                atoms_number: (Natoms,) torch.Tensor
                atomic_numbers: (Natoms,) torch.Tensor
                positions: (Natoms, 3) torch.Tensor
                forces: (Natoms, 3) torch.Tensor
                energy: (Natoms,) torch.Tensor
                charge: (Natoms,) torch.Tensor
                idx_i: (Npairs,) torch.Tensor
                idx_j: (Npairs,) torch.Tensor
                pbc_offset_ij: (Npairs, 3) torch.Tensor
                idx_u: (Npairs,) torch.Tensor
                idx_v: (Npairs,) torch.Tensor
                pbc_offset_uv: (Npairs, 3) torch.Tensor
                sys_i: (Natoms,) torch.Tensor

        """

        # Collected batch system properties
        coll_batch = {}

        # Get batch size a.k.a. number of systems
        Nsys = len(batch)

        # Get atoms number per system segment
        coll_batch['atoms_number'] = torch.tensor(
            [b['atoms_number'] for b in batch],
            device=self.device, dtype=torch.int64)

        # System segment index of atom i
        coll_batch['sys_i'] = torch.repeat_interleave(
            torch.arange(Nsys, device=self.device, dtype=torch.int64),
            repeats=coll_batch['atoms_number'], dim=0).to(
                device=self.device, dtype=torch.int64)

        # Atomic numbers properties
        coll_batch['atomic_numbers'] = torch.cat(
            [b['atomic_numbers'] for b in batch], 0).to(
                device=self.device, dtype=torch.int64)

        # Periodic boundary conditions
        coll_batch['positions'] = torch.cat(
            [b['positions'] for b in batch], 0).to(
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

        # Compute the cumulative segment size number
        atomic_numbers_cumsum = torch.cat(
            [
                torch.zeros(
                    (1,), device=self.device, dtype=coll_batch['sys_i'].dtype),
                torch.cumsum(coll_batch["atoms_number"][:-1], dim=0)
            ],
            dim=0).to(
                device=self.device, dtype=torch.int64)

        # Iterate over batch properties
        skip_props = [
            'atoms_number', 'atomic_numbers', 'positions', 'pbc', 'cell']
        for prop_i in batch[0]:

            # Skip previous parameter and None
            if prop_i in skip_props or batch[0].get(prop_i) is None:

                continue

            # Special property: energy
            elif prop_i == 'energy':
                
                if self.atomic_energies_shift is None:
                    
                    coll_batch[prop_i] = torch.tensor(
                        [b[prop_i] for b in batch],
                        device=self.device, dtype=self.dtype)
                    

                else:

                    shifted_energy = [
                        b[prop_i]
                        - torch.sum(
                            self.atomic_energies_shift[b['atomic_numbers']])
                        for b in batch]
                    coll_batch[prop_i] = torch.tensor(
                        shifted_energy,
                        device=self.device, dtype=self.dtype)
            
            # Special property: atomic energies
            elif prop_i == 'atomic_energies':
                
                if self.atomic_energies_shift is None:
                    
                    coll_batch[prop_i] = torch.cat(
                        [b[prop_i] for b in batch], 0).to(
                            device=self.device, dtype=self.dtype)

                else:
                    
                    shifted_atomic_energies = [
                        b[prop_i]
                        - torch.sum(
                            self.atomic_energies_shift[b['atomic_numbers']])
                        for b in batch]
                    coll_batch[prop_i] = torch.cat(
                        shifted_atomic_energies, 0).to(
                            device=self.device, dtype=self.dtype)
            
            # Properties (float data)
            else:

                # Concatenate tensor data
                if batch[0][prop_i].size():
                    coll_batch[prop_i] = torch.cat(
                        [b[prop_i] for b in batch], 0).to(
                            device=self.device, dtype=self.dtype)

                # Concatenate numeric data
                else:
                    coll_batch[prop_i] = torch.tensor(
                        [b[prop_i] for b in batch],
                        device=self.device, dtype=self.dtype)

        # Compute pair indices and position offsets
        if self.neighbor_list is None:
            self.init_neighbor_list()
        coll_batch = self.neighbor_list(
            coll_batch,
            atomic_numbers_cumsum=atomic_numbers_cumsum)

        return coll_batch

    @property
    def data_properties(self):
        return self.dataset.data_properties

