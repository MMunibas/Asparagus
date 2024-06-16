import os
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple, Union

import torch

from .. import data
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataLoader']

class DataLoader(torch.utils.data.DataLoader):
    """
    Data loader class from a dataset
    
    Parameters
    ----------
    dataset: (data.DataSet, data.DataSubSet)
        DataSet or DataSubSet instance of reference data
    data_batch_size: int
        Number of atomic systems per batch
    data_shuffle: bool
        Shuffle batch compilation after each epoch
    data_num_workers: int
        Number of parallel workers for collecting data
    data_collate_fn: callable, optional, default None
        Callable function that prepare and return batch data
    data_pin_memory: bool, optional, default False
        If True data are loaded to GPU
    device: str, optional, default 'cpu'
        Device type for data allocation
    dtype: dtype object, optional, default 'torch.float64'
        Reference data type to convert to

    """

    def __init__(
        self,
        dataset: Union[data.DataSet, data.DataSubSet],
        data_batch_size: int,
        data_shuffle: bool,
        data_num_workers: int,
        device: str,
        dtype: object,
        data_collate_fn: Optional[object] = None,
        data_pin_memory: Optional[bool] = False,
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
        self.data_batch_size = data_batch_size

        if device == 'cpu':
            self.data_num_workers = 0
        else:
            self.data_num_workers = data_num_workers

        # Initiate DataLoader
        super(DataLoader, self).__init__(
            dataset=dataset,
            batch_size=data_batch_size,
            shuffle=data_shuffle,
            num_workers=self.data_num_workers,
            collate_fn=self.data_collate_fn,
            pin_memory=data_pin_memory,
            **kwargs
        )
        
        # Initialize neighbor list function class parameter
        self.neighbor_list = None
        
        # Assign reference data conversion parameter
        self.device = device
        self.dtype = dtype

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
        self.neighbor_list = utils.TorchNeighborListRangeSeparated(
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

