import os
import time
import json
import logging
from typing import Optional, Union, List, Dict, Tuple, Callable, Any

import numpy as np

import torch

import h5py

from .. import data
from .. import utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['DataBase_hdf5']

# Current npz database version
VERSION = 0

# Structural property labels and dtypes
structure_properties_dtype = {
    'atoms_number':     np.int32,
    'atomic_numbers':   np.int32,
    'positions':        np.float32,
    'charge':           np.float32,
    'cell':             np.float32,
    'pbc':              np.bool_,
    'idx_i':            np.int32,
    'idx_j':            np.int32,
    'pbc_offset':       np.float32,
}
properties_numpy_dtype = np.float64
properties_torch_dtype = torch.float64

# Structural property labels and array shape
structure_properties_shape = {
    'atoms_number':     (),
    'atomic_numbers':   (-1,),
    'positions':        (-1, 3,),
    'charge':           (-1,),
    'cell':             (-1),
    'pbc':              (1, 3,),
    'idx_i':            (-1,),
    'idx_j':            (-1,),
    'pbc_offset':       (-1, 3,),
}
reference_properties_shape = {
    # 'energy':           (-1,),
    # 'atomic_energies':  (-1,),
    'forces':           (-1, 3,),
    # 'hessian':          (-1,),
    # 'atomic_charge':    (-1,),
    # 'dipole':           (3),
    # 'atomic_dipoles':   (-1,),
    'polarizability':   (3, 3,),
    }


def connect(
    data_file,
    mode: Optional[str] = 'r',
    **kwargs,
) -> object:
    """
    Create connection to database.

    Parameters
    ----------
    data_file: str
        Database file path
    mode: str, optional, default 'r'
        Mode to open file ('r': just reading, 'w': writing, 'a': appending)

    Returns
    -------
    data.Database_hdf5
        HDF5 database interface object

    """
    if os.path.exists(data_file) or mode != 'r':
        return DataBase_hdf5(data_file, mode)
    else:
        return DataBase_hdf5(data_file, 'w')


class DataBase_hdf5(data.DataBase):
    """
    HDF5 data base class
    """

    # Initialize metadata parameter
    _metadata = {}

    def __init__(
        self,
        data_file: str,
        mode: str,
    ):
        """
        HDF5 Database object that contain reference data.

        Parameters
        ----------
        data_file: str
            Reference database file
        mode: str,
            Conection mode such as read(r), write(w) or append(a).

        Returns
        -------
        data.DataBase
            h5py object for data storing

        """

        # Inherit from DataBase base class
        super().__init__(data_file)

        self.data = None
        self.mode = mode

        return

    def __enter__(self):
        self.data = h5py.File(self.data_file, self.mode)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.data.name:
            self.data.close()
            self.data = None
        if exc_type is not None:
            raise exc_type
        return

    def _set_metadata(self, metadata):

        # Convert metadata dictionary
        md = json.dumps(metadata)

        # Update or set metadata
        if self.data.get('metadata') is not None:
            del self.data['metadata']
        self.data['metadata'] = md

        # Store metadata
        self._metadata = metadata

        return

    def _get_metadata(self):

        # Read metadata
        if not len(self._metadata):

            if self.data.get('metadata') is None:
                self._metadata = {}
            else:
                md = self.data['metadata']
                self._metadata = json.loads(str(np.array(md, dtype=str)))

        return self._metadata

    def _init_systems(self):

        # Set or check version
        if self.data.get('version') is None:
            self.data['version'] = VERSION
        else:
            self.version = np.array(self.data['version'], dtype=np.int32)
            if self.version > VERSION:
                raise IOError(
                    f"Can not read newer version of the database format "
                    f"(version {self.version}).")

        if self.data.get('systems') is None:
            self.data.create_group('systems')
        if self.data.get('last_id') is None:
            self.data['last_id'] = 0

        return

    def _reset(self):

        # Reset stored metadata dictionary
        self._metadata = {}
        return

    def _get(self, selection, **kwargs):
        rows = []
        metadata = self._get_metadata()
        for idx in selection:
            row = {}
            for prop_i, dtype_i in structure_properties_dtype.items():
                item = self.data['systems'][f"{selection[0]:d}"].get(prop_i)
                if item is not None:
                    row[prop_i] = torch.tensor(np.array(item, dtype=dtype_i))
            for prop_i in metadata.get('load_properties'):
                item = self.data['systems'][f"{selection[0]:d}"].get(prop_i)
                if item is not None:
                    row[prop_i] = torch.tensor(
                        np.array(item, dtype=properties_numpy_dtype))
            rows.append(row)
        return rows

    def get_last_id(self):

        # Get last row id
        if self.data.get('last_id') is not None:
            row_id = np.array(self.data['last_id'], dtype=np.int32)
        elif (
            self.data.get('last_id') is None
            and self.data.get('systems') is None
        ):
            row_id = 0
        elif (
            self.data.get('last_id') is None
            and self.data.get('systems') is not None
        ):
            row_id = 0
            for key in self.data['systems']:
                try:
                    key_id = int(key)
                except ValueError:
                    pass
                else:
                    if key_id > row_id:
                        row_id = key_id
        else:
            row_id = 0

        return row_id

    def _write(self, properties, row_id):

        # Reference data list
        ref_data = {}

        # Current datatime and User name
        ref_data['mtime'] = time.ctime()
        ref_data['username'] = os.getenv('USER')

        # Structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():
            if properties.get(prop_i) is None:
                ref_data[prop_i] = None
            elif utils.is_array_like(properties.get(prop_i)):
                ref_data[prop_i] = np.array(
                    properties.get(prop_i), dtype=dtype_i)
            else:
                ref_data[prop_i] = dtype_i(properties.get(prop_i))

        # Reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype.keys():
                if properties.get(prop_i) is None:
                    ref_data[prop_i] = None
                elif utils.is_array_like(properties.get(prop_i)):
                    ref_data[prop_i] = np.array(
                        properties.get(prop_i), dtype=dtype_i)
                else:
                    ref_data[prop_i] = dtype_i(properties.get(prop_i))

        # Add or update database values
        if row_id is None:
            row_id = self.get_last_id() + 1
        row_id = self._update(row_id, ref_data)
    
        return row_id

    def update(self, row_id, properties):

        # Reference data list
        ref_data = {}
        
        # Current datatime and User name
        ref_data['mtime'] = time.ctime()
        ref_data['username'] = os.getenv('USER')
        
        # Structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():
            if properties.get(prop_i) is None:
                ref_data[prop_i] = None
            elif utils.is_array_like(properties.get(prop_i)):
                ref_data[prop_i] = np.array(
                    properties.get(prop_i), dtype=dtype_i)
            else:
                ref_data[prop_i] = dtype_i(properties.get(prop_i))

        # Reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype.keys():
                if properties.get(prop_i) is None:
                    ref_data[prop_i] = None
                elif utils.is_array_like(properties.get(prop_i)):
                    ref_data[prop_i] = np.array(
                        properties.get(prop_i), dtype=dtype_i)
                else:
                    ref_data[prop_i] = dtype_i(properties.get(prop_i))

        # Add or update database values
        row_id = self._update(row_id, ref_data)

        return row_id

    def _update(self, row_id, ref_data=None, properties=None):
        
        if ref_data is None and properties is None:
            
            raise SyntaxError(
                "At least one input 'ref_data' or 'properties' should "
                + "contain reference data!")
        
        elif ref_data is None:
            
            row_id = self._write(properties, row_id)
            
        else:
        
            # Add or update database values
            key_id = f"{row_id:d}"
            if self.data['systems'].get(key_id) is None:
                self.data['systems'].create_group(key_id)
                for key, item in ref_data.items():
                    if item is not None:
                        self.data['systems'][key_id][key] = item
                self.data['last_id'][...] = row_id
            else:
                del self.data['systems'][key_id]
                for key, item in ref_data.items():
                    self.data['systems'][key] = item

        return row_id

    def _count(self, selection, **kwargs):
        if self.data.get('systems') is None:
            return 0
        else:
            return len(self.data['systems'])

    def _delete(self, row_ids):
        raise NotImplementedError()

    def _delete_file(self):
        """
        Delete database file
        """
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        return

    @property
    def metadata(self):
        return self._get_metadata()
        

    
