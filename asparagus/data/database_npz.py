import os
import time
import json
import logging
import functools
from typing import Optional, Union, List, Dict, Tuple, Callable, Any

from ase.parallel import DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock

import numpy as np

import torch

from .. import data
from .. import utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['DataBase_npz']

# Current npz database version
VERSION = 0

# Structural property labels and dtypes
integer_numpy_dtype = np.int32
string_numpy_dtype = 'U24'
structure_properties_dtype = {
    'atoms_number':     integer_numpy_dtype,
    'atomic_numbers':   integer_numpy_dtype,
    'positions':        np.float32,
    'charge':           np.float32,
    'cell':             np.float32,
    'pbc':              np.bool_,
}
properties_numpy_dtype = np.float64
reference_properties_torch_dtype = torch.float64

# Structural property labels and array shape
structure_properties_shape = {
    'atoms_number':     (-1,),
    'atomic_numbers':   (-1,),
    'positions':        (-1, 3,),
    'charge':           (-1,),
    'cell':             (-1, 3), # TODO Check cell shape when adding to db
    'pbc':              (-1, 3,),
    }

# Known reference property array shape
reference_properties_shape = {
    'energy':           (-1,),
    'atomic_energies':  (-1,),
    'forces':           (-1, 3,),
    #'hessian':          (-1,),
    'atomic_charge':    (-1,),
    'dipole':           (-1, 3,),
    'atomic_dipoles':   (-1, 3,),
    'polarizability':   (-1, 3, 3,),
    }

# Structural property identification indices
structure_properties_ids = {
    'atoms_number':     'system:id',
    'atomic_numbers':   'atoms:id',
    'positions':        'atoms:id',
    'charge':           'system:id',
    'cell':             'system:id',
    'pbc':              'system:id',
}
system_id_property = 'atoms_number'
atoms_id_property = 'atomic_numbers'


def connect(
    data_file,
    lock_file: Optional[bool] = True,
    **kwargs,
) -> object:
    """
    Create connection to database.

    Parameters
    ----------
    data_file: str
        Database file path
    lock_file: bool, optional, default True
        Use a lock file

    Returns
    -------
    data.Database_npz
        Numpy npz database interface object

    """

    return DataBase_npz(data_file, lock_file)


def lock(method):
    """
    Decorator for using a lock-file.
    """
    @functools.wraps(method)
    def new_method(self, *args, **kwargs):
        if self.lock is None:
            return method(self, *args, **kwargs)
        else:
            with self.lock:
                return method(self, *args, **kwargs)
    return new_method


class DataBase_npz(data.DataBase):
    """
    Numpy npz data base class
    """

    _metadata = {}

    def __init__(
        self,
        data_file: str,
        lock_file: bool,
    ):
        """
        Numpy Database object that contain reference data.

        Parameters
        ----------
        data_file: str
            Reference database file
        lock_file: bool
            Use a lock file when manipulating the database to prevent
            parallel manipulation by multiple processes.

        """

        # Inherit from DataBase base class
        super().__init__(data_file)

        # Check for .npz suffix
        if self.data_file.split('.')[-1].lower() != 'npz':
            self.data_file = f"{self.data_file:s}.npz"

        # Prepare data locker
        if lock_file and utils.is_string(data_file):
            self.lock = Lock(self.data_file + '.lock', world=DummyMPI())
        else:
            self.lock = None

        # Prepare metadata file path
        self.metadata_file = os.path.join(f"{self.data_file:s}.json")
        self.metadata_indent = 2
        
        # Initialize class variables
        self.data = None
        self.next_id = None
        self.data_new = {}

        return

    def _load(self):
        if os.path.exists(self.data_file):
            self.data = np.load(self.data_file)
            self.next_id = self.get_last_id() + 1
        else:
            self.data = None
            self.next_id = None
        return

    def _save(self):
        if self.data_new:
            self._merge()
            np.savez(self.data_file, **self.data)
        # TODO update flag
        return

    def _merge(self):

        for prop in self.data:
            
            # Skip system and atoms id
            if prop in ['system:id', 'atoms:id']:
                continue
            
            # Generate system id lists from certain property and merge
            if prop == system_id_property:
                next_sys_id = self.data['system:id'][-1]
                new_sys_id = next_sys_id + np.cumsum(
                    [data_i.shape[0] for data_i in self.data_new[prop]])
                self.data['system:id'] = np.concatenate(
                    (self.data['system:id'], new_sys_id),
                    axis=0)

            # Generate atoms id lists from certain property and merge
            if prop == atoms_id_property:
                next_sys_id = self.data['atoms:id'][-1]
                new_sys_id = next_sys_id + np.cumsum(
                    [data_i.shape[0] for data_i in self.data_new[prop]])
                self.data['atoms:id'] = np.concatenate(
                    (self.data['atoms:id'], new_sys_id),
                    axis=0)

            # Each other property must be found in the new data
            if self.data_new.get(prop) is None:
                raise SyntaxError(
                    "New data do not contain property information for "
                    + f"'{prop:s}'!")

            # Finalize property id lists and merge
            if len(prop) > 3 and prop[-3:] == ':id':
                
                next_prop_id = self.data[prop][-1]
                new_prop_id = next_prop_id + np.cumsum(
                    [data_i for data_i in self.data_new[prop]])
                self.data[prop] = np.concatenate(
                    (self.data[prop], new_prop_id),
                    axis=0)
            
            # Merge property lists
            else:

                self.data[prop] = np.concatenate(
                    (self.data[prop], *self.data_new[prop]),
                    axis=0)

        # TODO Update ids
        self.data_new = {}
        
        return

    def __enter__(self):
        self._load()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            raise exc_type
        else:
            self._save()
        self.data = None
        self.next_id = None
        self.data_new = {}
        return

    @lock
    def _set_metadata(self, metadata):

        # Store metadata to file
        with open(self.metadata_file, 'w') as f:
            json.dump(
                metadata,
                f,
                indent=self.metadata_indent,
                default=str)

        # Store metadata as class variable
        self._metadata = metadata

        return self._metadata

    def _get_metadata(self):

        if not len(self._metadata):

            if os.path.exists(self.metadata_file):
                
                # Read metadata
                with open(self.metadata_file, 'r') as f:
                    self._metadata = json.load(f)

            else:

                # Initialize metadata
                self._metadata = {}

        return self._metadata

    def _init_systems(self):
        
        # Initialize data dictionary
        self.data = {}

        # Initialize data id and system id list
        self.data['id'] = np.array([], dtype=integer_numpy_dtype)
        self.data['system:id'] = np.array([0], dtype=integer_numpy_dtype)
        self.data['atoms:id'] = np.array([0], dtype=integer_numpy_dtype)

        # Initialize lists for current data time and User name
        self.data['mtime'] = np.array([], dtype=string_numpy_dtype)
        self.data['username'] = np.array([], dtype=string_numpy_dtype)

        # Initialize lists for structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():
            self.data[prop_i] = np.array([], dtype=dtype_i).reshape(
                structure_properties_shape[prop_i])

        # Initialize lists for reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype:
                self.data[prop_i] = np.array(
                    [], dtype=properties_numpy_dtype).reshape(
                        reference_properties_shape[prop_i])
                self.data[prop_i + ':id'] = np.array(
                    [0], dtype=integer_numpy_dtype)

        # Initialize next id
        self.next_id = 1

        return

    def _get(self, selection, **kwargs):

        # Reference Data list
        rows = []

        # Iterate over selection
        for idx in selection:
            
            # Initialize reference data dictionary
            row = {}
            
            # Get structural properties
            for prop_i in structure_properties_dtype:
                
                # Get system ids
                id_start = self.data[structure_properties_ids[prop_i]][idx - 1]
                id_end = self.data[structure_properties_ids[prop_i]][idx]

                # Get property
                row[prop_i] = torch.tensor(self.data[prop_i][id_start:id_end])

            # Get reference properties
            for prop_i in self.metadata.get('load_properties'):
                
                # Get system ids
                id_start = self.data[prop_i + ':id'][idx - 1]
                id_end = self.data[prop_i + ':id'][idx]

                # Get property
                row[prop_i] = torch.tensor(self.data[prop_i][id_start:id_end])

            # Append data
            rows.append(row)

        return rows

    def _init_data_new(self):

        # Initialize data dictionary
        self.data_new = {}

        # Initialize data id list
        self.data_new['id'] = []

        # Initialize lists for current data time and User name
        self.data_new['mtime'] = []
        self.data_new['username'] = []

        # Initialize lists for structural properties
        for prop_i in structure_properties_dtype:
            self.data_new[prop_i] = []

        # Initialize lists for reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype:
                self.data_new[prop_i] = []
                self.data_new[prop_i + ':id'] = []

        return

    def _reset(self):

        # Reset stored metadata dictionary
        self._metadata = {}
        return

    @lock
    def _write(self, properties, row_id):

        # Check data dictionary
        if self.data is None:
            self._init_systems()

        # Check if 'row_id' already already occupied
        if row_id is not None and row_id in self.data['id']:
            row_id = self._update(row_id, properties)
            return row_id

        # Initialize new data dictionary
        if not self.data_new:
            self._init_data_new()

        # Assign new id
        if row_id is None:
            row_id = self.next_id
        self.data_new['id'].append(
            np.array([row_id], dtype=integer_numpy_dtype))
        
        # Current datatime and User name
        self.data_new['mtime'].append(
            np.array([time.ctime()], dtype=string_numpy_dtype))
        self.data_new['username'].append(
            np.array([os.getenv('USER')], dtype=string_numpy_dtype))

        # Structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():
            if properties.get(prop_i) is None:
                self.data_new[prop_i].append(None)
            else:
                self.data_new[prop_i].append(
                    np.array(properties.get(prop_i), dtype=dtype_i).reshape(
                        structure_properties_shape[prop_i]))

        # Reference properties
        for prop_i in self.metadata.get('load_properties'):
            if prop_i not in structure_properties_dtype:
                if properties.get(prop_i) is None:
                    self.data_new[prop_i].append(None)
                else:
                    data_i = np.array(
                        [properties.get(prop_i)],
                        dtype=properties_numpy_dtype).reshape(
                            reference_properties_shape[prop_i])
                    self.data_new[prop_i].append(data_i)
                    self.data_new[prop_i + ':id'].append(
                        np.array(
                            data_i.shape[0],
                            dtype=integer_numpy_dtype)
                        )

        # Increment next id
        self.next_id += 1
        
        return row_id

    def _update(self, row_id, properties):

        # Check data dictionary
        if self.data is None:
            self._init_systems()

        # Check if 'row_id' already already occupied
        if row_id is None or row_id not in self.data['id']:
            row_id = self._write(properties, row_id)
            return row_id

        raise NotImplementedError
        ## Current datatime and User name
        #self.data_new['mtime'].append(time.ctime())
        #self.data_new['username'].append(os.getenv('USER'))

        ## Structural properties
        #for prop_i, dtype_i in structure_properties_dtype.items():
            #if properties.get(prop_i) is None:
                #self.data_new[prop_i].append(None)
            #else:
                #self.data_new[prop_i].append(
                    #np.array([properties.get(prop_i)], dtype=dtype_i))

        ## Reference properties
        #for prop_i in self.metadata.get('load_properties'):
            #if prop_i not in structure_properties_dtype:
                #if properties.get(prop_i) is None:
                    #self.data_new[prop_i].append(None)
                #else:
                    #self.data_new[prop_i].append(
                        #np.array([properties.get(prop_i)], dtype=dtype_i))

        #if not new_data:

            #raise SyntaxError(
                #"At least one input 'ref_data' or 'properties' should "
                #+ "contain reference data!")

        #elif ref_data is None:

            #row_id = self._write(properties, row_id)

        #else:

            ## Add or update database values
            #with self.managed_connection() as data:

                #key_id = f"id{row_id:d}"
                #if not isinstance(self.data, dict):
                    #self.data = dict(self.data)
                #self.data[key_id] = ref_data

        return row_id

    def get_last_id(self):
        
        # Get last row id
        if 'id' in self.data:
            row_id = self.data['id'][-1]
            self.next_id = row_id + 1
        else:
            self.next_id = 1

        return row_id

    def _count(self, cmps):

        if self.data is None:
            self._load()
        elif self.data_new:
            self._merge()

        return self.data['id'][-1]

    def _delete(self, row_ids):
        raise NotImplementedError()

    def _delete_file(self):
        """
        Delete database and metadata file
        """
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        return
            

    @property
    def metadata(self):
        return self._get_metadata()
