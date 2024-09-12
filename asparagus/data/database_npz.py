import os
import time
import json
import functools
from typing import Optional, Union, List, Dict, Tuple, Callable, Any

import traceback

from ase.parallel import DummyMPI
from ase.utils import Lock

import numpy as np

import torch

from asparagus import data
from asparagus import utils

__all__ = ['DataBase_npz']

# Current npz database version
VERSION = 1

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

# Structural property labels and array shape
structure_properties_shape = {
    'atoms_number':     (-1,),
    'atomic_numbers':   (-1,),
    'positions':        (-1, 3,),
    'charge':           (-1,),
    'cell':             (-1, 9,),
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

    # Structural and reference property dtypes
    properties_numpy_dtype = np.float64
    properties_torch_dtype = torch.float64

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
        self.metadata_label = 'metadata:'
        self.metadata_nlabel = len(self.metadata_label)

        # Initialize class variables
        self.connected = False
        self.data = None
        self.data_new = {}
        self.metadata_new = False

        return

    def _load(self):
        if os.path.exists(self.data_file):
            self.data = np.load(self.data_file)
        else:
            self.data = None
        self.connected = True
        return

    def _save(self):
        
        # Check for changed data
        if self.data_new or self.metadata_new:
            
            # Check if data are loaded
            if not self.connected:
                self._load()

            # Store new data
            data_merged = self._merge()
            np.savez(self.data_file, **data_merged)

            # Reset new data flags
            self.data_new = {}
            self.metadata_new = False
            
        # TODO update flag
        return

    def _merge(self):

        # If no new data or metadata are available return current data
        if not (bool(self.data_new) or self.metadata_new):
            return self.data

        # Update data if new data are available,
        # otherwise set as current data dictionary
        if self.data_new:
            data_merged = self._merge_data()
        else:
            data_merged = {}
            for prop in self.data:
                data_merged[prop] = self.data[prop]

        # Update metadata if changed
        if self.metadata_new:
        
            # Convert metadata to npz compatible dictionary
            md = self._convert_metadata()
        
            # Update data with metadata
            data_merged.update(md)

        return data_merged

    def _merge_data(self):
        
        # Initialize merged data dictionary
        data_merged = {}
        
        # Iterate over data properties and ids
        for prop in self.data:

            # Skip system and atoms id
            if prop in ['system:id', 'atoms:id']:
                continue

            # Skip metadata keys
            if (
                len(prop) > self.metadata_nlabel
                and prop[:self.metadata_nlabel] == self.metadata_label
            ):
                continue

            # Generate row id and system id lists from certain property
            if prop == system_id_property:
                
                next_sys_id = self.data['system:id'][-1]
                new_sys_id = next_sys_id + np.cumsum(
                    [data_i.shape[0] for data_i in self.data_new[prop]])
                data_merged['system:id'] = np.concatenate(
                    (self.data['system:id'], new_sys_id),
                    axis=0)
                
                last_row_id = self.get_last_id()
                new_row_id = last_row_id + 1 + np.arange(
                    len(self.data_new[prop]), dtype=integer_numpy_dtype)
                data_merged['id'] = np.concatenate(
                    (self.data['id'], new_row_id),
                    axis=0)

            # Generate atoms id lists from certain property
            if prop == atoms_id_property:
                next_sys_id = self.data['atoms:id'][-1]
                new_sys_id = next_sys_id + np.cumsum(
                    [data_i.shape[0] for data_i in self.data_new[prop]])
                data_merged['atoms:id'] = np.concatenate(
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
                data_merged[prop] = np.concatenate(
                    (self.data[prop], new_prop_id),
                    axis=0)

            # Merge property lists
            else:

                data_merged[prop] = np.concatenate(
                    (self.data[prop], *self.data_new[prop]),
                    axis=0)

        return data_merged

    def __enter__(self):
        self._load()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            raise exc_type
        else:
            self._save()
        self.connected = False
        self.data = None
        self.data_new = {}
        return

    @lock
    def _set_metadata(self, metadata):

        # Set new metadata flag
        self.metadata_new = True

        # Store metadata as class variable
        self._metadata = metadata

        # Add version
        self._metadata['version'] = VERSION

        # Initialize data system
        self._init_systems()

        return self._metadata

    def _get_metadata(self):

        # Load metadata from data if not stored
        if not len(self._metadata):
            self._metadata = self._load_metadata()

        return self._metadata

    def _load_metadata(self):

        # Check if data are loaded
        if not self.connected:
            self._load()

        # Initialize metadata dictionary
        metadata = {}

        # Check data dictionary
        if self.data is None:
            return metadata

        # Iterate over data keys
        for key in self.data:

            # Skip if not a metadata label
            if (
                len(key) < self.metadata_nlabel
                or key[:self.metadata_nlabel] != self.metadata_label
            ):
                continue

            # Extract metadata key
            metadata_key = key[self.metadata_nlabel:]

            # Check metadata item
            if utils.is_array_like(self.data[key]):
                data_item = self.data[key].tolist()
            elif utils.is_string(self.data[key]):
                data_item = str(self.data[key])
            else:
                data_item = self.data[key]

            # Check for converted subdictionary keys or items
            if ':' in metadata_key:
                
                # Get metadata key, as well as subdictionary key and item
                metadata_keys = metadata_key.split(':')
                metadata_key = metadata_keys[0]
                dict_key = metadata_keys[1]

                # Set subdictionary key and items
                if metadata_key not in metadata:
                    metadata[metadata_key] = {}
                metadata[metadata_key][dict_key] = data_item
            
            else:
                
                # Set metadata property
                metadata[metadata_key] = data_item

        return metadata

    def _convert_metadata(self):

        # Initialize npz compatible metadata dictionary
        metadata_npz = {}
        
        for key, metadata_item in self._metadata.items():
            
            # Create metadata key
            metadata_key = self.metadata_label + key

            # Check for dictionary conversion
            if utils.is_dictionary(metadata_item):
                
                # Set subdictionary keys and items separately
                for dict_key, dict_item in metadata_item.items():

                    # Check metadata item
                    if utils.is_array_like(dict_item):
                        dict_item = np.array(dict_item)

                    # Modify metadata key and add to metadata
                    metadata_dict_key = metadata_key + ':' + str(dict_key)
                    metadata_npz[metadata_dict_key] = dict_item

            # Else, store directly
            else:
                
                # Check metadata item
                if utils.is_array_like(metadata_item):
                    metadata_item = np.array(metadata_item)

                metadata_npz[metadata_key] = metadata_item

        return metadata_npz

    def _init_systems(self):
        
        # Get metadata
        metadata = self._get_metadata()
        
        # Init data dictionary
        if self.data is None and metadata.get('load_properties') is not None:
            self._init_data()
        
        # Get version
        if metadata.get('version') is None:
            self.version = VERSION
        else:
            self.version = metadata.get('version')

        # Check version compatibility
        if self.version > VERSION:
            raise IOError(
                f"Can not read newer version of the database format "
                f"(version {self.version}).")

        return

    def _get(self, selection, **kwargs):

        # Check if data are loaded
        if not self.connected:
            self._load()

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
                row[prop_i] = torch.tensor(
                    self.data[prop_i][id_start:id_end],
                    dtype=self.properties_torch_dtype)

            # Append data
            rows.append(row)

        return rows

    def _init_data(self):

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
                    [], dtype=self.properties_numpy_dtype).reshape(
                        reference_properties_shape[prop_i])
                self.data[prop_i + ':id'] = np.array(
                    [0], dtype=integer_numpy_dtype)

        return

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

        # Check if data are loaded
        if not self.connected:
            self._load()

        # Check data dictionary
        if self.data is None:
            self._init_data()

        # Check if 'row_id' already already occupied
        if row_id is not None and row_id in self.data['id']:
            row_id = self._update(row_id, properties)
            return row_id

        # Initialize new data dictionary
        if not self.data_new:
            self._init_data_new()

        # Assign new id
        if row_id is None:
            row_id = self.get_last_id() + 1
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
                        properties.get(prop_i),
                        dtype=self.properties_numpy_dtype).reshape(
                            reference_properties_shape[prop_i])
                    self.data_new[prop_i].append(data_i)
                    self.data_new[prop_i + ':id'].append(
                        np.array(
                            data_i.shape[0],
                            dtype=integer_numpy_dtype)
                        )

        return row_id

    def _update(self, row_id, properties):

        # Check if data are loaded
        if not self.connected:
            self._load()

        # Check data dictionary
        if self.data is None:
            self._init_data()

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

        # Check if data are loaded
        if not self.connected:
            self._load()

        # Get last row id
        if 'id' in self.data and len(self.data['id']):
            row_id = self.data['id'][-1]
        else:
            row_id = 0

        return row_id

    def _count(self, cmps):

        # Check if data are loaded or new data are in the queue
        if not self.connected:
            self._load()
        if self.data_new:
            self._merge()

        if self.data is None:
            return 0
        elif 'id' in self.data and len(self.data['id']):
            return self.data['id'][-1]
        else:
            return 0

    def _delete(self, row_ids):
        raise NotImplementedError()

    def _delete_file(self):
        """
        Delete database and metadata file
        """
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        return

    @property
    def metadata(self):
        return self._get_metadata()
