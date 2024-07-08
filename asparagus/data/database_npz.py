import os
import sys
import time
import json
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import torch

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['DataBase_npz']

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
properties_numpy_dtype = np.float32
properties_torch_dtype = torch.float32

# Structural property labels and array shape
structure_properties_shape = {
    'atoms_number':     (),
    'atomic_numbers':   (-1,),
    'positions':        (-1, 3,),
    'charge':           (-1,),
    'cell':             (1, 3,),
    'pbc':              (1, 3,),
    'idx_i':            (-1,),
    'idx_j':            (-1,),
    'pbc_offset':       (-1, 3,),
}


class DataBase_npz(data.DataBase):
    """
    Numpy npz data base class
    """
    
    def __init__(
        self, 
        data_file, 
        lock_file
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
                
        Returns
        -------
            object
                Numpy Database for data storing
        """
        
        # Inherit from DataBase base class
        super().__init__(data_file, lock_file)
        
        return
    
    def _load(self):
        if os.path.exists(self.data_file):
            self.data = np.load(self.data_file)
        else:
            self.data = {}
        return 

    def _save(self, data=None):
        if data is None:
            np.savez(self.data_file, **self.data)
        else:
            np.savez(self.data_file, **data)
        return 

    def __enter__(self):
        self._load()
        return self.data

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            raise exc_type
        else:
            self._save()
        return

    def _set_metadata(self, metadata):
        
        # Convert metadata dictionary
        md = json.dumps(metadata)
        
        # Update or set metadata
        self.data = dict(self.data)
        self.data['metadata'] = md
        self._save()
    
    def _get_metadata(self):
        
        # Read metadata
        metadata = self.data['metadata']
            
        return metadata

    def _init_systems(self):
        
        self._load()
        self._save()
        
        return

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
            with self.managed_connection() as data:
                
                key_id = f"id{row_id:d}"
                if not isinstance(self.data, dict):
                    self.data = dict(self.data)
                self.data[key_id] = ref_data

        return row_id

    def get_last_id(self):
        
        data_files = self.data.files
        
        # Get last row id
        if 'last_id' in data_files:
            row_id = self.data['last_id']
            next_key_id = f"id{row_id+1:d}"
        else:
            row_id = 0
            next_key_id = f"id{row_id+1:d}"
        
        # Check last row id
        while next_key_id in data_files:
            row_id += 1
            next_key_id = f"id{row_id+1:d}"

        return row_id
    
    def _select(self, cmps, verbose=False):
        
        sql, args = self.create_select_statement(cmps)
        metadata = self._get_metadata()
        
        with self.managed_connection() as con:
            
            # Execute SQL request
            cur = con.cursor()
            cur.execute(sql, args)
            
            for row in cur.fetchall():
                
                yield self.convert_row(row, metadata, verbose=verbose)
                
    def _count(self, cmps):
        
        sql, args = self.create_select_statement(cmps, what='COUNT(*)')

        with self.managed_connection() as con:
            cur = con.cursor()
            try:
                cur.execute(sql, args)
                return cur.fetchone()[0]
            except sqlite3.OperationalError:
                return 0

    def delete(self, row_ids):
        """
        Delete rows.
        
        Parameters
        ----------
            row_ids: int or list of int
                Row index or list of indices to delete
        """
        
        if len(row_ids) == 0:
            return
        
        self._delete(row_ids)

    def _delete(self, row_ids):
        raise NotImplementedError()

    @property
    def metadata(self):
        return self._get_metadata()
        

    
