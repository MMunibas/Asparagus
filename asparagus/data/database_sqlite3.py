import os
import sys
import time
import json
import sqlite3
import functools
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
from contextlib import contextmanager

from ase.parallel import world, DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock
from ase.io.jsonio import create_ase_object

import numpy as np

import torch

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['DataBase_SQLlite3']

# Current SQLite3 database version
VERSION = 0

all_tables = ['systems']


# Initial SQL statement lines
init_systems = [
    """CREATE TABLE systems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mtime TEXT,
    username TEXT,
    atoms_number BLOB,
    atomic_numbers BLOB,
    positions BLOB,
    charge BLOB,
    cell BLOB,
    pbc BLOB,
    idx_i BLOB,
    idx_j BLOB,
    pbc_offset BLOB,
    """]

init_information = [
    """CREATE TABLE information (
    name TEXT,
    value TEXT)""",
    "INSERT INTO information VALUES ('version', '{}')".format(VERSION)]

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
    'cell':             (-1,),
    'pbc':              (1, 3,),
    'idx_i':            (-1,),
    'idx_j':            (-1,),
    'pbc_offset':       (-1, 3,),
}


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


def object_to_bytes(
    obj: Any
) -> bytes:
    """
    Serialize Python object to bytes.
    """
    
    parts = [b'12345678']
    obj = o2b(obj, parts)
    offset = sum(len(part) for part in parts)
    x = np.array(offset, np.int64)
    if not np.little_endian:
        x.byteswap(True)
    parts[0] = x.tobytes()
    parts.append(json.dumps(obj, separators=(',', ':')).encode())
    return b''.join(parts)


def bytes_to_object(
    b: bytes
) -> Any:
    """
    Deserialize bytes to Python object.
    """
    
    x = np.frombuffer(b[:8], np.int64)
    if not np.little_endian:
        x = x.byteswap()
    offset = x.item()
    obj = json.loads(b[offset:].decode())
    return b2o(obj, b)


def o2b(
    obj: Any, 
    parts: List[bytes]
) -> Any:
    
    if (
        obj is None 
        or utils.is_numeric(obj) 
        or utils.is_bool(obj) 
        or utils.is_string(obj)
    ):
        return obj
    
    if utils.is_dictionary(obj):
        return {key: o2b(value, parts) for key, value in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [o2b(value, parts) for value in obj]
    
    if isinstance(obj, np.ndarray):
        
        assert obj.dtype != object, \
            'Cannot convert ndarray of type "object" to bytes.'
        offset = sum(len(part) for part in parts)
        if not np.little_endian:
            obj = obj.byteswap()
        parts.append(obj.tobytes())
        return {'__ndarray__': [obj.shape, obj.dtype.name, offset]}
    
    if isinstance(obj, complex):
        return {'__complex__': [obj.real, obj.imag]}
    
    objtype = type(obj)
    raise ValueError(
        f"Objects of type {objtype} not allowed")


def b2o(
    obj: Any, 
    b: bytes
) -> Any:
    
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj

    if isinstance(obj, list):
        return [b2o(value, b) for value in obj]

    assert isinstance(obj, dict)

    x = obj.get('__complex__')
    if x is not None:
        return complex(*x)

    x = obj.get('__ndarray__')
    if x is not None:
        shape, name, offset = x
        dtype = np.dtype(name)
        size = dtype.itemsize * np.prod(shape).astype(int)
        a = np.frombuffer(b[offset:offset + size], dtype)
        a.shape = shape
        if not np.little_endian:
            a = a.byteswap()
        return a

    dct = {key: b2o(value, b) for key, value in obj.items()}
    objtype = dct.pop('__ase_objtype__', None)
    if objtype is None:
        return dct
    return create_ase_object(objtype, dct)


class DataBase_SQLite3(data.DataBase):
    """
    SQL lite 3 data base class
    """
    
    # Initialize connection interface
    connection = None
    _metadata = {}
    
    # Used for autoincrement id
    default = 'NULL'
    
    def __init__(
        self, 
        data_file: str,
        lock_file: bool,
    ):
        """
        SQLite3 dataBase object that contain reference data.
        This is a condensed version of the ASE Database class:
        https://gitlab.com/ase/ase/-/blob/master/ase/db/sqlite.py

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
            SQLite3 dataBase for data storing
        """
        
        # Inherit from DataBase base class
        super().__init__(data_file)
        
        # Prepare data locker
        if lock_file and utils.is_string(data_file):
            self.lock = Lock(data_file + '.lock', world=DummyMPI())
        else:
            self.lock = None
        
        return
    
    def _connect(self):
        return sqlite3.connect(self.data_file, timeout=20)

    def __enter__(self):
        assert self.connection is None
        self.change_count = 0
        self.connection = self._connect()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        self.connection = None

    @contextmanager
    def managed_connection(self, commit_frequency=5000):
        try:
            con = self.connection or self._connect()
            yield con
        except ValueError as exc:
            if self.connection is None:
                con.close()
            raise exc
        else:
            if self.connection is None:
                con.commit()
                con.close()
            else:
                self.change_count += 1
                if self.change_count % commit_frequency == 0:
                    con.commit()

    @lock
    def _set_metadata(self, metadata):
        
        # Convert metadata dictionary
        md = json.dumps(metadata)
        
        with self.managed_connection() as con:
            
            # Select any data in the database
            cur = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='information'")

            if cur.fetchone()[0]:

                # Update metadata if existing
                cur.execute(
                    "UPDATE information SET value=? WHERE name='metadata'", 
                    [md])
                con.commit()

            else:
                
                # Initialize data columns 
                for statement in init_information:
                    con.execute(statement)
                con.commit()
                
                # Write metadata
                cur.execute(
                    "INSERT INTO information VALUES (?, ?)", ('metadata', md))
                con.commit()
        
        # Store metadata
        self._metadata = metadata

    def _get_metadata(self):
        
        # Read metadata if not in memory
        if not len(self._metadata):

            with self.managed_connection() as con:

                # Check if table 'information' exists
                cur = con.execute(
                    'SELECT name FROM sqlite_master "'
                    + '"WHERE type="table" AND name="information"')
                result = cur.fetchone()
                
                if result is None:
                    self._metadata = {}
                    return self._metadata
                
                # Check if metadata exist
                cur = con.execute(
                    'SELECT count(name) FROM information '
                    + 'WHERE name="metadata"')
                result = cur.fetchone()[0]
                
                # Read metadata if exist
                if result:
                    cur = con.execute(
                        'SELECT value FROM information WHERE name="metadata"')
                    results = cur.fetchall()
                    if results:
                        self._metadata = json.loads(results[0][0])
                else:
                    self._metadata = {}

        return self._metadata

    @lock
    def _init_systems(self):
        
        # Get metadata
        metadata = self._get_metadata()

        with self.managed_connection() as con:
            
            # Select any data in the database
            cur = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='systems'")
            
            # If no system in database
            if cur.fetchone()[0] == 0:
                
                # Update initial statements with properties to load
                init_systems_execute = init_systems[:]
                for prop_i in metadata.get('load_properties'):
                    if prop_i not in structure_properties_dtype.keys():
                        init_systems_execute[0] += f"{prop_i} BLOB,\n"
                init_systems_execute[0] = init_systems_execute[0][:-2] + ")"
                
                # Initialize data columns 
                for statement in init_systems_execute:
                    con.execute(statement)
                con.commit()
                
                self.version = VERSION
            
            # Else get information from database
            else:
                
                cur = con.execute(
                    'SELECT value FROM information WHERE name="version"')
                self.version = int(cur.fetchone()[0])
            
        # Check version compatibility
        if self.version > VERSION:
            raise IOError(
                f"Can not read newer version of the database format "
                f"(version {self.version}).")
        
        return

    def _reset(self):
        
        # Reset stored metadata dictionary
        self._metadata = {}
        return

    def _get(self, selection, **kwargs):
        
        # Get row of selection
        row = list(self.select(selection, **kwargs))
        
        # Check selection results
        if row is None:
            raise KeyError('no match')

        return row

    def parse_selection(self, selection, **kwargs):
        """
        Interpret the row selection
        """

        if selection is None or selection == '':
            cmps = []
        elif utils.is_integer(selection):
            cmps = [('id', '=', selection)]
        elif utils.is_integer_array(selection):
            cmps = [('id', '=', selection_i) for selection_i in selection]
        else:
            raise ValueError(
                f"Database selection '{selection}' is not a valid input!\n" +
                f"Provide either an index or list of indices.")

        return cmps

    @parallel_generator
    def select(
        self, 
        selection=None, 
        selection_filter=None, 
        **kwargs):
        """
        Select rows.

        Return AtomsRow iterator with results.  Selection is done
        using key-value pairs.
        
        Parameters
        ----------
        selection: (int, list(int))
            Row index or list of indices

        selection_filter: function
            A function that takes as input a row and returns True or False.

        Returns
        -------
            dict
                Returns entry of the selection.
        """
        
        # Check and interpret selection
        cmps = self.parse_selection(selection)
        
        # Iterate over selection
        #row = list(self._select(cmps))
        for row in self._select(cmps):
        
            # Apply potential reference data filter or not 
            if selection_filter is None or selection_filter(row):
                    
                yield row

    def encode(self, obj):
        return object_to_bytes(obj)
        
        
    def decode(self, txt):
        return bytes_to_object(txt)
        

    def blob(self, item):
        """
        Convert an item to blob/buffer object if it is an array.
        """

        if item is None:
            return None
        elif utils.is_integer(item):
            return item
        elif utils.is_numeric(item):
            return item
        
        if item.dtype == np.int64:
            item = item.astype(np.int32)
        if item.dtype == torch.int64:
            item = item.astype(torch.int32)
        if not np.little_endian:
            item = item.byteswap()
        return memoryview(np.ascontiguousarray(item))
        

    def deblob(self, buf, dtype=torch.float, shape=None):
        """
        Convert blob/buffer object to ndarray of correct dtype and shape.
        (without creating an extra view).
        """
        
        if buf is None:
            return None
        
        if len(buf) == 0:
            item = np.zeros(0, dtype)
        else:
            item = np.frombuffer(buf, dtype)
            if not np.little_endian:
                item = item.byteswap()
        
        if shape is not None:
            item = item.reshape(shape)
            
        return item

    @lock
    def _write(self, properties, row_id):
        
        # Reference data list
        columns = []
        values = []
        
        # Current datatime and User name
        columns += ['mtime', 'username']
        values += [time.ctime(), os.getenv('USER')]

        # Structural properties
        structure_values = []
        for prop_i, dtype_i in structure_properties_dtype.items():

            columns += [prop_i]
            if properties.get(prop_i) is None:
                values += [None]
            elif utils.is_array_like(properties.get(prop_i)):
                values += [self.blob(
                    np.array(properties.get(prop_i), dtype=dtype_i))]
            else:
                values += [dtype_i(properties.get(prop_i))]
        
        # Reference properties
        for prop_i in self.metadata.get('load_properties'):

            if prop_i not in structure_properties_dtype.keys():
                
                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(
                            properties.get(prop_i),
                            dtype=properties_numpy_dtype))]
                else:
                    values += [properties_numpy_dtype(properties.get(prop_i))]

        # Convert values to tuple
        columns = tuple(columns)
        values = tuple(values)
        
        # Add or update database values
        with self.managed_connection() as con:
            
            # Add values to database
            if row_id is None:
                
                # Get SQL cursor
                cur = con.cursor()
                
                # Add to database
                q = self.default + ', ' + ', '.join('?' * len(values))
                cur.execute(
                    f"INSERT INTO systems VALUES ({q})", values)
                row_id = self.get_last_id(cur)
                
            else:

                row_id = self._update(row_id, values=values, columns=columns)
                
        return row_id

    def update(self, row_id, properties):

        # Reference data list
        columns = []
        values = []

        # Current datatime and User name
        columns += ['mtime', 'username']
        values += [time.ctime(), os.getenv('USER')]

        # Structural properties
        for prop_i, dtype_i in structure_properties_dtype.items():

            if prop_i in properties:
                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(properties.get(prop_i), dtype=dtype_i))]
                else:
                    values += [dtype_i(properties.get(prop_i))]

        # Reference properties
        for prop_i in self.metadata.get('load_properties'):

            if (
                    prop_i in properties
                    and prop_i not in structure_properties_dtype
            ):
                
                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(
                            properties.get(prop_i),
                            dtype=properties_numpy_dtype))]
                else:
                    values += [properties_numpy_dtype(properties.get(prop_i))]
        
        # Convert values to tuple
        columns = tuple(columns)
        values = tuple(values)

        # Add or update database values
        row_id = self._update(row_id, values=values, columns=columns)

        return row_id

    @lock
    def _update(self, row_id, values=None, columns=None, properties=None):
        
        if values is None and properties is None:
            
            raise SyntaxError(
                "At least one input 'values' or 'properties' should "
                + "contain reference data!")
        
        elif values is None:
            
            row_id = self._write(properties, row_id)
            
        else:
        
            # Add or update database values
            with self.managed_connection() as con:
                
                # Get SQL cursor
                cur = con.cursor()
                
                # Update values in database
                q = ', '.join([f'{column:s} = ?' for column in columns])
                cur.execute(
                    f"UPDATE systems SET {q} WHERE id=?",
                    values + (row_id,))
            
        return row_id


    def get_last_id(self, cur):
        
        # Select last seqeuence  number from database
        cur.execute('SELECT seq FROM sqlite_sequence WHERE name="systems"')
        
        # Get next row id
        result = cur.fetchone()
        if result is not None:
            row_id = result[0]
            return row_id
        else:
            return 0
    
    
    def _select(self, cmps, verbose=False):
        
        sql, args = self.create_select_statement(cmps)
        metadata = self._get_metadata()
        with self.managed_connection() as con:
            
            # Execute SQL request
            cur = con.cursor()
            cur.execute(sql, args)
            
            for row in cur.fetchall():
                
                yield self.convert_row(row, metadata, verbose=verbose)
                
    
    def create_select_statement(self, cmps, what='systems.*'):
        """
        Translate selection to SQL statement.
        """

        tables = ['systems']
        where = []
        args = []
        
        # Prepare SQL statement
        for key, op, value in cmps:
            where.append('systems.{}{}?'.format(key, op))
            args.append(value)
        
        # Create SQL statement
        sql = "SELECT {} FROM\n  ".format(what) + ", ".join(tables)
        if where:
            sql += "\n  WHERE\n  " + " AND\n  ".join(where)
            
        return sql, args


    def convert_row(self, row, metadata, verbose=False):
        
        # Convert reference properties to a dictionary
        properties = {}
        
        # Add database information
        if verbose:
            
            # Get row id
            properties["row_id"] = row[0]
            
            # Get modification date
            properties["mtime"] = row[1]
            
            # Get username
            properties["user"] = row[2]
        
        # Structural properties
        Np = 3
        for prop_i, dtype_i in structure_properties_dtype.items():
            
            if row[Np] is None:
                properties[prop_i] = None
            elif isinstance(row[Np], bytes):
                properties[prop_i] = torch.from_numpy(
                    self.deblob(
                        row[Np], dtype=dtype_i, 
                        shape=structure_properties_shape[prop_i]).copy())
            else:
                properties[prop_i] = torch.reshape(
                    torch.tensor(row[Np], dtype=dtype_i),
                    structure_properties_shape[prop_i])
            
            #if prop_i == "positions":
                #properties[prop_i] = properties[prop_i].reshape(-1, 3)
                
            Np += 1
            
            
        for prop_i in metadata.get('load_properties'):
            
            if prop_i not in structure_properties_dtype.keys():
                
                if row[Np] is None:
                    properties[prop_i] = None
                elif isinstance(row[Np], bytes):
                    properties[prop_i] = torch.from_numpy(
                        self.deblob(
                            row[Np], dtype=properties_numpy_dtype).copy()
                        ).to(properties_torch_dtype)
                else:
                    properties[prop_i] = torch.tensor(
                        row[Np], dtype=properties_torch_dtype)
                
                if prop_i == "forces":
                    properties[prop_i] = properties[prop_i].reshape(-1, 3)
            
                
                Np += 1
        
        return properties
    
    @parallel_function
    def _count(self, selection, **kwargs):

        # Check selection
        cmps = self.parse_selection(selection)

        sql, args = self.create_select_statement(cmps, what='COUNT(*)')

        with self.managed_connection() as con:
            cur = con.cursor()
            try:
                cur.execute(sql, args)
                return cur.fetchone()[0]
            except sqlite3.OperationalError:
                return 0

    @parallel_function
    @lock
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
            
        self.vacuum()


    def _delete(self, row_ids):
        
        with self.managed_connection() as con:
            cur = con.cursor()
            selection = ', '.join([str(row_id) for row_id in row_ids])
            for table in all_tables[::-1]:
                cur.execute(
                    f"DELETE FROM {table} WHERE id in ({selection});")
    
    
    def vacuum(self):
        """
        Execute SQL command 'Vacuum' (?)
        """
        
        with self.managed_connection() as con:
            con.commit()
            con.cursor().execute("VACUUM")


    @property
    def metadata(self):
        return self._get_metadata()
