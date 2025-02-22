import os
import time
import json
import sqlite3
import functools
from typing import Optional, Union, List, Dict, Tuple, Callable, Any
from contextlib import contextmanager

from ase.parallel import DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock
from ase.io.jsonio import create_ase_object

import numpy as np

import torch

from asparagus import data
from asparagus import utils

__all__ = ['connect', 'DataBase_SQLite3']

# Current SQLite3 database version
VERSION = 3

all_tables = ['systems']

# Initial SQL statement lines
init_systems_version = {
    0: [
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
            """
        ],
    2: [
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
            """
        ],

    3: [
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
            atom_types BLOB,
            fragments BLOB,
            """
        ],
    }


init_information = [
    """CREATE TABLE information (
    name TEXT,
    value TEXT)""",
    "INSERT INTO information VALUES ('version', '{}')".format(VERSION)]

# Structural property labels and dtypes
structure_properties_dtype_version = {
    0: {
            'atoms_number':     np.int32,
            'atomic_numbers':   np.int32,
            'positions':        np.float32,
            'charge':           np.float32,
            'cell':             np.float32,
            'pbc':              np.bool_,
            'idx_i':            np.int32,
            'idx_j':            np.int32,
            'pbc_offset':       np.float32,
        },
    2: {
            'atoms_number':     np.int32,
            'atomic_numbers':   np.int32,
            'positions':        np.float32,
            'charge':           np.float32,
            'cell':             np.float32,
            'pbc':              np.bool_,
        },
    3: {
            'atoms_number':     np.int32,
            'atomic_numbers':   np.int32,
            'positions':        np.float32,
            'charge':           np.float32,
            'cell':             np.float32,
            'pbc':              np.bool_,
            'atom_types':       'U4',
            'fragments':        np.int32,
        },
}

# Structural property labels and array shape
structure_properties_shape_version = {
    0: {
            'atoms_number':     (-1,),
            'atomic_numbers':   (-1,),
            'positions':        (-1, 3,),
            'charge':           (-1,),
            'cell':             (-1, 3,),
            'pbc':              (-1, 3,),
            'idx_i':            (-1,),
            'idx_j':            (-1,),
            'pbc_offset':       (-1, 3,),
    },
    2: {
            'atoms_number':     (-1,),
            'atomic_numbers':   (-1,),
            'positions':        (-1, 3,),
            'charge':           (-1,),
            'cell':             (-1, 3,),
            'pbc':              (-1, 3,),
    },
    3: {
            'atoms_number':     (-1,),
            'atomic_numbers':   (-1,),
            'positions':        (-1, 3,),
            'charge':           (-1,),
            'cell':             (-1, 3,),
            'pbc':              (-1, 3,),
            'atom_types':       (-1,),
            'fragments':        (-1,),
    },
}
reference_properties_shape = {
    # 'energy':               (-1,),
    # 'atomic_energies':      (-1,),
    'forces':               (-1, 3,),
    # 'hessian':              (-1,),
    # 'atomic_charges':       (-1,),
    # 'dipole':               (3),
    # 'atomic_dipoles':       (-1,),
    'polarizability':       (3, 3,),
    # 'fragment_energies':    (-1,),
    # 'interaction_energy':   (-1,),
    }


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
    data.Database_SQLite3
        SQL database interface object

    """
    return DataBase_SQLite3(data_file, lock_file)


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

    # Structural and reference property dtypes
    properties_numpy_dtype = np.float64
    properties_torch_dtype = torch.float64

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

        """

        # Inherit from DataBase base class
        super().__init__(data_file)

        # Prepare data locker
        self.lock_file = self.data_file + '.lock'
        if lock_file and utils.is_string(self.data_file):
            self.lock = Lock(self.data_file + '.lock', world=DummyMPI())
        else:
            self.lock = None

        with self.managed_connection() as con:

            # Select any data in the database
            cur = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='systems'")
            data_fetched = cur.fetchone()[0]

            # Get and assign version number
            if data_fetched == 0:
                self.version = VERSION
            else:
                cur = con.execute(
                    'SELECT value FROM information WHERE name="version"')
                self.version = int(cur.fetchone()[0])

            # Initialize version compatible parameters. If not compatible
            # parameter are defined, take the latest version smaller than the
            # requested one
            if self.version in init_systems_version:
                self.init_systems_execute = [
                    init_system.strip()
                    for init_system in init_systems_version[self.version]]
            else:
                version = self.latest_version(init_systems_version)
                self.init_systems_execute = (init_systems_version[version][:])
            if self.version in structure_properties_dtype_version:
                self.structure_properties_dtype = (
                    structure_properties_dtype_version[self.version])
            else:
                version = self.latest_version(
                    structure_properties_dtype_version)
                self.structure_properties_dtype = (
                    structure_properties_dtype_version[version])
            if self.version in structure_properties_shape_version:
                self.structure_properties_shape = (
                    structure_properties_shape_version[self.version])
            else:
                version = self.latest_version(
                    structure_properties_shape_version)
                self.structure_properties_shape = (
                    structure_properties_shape_version[version])

        # Check version compatibility
        if self.version > VERSION:
            raise IOError(
                f"Can not read newer version of the database format "
                f"(version {self.version}).")

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
        return

    def close(self):
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
        self.connection = None
        return

    @contextmanager
    def managed_connection(
        self,
        commit_frequency: Optional[int] = 5000,
    ):
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
    def _set_metadata(
        self,
        metadata: Dict[str, Any],
    ):

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

        # Initialize data system
        self._init_systems()

        return

    def _get_metadata(self) -> Dict[str, Any]:

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

    def latest_version(self, parameter_version):
        version_available = [
            int(version) for version in parameter_version.keys()
            if int(version) <= self.version]
        return max(version_available)

    def _init_systems(self):

        # Get metadata
        metadata = self._get_metadata()

        with self.managed_connection() as con:

            # Select any data in the database
            cur = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='systems'")
            data_fetched = cur.fetchone()[0]

            # If no system in database, initialize
            if data_fetched == 0:

                # Update initial statements with properties to load
                for prop_i in metadata.get('load_properties'):
                    if prop_i not in self.structure_properties_dtype.keys():
                        self.init_systems_execute[0] += f"{prop_i} BLOB,\n"
                self.init_systems_execute[0] = (
                    self.init_systems_execute[0].strip()[:-1] + ")")

                # Initialize data columns
                for statement in self.init_systems_execute:
                    con.execute(statement)
                con.commit()

        # # Check version compatibility
        # if self.version > VERSION:
        #     raise IOError(
        #         f"Can not read newer version of the database format "
        #         f"(version {self.version}).")

        return

    def _reset(self):

        # Reset stored metadata dictionary
        self._metadata = {}
        return

    def _get(
        self,
        selection: Union[int, List[int]],
        **kwargs,
    ) -> Dict[str, Any]:

        # Get row of selection
        row = list(self.select(selection, **kwargs))

        # Check selection results
        if row is None:
            raise KeyError('no match')

        return row

    def parse_selection(
        self,
        selection: Union[int, List[int]],
        **kwargs,
    ) -> List[Tuple[str]]:
        """
        Interpret the row selection
        """

        if selection is None or selection == '':
            cmps = []
        elif utils.is_integer(selection):
            cmps = [('id', '=', int(selection))]
        elif utils.is_integer_array(selection):
            cmps = [('id', '=', int(selection_i)) for selection_i in selection]
        else:
            raise ValueError(
                f"Database selection '{selection}' is not a valid input!\n" +
                "Provide either an index or list of indices.")

        return cmps

    @parallel_generator
    def select(
        self,
        selection: Optional[Union[int, List[int]]] = None,
        selection_filter: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Select rows.

        Return AtomsRow iterator with results.  Selection is done
        using key-value pairs.

        Parameters
        ----------
        selection: (int, list(int)), optional, default None
            Row index or list of indices
        selection_filter: callable, optional, default None
            A function that takes as input a row and returns True or False.

        Returns
        -------
        dict
            Returns entry of the selection.
        """

        # Check and interpret selection
        cmps = self.parse_selection(selection)

        # Iterate over selection
        for row in self._select(cmps):

            # Apply potential reference data filter or not
            if selection_filter is None or selection_filter(row):

                yield row

    def encode(
        self,
        obj: Any,
    ) -> Any:
        return object_to_bytes(obj)

    def decode(
        self,
        txt: str,
    ) -> Any:
        return bytes_to_object(txt)

    def blob(
        self,
        item: Any,
    ) -> Any:
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

    def deblob(
        self,
        buf,
        dtype=None,
        shape=None
    ) -> Any:
        """
        Convert blob/buffer object to ndarray of correct dtype and shape.
        (without creating an extra view).
        """

        if buf is None:
            return None

        if dtype is None:
            dtype = self.properties_numpy_dtype

        if len(buf):
            item = np.frombuffer(buf, dtype)
            if not np.little_endian:
                item = item.byteswap()
        else:
            item = np.zeros(0, dtype)

        if shape is not None:
            item = item.reshape(shape)

        return item

    @lock
    def _write(
        self,
        properties: Dict[str, Any],
        row_id: int,
    ) -> int:

        # Reference data list
        columns = []
        values = []

        # Current datatime and User name
        columns += ['mtime', 'username']
        values += [time.ctime(), os.getenv('USER')]

        # Structural properties
        for prop_i, dtype_i in self.structure_properties_dtype.items():

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

            if prop_i not in self.structure_properties_dtype:

                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(
                            properties.get(prop_i),
                            dtype=self.properties_numpy_dtype))]
                else:
                    values += [self.properties_numpy_dtype(
                        properties.get(prop_i))]

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

    def update(
        self,
        row_id: int,
        properties: Dict[str, Any],
    ) -> int:

        # Reference data list
        columns = []
        values = []

        # Current datatime and User name
        columns += ['mtime', 'username']
        values += [time.ctime(), os.getenv('USER')]

        # Structural properties
        for prop_i, dtype_i in self.structure_properties_dtype.items():

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
                    and prop_i not in self.structure_properties_dtype
            ):

                columns += [prop_i]
                if properties.get(prop_i) is None:
                    values += [None]
                elif utils.is_array_like(properties.get(prop_i)):
                    values += [self.blob(
                        np.array(
                            properties.get(prop_i),
                            dtype=self.properties_numpy_dtype))]
                else:
                    values += [self.properties_numpy_dtype(
                        properties.get(prop_i))]

        # Convert values to tuple
        columns = tuple(columns)
        values = tuple(values)

        # Add or update database values
        row_id = self._update(row_id, values=values, columns=columns)

        return row_id

    @lock
    def _update(
        self,
        row_id: int,
        values: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        properties: Optional[Any] = None,
    ) -> int:

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

    def get_last_id(
        self,
        cur: object,
    ) -> int:

        # Select last seqeuence  number from database
        cur.execute('SELECT seq FROM sqlite_sequence WHERE name="systems"')

        # Get next row id
        result = cur.fetchone()
        if result is not None:
            row_id = result[0]
            return row_id
        else:
            return 0

    def _select(
        self,
        cmps: List[Tuple[str]],
        verbose=False,
    ) -> Dict[str, Any]:

        sql, args = self.create_select_statement(cmps)
        metadata = self._get_metadata()
        with self.managed_connection() as con:

            # Execute SQL request
            cur = con.cursor()
            cur.execute(sql, args)

            for row in cur.fetchall():

                yield self.convert_row(row, metadata, verbose=verbose)

    def create_select_statement(
        self,
        cmps: List[Tuple[str]],
        what: Optional[str] = 'systems.*',
    ) -> (str, List[int]):
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

    def convert_row(
        self,
        row: List[Any],
        metadata: Dict[str, Any],
        verbose: Optional[bool] = False,
    ) -> Dict[str, Any]:

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
        for prop_i, dtype_i in self.structure_properties_dtype.items():
            if row[Np] is None:
                properties[prop_i] = None
            elif isinstance(row[Np], bytes):
                properties[prop_i] = torch.from_numpy(
                    self.deblob(
                        row[Np],
                        dtype=dtype_i,
                        shape=self.structure_properties_shape[prop_i]
                    ).copy())
            else:
                properties[prop_i] = torch.reshape(
                    torch.tensor(row[Np], dtype=dtype_i),
                    self.structure_properties_shape[prop_i])

            Np += 1

        for prop_i in metadata.get('load_properties'):

            if prop_i in self.structure_properties_dtype:
                continue

            if row[Np] is None:
                properties[prop_i] = None
            elif isinstance(row[Np], bytes):
                properties[prop_i] = torch.from_numpy(
                    self.deblob(
                        row[Np], dtype=self.properties_numpy_dtype).copy()
                    ).to(self.properties_torch_dtype)
            else:
                properties[prop_i] = torch.tensor(
                    row[Np], dtype=self.properties_torch_dtype)

            if prop_i in reference_properties_shape:
                properties[prop_i] = torch.reshape(
                    properties[prop_i],
                    reference_properties_shape[prop_i])
            elif 'std_' in prop_i and prop_i[4:] in reference_properties_shape:
                properties[prop_i] = torch.reshape(
                    properties[prop_i],
                    reference_properties_shape[prop_i[4:]])

            Np += 1

        return properties

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
    def delete(
        self,
        row_ids: Union[int, List[int]],
    ):
        """
        Delete rows.

        Parameters
        ----------
        row_ids: int or list of int
            Row index or list of indices to delete
        """

        if not len(row_ids):
            return

        self._delete(row_ids)
        self.vacuum()

        return

    def _delete(
        self,
        row_ids: Union[int, List[int]]
    ):

        with self.managed_connection() as con:
            cur = con.cursor()
            selection = ', '.join([str(row_id) for row_id in row_ids])
            for table in all_tables[::-1]:
                cur.execute(
                    f"DELETE FROM {table} WHERE id in ({selection});")

        return

    def vacuum(self):
        """
        Execute SQL command 'Vacuum' (?)
        """

        with self.managed_connection() as con:
            con.commit()
            con.cursor().execute("VACUUM")

        return

    def _delete_file(self):
        """
        Delete database file
        """
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)
        return

    @property
    def metadata(self):
        return self._get_metadata()
