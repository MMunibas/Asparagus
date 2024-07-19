"""
This data directory contains all modules for the data management in Asparagus.

"""

from .datacontainer import (
    DataContainer
)
from .dataset import (
    DataSet, DataSubSet
)
from .datareader import (
    DataReader, check_data_format
)
from .database import (
    DataBase, connect, get_connect, get_metadata
)
from .database_sqlite3 import (
    DataBase_SQLite3, lock, object_to_bytes, bytes_to_object
)
from .database_npz import (
    DataBase_npz
)
from .database_hdf5 import (
    DataBase_hdf5
)
from .dataloader import (
    DataLoader
)

# Default arguments for data modules
import numpy as np
_default_args = {
    'data_file':                    'data.db',
    'data_file_format':             'sql',
    'data_source':                  [],
    'data_source_format':           [],
    'data_unit_positions':          'Ang',
    'data_load_properties':         ['energy', 'forces', 'dipole'],
    'data_unit_properties':         {'energy': 'eV',
                                     'forces': 'eV/Ang',
                                     'dipole': 'eAng'},
    'data_alt_property_labels':     {},
    'data_num_train':               0.8,
    'data_num_valid':               0.1,
    'data_num_test':                None,
    'data_train_batch_size':        32,
    'data_valid_batch_size':        32,
    'data_test_batch_size':         32,
    'data_num_workers':             1,
    'data_overwrite':               False,
    'data_seed':                    np.random.randint(1E6),
    }
    
# Expected data types of input variables
from .. import utils
_dtypes_args = {
    'data_file':                    [utils.is_string],
    'data_file_format':             [utils.is_string],
    'data_source':                  [utils.is_string, utils.is_string_array],
    'data_source_format':           [utils.is_string, utils.is_string_array],
    'data_unit_positions':          [utils.is_string],
    'data_load_properties':         [utils.is_array_like],
    'data_unit_properties':         [utils.is_dictionary],
    'data_alt_property_labels':     [utils.is_dictionary],
    'data_num_train':               [utils.is_numeric],
    'data_num_valid':               [utils.is_numeric],
    'data_num_test':                [utils.is_numeric, utils.is_None],
    'data_train_batch_size':        [utils.is_integer],
    'data_val_batch_size':          [utils.is_integer],
    'data_test_batch_size':         [utils.is_integer],
    'data_num_workers':             [utils.is_integer],
    'data_overwrite':               [utils.is_bool],
    'data_seed':                    [utils.is_numeric],
    }
