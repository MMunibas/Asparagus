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
    check_data_format, DataReader
)

from .datastats import (
    compute_property_scaling, compute_system_property_scaling,
    compute_atomic_property_scaling, compute_atomic_property_sum_scaling,
    compute_atomic_energies_scaling
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
