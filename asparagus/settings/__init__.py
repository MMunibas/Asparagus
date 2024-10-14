"""

Basic settings for the Asparagus package and global variable managment.

Definition of default variables and functions to set them, additionally it
handles the configuration file.

"""

from .config import (
    get_config, Configuration
)

from .default import (
    _default_args, _default_calculator_model, _default_device, _default_dtype
)

from .dtypes import (
    _dtypes_args, _dtype_library
)

from .properties import (
    _default_property_labels, _valid_properties, _alt_property_labels,
    _ase_units, _default_units, _related_unit_properties
)
