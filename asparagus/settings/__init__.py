"""
Basic settings for the Asparagus package.

Definition of global variables and functions to set them, additionally it 
handles the configuration file.

"""

from .default import (
    _default_args, _default_calculator_model, _default_device, _default_dtype
)

from .dtypes import (
    _dtypes_args, _dtype_library
)

from .properties import (
    _default_property_labels, _valid_properties, _alt_property_labels, 
    _default_units, _related_unit_properties
)

from .config import (
    Configuration, get_config
)
