"""

Utils module for Asparagus.

Contains functions for checking the input, the units, and the data types.
Additionally, it manage the creation of checkpoints and writting to
tensorboard.

Legacy function should make transformation of modules in tensorflow 1 to
asparagus.

"""

from .check_dtype import(
    is_None, is_string, is_bool, is_boolean, 
    is_numeric, is_integer, is_dictionary,
    is_array_like, is_numeric_array, is_integer_array, is_string_array,
    is_string_array_inhomogeneous, is_bool_array, is_boolean_array,
    is_None_array, is_object, is_callable, is_ase_atoms, is_ase_atoms_array,
    is_grad_enabled, in_cuda, is_attached
)

from .check_config import(
    check_input_args, check_input_dtype, check_device_option,
    check_dtype_option, check_property_label,
    merge_dictionaries, merge_dictionary_lists,
    get_input_args, get_function_location,
    get_default_args, get_dtype_args
)

from .check_units import(
    check_units
)

from .output import(
    set_logger, get_header, print_ProgressBar
)

from .functions import(
    segment_sum, softplus_inverse, gather_nd, detach_tensor, flatten_array_like
)

from .data_ase import(
    chemical_symbols, atomic_names, atomic_masses, atomic_numbers
)
