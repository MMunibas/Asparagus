import inspect
import platform
from typing import Optional, Union, List, Dict, Tuple, Iterator, Callable, Any

from asparagus import utils
from asparagus import settings

# --------------- ** Checking input parameter ** ---------------


def check_input_args(
    instance: Optional[object] = None,
    argitems: Optional[Iterator] = None,
    argsskip: Optional[List[str]] = None,
    check_default: Optional[Dict] = None,
    check_dtype: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Iterate over arg, item pair, eventually check for default and dtype,
    and set as class variable of instance

    Parameters:
    -----------
    instance: object, optional, default None
        Class instance to set arg, item pair as class variable. If None,
        skip.
    argitems: iterator, optional, default None
        Iterator for arg, item pairs. If None, skip.
    argskipt: list(str), optional, default None
        List of arguments to skip.
    check_default: dict, optional, default None
        Default argument parameter dictionary.
    check_dtype: dict, optional, default None
        Default argument data type dictionary.

    Return:
    -------
    dict[str, any]
        Updated input arguments

    """

    # Return empty dictionary if no arg, item pair iterator is defined
    if argitems is None:
        return {}
    else:
        dict_update = {}

    # Check arguments to skip
    default_argsskip = ['self', 'kwargs', '__class__']
    if argsskip is None:
        argsskip = default_argsskip
    else:
        argsskip = default_argsskip + list(argsskip)
    argsskip.append('default_args')

    # Iterate over arg, item pairs
    for arg, item in argitems.items():

        # Skip exceptions
        if arg in argsskip:
            continue

        # Check if input parameter is None, if so take default value
        if check_default is not None and item is None:
            if arg in check_default:
                item = check_default[arg]

        # Check datatype of defined arguments
        if check_dtype is not None and arg in check_dtype:
            _ = utils.check_input_dtype(
                arg, item, check_dtype, raise_error=True)

        # Append arg, item pair to update dictionary
        dict_update[arg] = item

        # Set item as class parameter arg to instance
        if instance is not None:
            setattr(instance, arg, item)

    return dict_update


def check_input_dtype(
    arg: str,
    item: Any,
    dtypes_args: Dict[str, List[Callable]],
    raise_error: Optional[bool] = True,
    return_info: Optional[bool] = False,
) -> bool:
    """
    Check it the item (not None) of arg(ument) matchs the expectation
    in dtypes_args.

    """

    if (
        arg in dtypes_args.keys()
        and len(dtypes_args[arg])
        and item is not None
    ):
        matches, dtype_expectation = [], []
        for is_dtype in dtypes_args[arg]:
            match, dtype_input, dtype_expected = is_dtype(item, verbose=True)
            matches.append(match)
            dtype_expectation.append(dtype_expected)
        if not any(matches) and raise_error:
            raise ValueError(
                f"Argument '{arg:s}' has wrong dtype!\n" +
                f"Input: {dtype_input}\n" +
                f"Expected: {dtype_expectation}")

        if return_info:
            return any(matches), dtype_input, dtype_expectation
        else:
            return any(matches)

    else:

        if return_info:
            return False, None, None
        else:
            return False


def check_device_option(
    device: str,
    config: object,
):
    """
    Check and select device input.

    Parameters
    ----------
    device: str
        Device label
    config: settings.Configuration
        Asparagus configuration object for default options and conversion

    """

    # If no device options are given, take default device.
    if device is None and config.get('device') is None:
        return settings._default_device
    # If no device is defined, take device from config.
    elif device is None:
        return config.get('device')
    # If device is given, check if conversion is needed
    elif utils.is_string(device):
        return device

    raise SyntaxError(
        f"Torch device input '{device}' is of invalid data type!")


def check_dtype_option(
    dtype: Any,
    config: object,
) -> Callable:
    """
    Check and select dtype input and convert eventually to correct dtype class.

    Parameters
    ----------
    dtype: any
        Data type label or class to check
    config: settings.Configuration
        Asparagus configuration object for default options and conversion

    """

    # If no dtype options are given, take default dtype.
    if dtype is None and config.get('dtype') is None:
        return settings._default_dtype
    # If no dtype is defined, take converted dtype from config.
    elif dtype is None:
        return config.get('dtype')
    # If dtype is given, check if conversion is needed
    elif utils.is_string(dtype):
        return config.convert_dtype(dtype, 'read')
    else:
        return dtype

# --------------- ** Checking property labels ** ---------------


def check_property_label(
    property_label,
    valid_property_labels: Optional[List[str]] = None,
    alt_property_labels: Optional[Dict[str, List[str]]] = None,
    return_modified: Optional[bool] = True,
) -> bool:
    """
    Validate the property label by comparing with valid property labels in
    'valid_property_label' or compare with alternative labels in
    'alt_property_labels'. If valid or found in 'alt_properties', the valid
    lower case form is returned with bool for match and if modified.

    Parameters
    ----------
    property_label : str
        Property labels to be checked.
    valid_property_labels : list(str), optional, default None
        List of valid property labels. If not defined, valid property labels
        are taken from settings._valid_properties.
    alt_property_labels: dict, optional, default None
        Dictionary with alternative property labels as keys and valid property
        labels as values. If not defined, no check for alternatively spelled
        properties is done.
    return_modified : bool, optional, default True
        Return if property label was modified.

    """

    # Check if property label is valid
    if valid_property_labels is None:
        valid_property_labels = settings._valid_properties
    if property_label.lower() in valid_property_labels:
        # If already lower case or not
        if property_label.lower() == property_label:
            if return_modified:
                return True, False, property_label.lower()
            else:
                return True
        else:
            if return_modified:
                return True, True, property_label.lower()
            else:
                return True

    # Check if a valid alternative can be found for property label
    if alt_property_labels is None:
        alt_property_labels = settings._alt_property_labels
    for key, items in alt_property_labels.items():
        if utils.is_string(items):
            items_lower = [items.lower()]
        else:
            items_lower = [item.lower() for item in items]
        if property_label.lower() in items_lower:
            if return_modified:
                return True, True, key.lower()
            else:
                return True

    # Property label is not valid nor is an alternative found.
    if return_modified:
        return False, False, property_label
    else:
        return False

# --------------- ** Combine dictionaries ** ---------------


def merge_dictionaries(
    dict_old: Dict[str, Any],
    dict_new: Dict[str, Any],
    keep: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Merge keys and items of both dictionaries. If 'keep' is False, update
    key in dict_old with item of dict_new.
    """

    # Check dictionaries
    if dict_old is None and dict_new is None:
        return {}
    if dict_old is None:
        return dict_new
    if dict_new is None:
        return dict_old

    # Iterate over keys
    for key, item in dict_old.items():
        if key in dict_new:
            if keep:
                dict_new[key] = item
        else:
            dict_new[key] = item

    return dict_new


def merge_dictionary_lists(
    dictA: Dict[str, List[str]],
    dictB: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Combine two dictionaries lists and check for conflicts in item lists.
    If an item in the lists reüeats in dictA,  the assignment of 'dictA' is
    kept.

    """

    # Combined dictionary
    dictC = {}

    # Observed items in dictA and dictB
    observed_items = []

    # Iterate over dictA
    for keyA, itemsA in dictA.items():
        dictC[keyA] = itemsA
        for itemA in itemsA:
            if itemA not in observed_items:
                dictC[keyA].append(itemA)
                observed_items.append(itemA)

    # Iterate over dictB
    for keyB, itemsB in dictB.items():
        if keyB not in dictC.keys():
            dictC[keyB] = []
        for itemB in itemsB:
            # If itemB already appeared, raise error or drop itemB
            if itemB not in observed_items:
                dictC[keyB].append(itemB)
                observed_items.append(itemB)

    return dictC

# --------------- ** Get Function Input Arguments ** ---------------


def get_input_args():
    """
    Get input arguments of the function from where this function is called:
        inspect.stack()[0] <- this function
        inspect.stack()[1] <- the function this one is called from
        inspect.stack()[>1] <- previous functions
        (see http://kbyanc.blogspot.com/2007/07/python-aggregating-function
        -arguments.html)

    Returns
    -------
    dict
        Input argument and item dictionary of the function this function
        is called from.
    """

    args_info, args_name, kwargs_name, args_dict = inspect.getargvalues(
        inspect.stack()[1][0])

    return args_dict


def get_function_location(
    module_name: Optional[str] = 'asparagus'
):
    """
    Get function location from inspect.stack.

    Returns
    -------
    str
        Function location

    """

    # Detect OS to get split string
    if 'windows' in platform.system().lower():
        split_string = '\\'
    else:
        split_string = '/'
    func_files = inspect.stack()[1][0].f_code.co_filename.split(split_string)
    func_module_files = func_files[-(func_files[::-1].index(module_name) + 1):]
    func_path = "".join([file_i + "." for file_i in func_module_files])[:-3]
    func_name = inspect.stack()[1][0].f_code.co_name + '()'

    func_location = func_path + func_name

    return func_location

# --------------- ** Combine Default Dictionaries ** ---------------


def get_default_args(
    self_class: Callable,
    self_module: Callable,
) -> Dict[str, Any]:
    """
    Combine available default argument dictionaries. In case of conflicts, the
    priority is from top to bottom: self_class, self_module, settings.

    """

    # Get default argument dictionary
    default_args = settings._default_args

    # Add and overwrite with module default arguments
    if hasattr(self_module, '_default_args'):
        default_args.update(self_module._default_args)

    # Add and overwrite with class default arguments
    if hasattr(self_class, '_default_args'):
        default_args.update(self_class._default_args)

    return default_args


def get_dtype_args(
    self_class: Callable,
    self_module: Callable,
) -> Dict[str, Callable]:
    """
    Combine available argument data type dictionaries. In case of conflicts,
    the priority is from top to bottom: self_class, self_module, settings.

    """

    # Get default argument dictionary
    dtype_args = settings._dtypes_args

    # Add and overwrite with module arguments data types
    if self_module is not None and hasattr(self_module, '_dtypes_args'):
        dtype_args.update(self_module._dtypes_args)

    # Add and overwrite with class arguments data types
    if self_class is not None and hasattr(self_class, '_dtypes_args'):
        dtype_args.update(self_class._dtypes_args)

    return dtype_args
