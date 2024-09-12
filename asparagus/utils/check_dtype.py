import numpy as np

from typing import Optional, Callable, Any

import torch
import ase

from asparagus import utils

# --------------- ** Checking data types ** ---------------

# Numeric data types
dint_all = (
    int, np.int16, np.int32, np.int64,
    torch.int, torch.int16, torch.int32, torch.int64)
dflt_all = (
    float, np.float16, np.float32, np.float64,
    torch.float, torch.float16, torch.float32, torch.float64)
dnum_all = dint_all + dflt_all

# Array data types
darr_all = (tuple, list, np.ndarray, torch.Tensor)
darr_all_len = (tuple, list)
darr_all_shape = (np.ndarray, torch.Tensor)

# Bool data types
dbool_all = (bool, np.bool_)
dbool_all_shape = (bool, np.bool_, torch.bool)


def is_None(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is None

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """

    if isinstance(x, np.ndarray) and not x.shape:
        result = x == None
        xtype = x.dtype
    else:
        result = x is None
        xtype = type(x)
        
    if verbose:
        return result, xtype, "NoneType"
    else:
        return result


def is_none(x: Any, verbose: Optional[bool] = False) -> bool:
    return is_None(x, verbose=verbose)


def is_string(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is a string

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """
    
    if isinstance(x, np.ndarray) and not x.shape:
        result = x.dtype.char == 'U'
        xtype = x.dtype
    else:
        result = isinstance(x, str)
        xtype = type(x)

    if verbose:
        return result, xtype, "str"
    else:
        return result


def is_bool(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is a bool

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """
    
    if isinstance(x, (torch.Tensor, np.ndarray)) and not x.shape:
        result = (x.dtype in dbool_all_shape)
        xtype = x.dtype
    else:
        result = isinstance(x, dbool_all)
        xtype = type(x)

    if verbose:
        return result, xtype, "bool"
    else:
        return result


def is_boolean(x: Any, verbose: Optional[bool] = False) -> bool:
    return is_bool(x, verbose=verbose)


def is_numeric(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is a numeric type (int or float)

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """

    if isinstance(x, (torch.Tensor, np.ndarray)) and not x.shape:
        result = (x.dtype in dnum_all)
        xtype = x.dtype
    else:
        result = type(x) in dnum_all and not is_bool(x)
        xtype = type(x)

    if verbose:
        return (result, xtype, str(dnum_all))
    else:
        return result


def is_integer(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is an integer type

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """

    if isinstance(x, (torch.Tensor, np.ndarray)) and not x.shape:
        result = (x.dtype in dint_all)
        xtype = x.dtype
    else:
        result = type(x) in dint_all and not is_bool(x)
        xtype = type(x)

    if verbose:
        return (result, xtype, str(dint_all))
    else:
        return result


def is_callable(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is a callable object

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """
    if verbose:
        return isinstance(x, Callable), type(x), "callable object"
    else:
        return isinstance(x, Callable)


def is_object(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is an object

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """
    if verbose:
        return isinstance(x, object), type(x), "object"
    else:
        return isinstance(x, object)


def is_dictionary(x: Any, verbose=False):
    """
    Check if the input is a dictionary

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.
    """
    if verbose:
        return isinstance(x, dict), type(x), "dict"
    else:
        return isinstance(x, dict)


def is_array_like(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    """
    Check if the input is an array-like object.

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    inhomogeneity: bool, optional, default False
        If True, return positive match for inhomogeneous array like variables
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """
    
    if isinstance(x, darr_all_shape) and x.shape:
        result = True
    elif isinstance(x, darr_all_len) and len(x):
        result = True
    else:
        result = False
    xtype = type(x)
    
    if verbose:
        return (result, xtype, str(darr_all))
    else:
        return result


def is_numeric_array(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    """
    Check if the input is a numeric array-like object

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    inhomogeneity: bool, optional, default False
        If True, return positive match for inhomogeneous array like variables
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """

    if is_array_like(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        try:
            x_arr = np.asarray(x)
            if (x_arr.dtype in dnum_all):
                result = True
                xtype = f"({type(x)})[{x_arr.dtype}]"
            else:
                result = False
                xtype = f"({type(x)})[{x_arr.dtype}]"
        except (ValueError, TypeError):
            result = False
            xtype = f"({type(x)})[{type(x[0])}]"
        pass
    else:
        result = False
        xtype = f"{type(x)}"

    if verbose:
        return (result, xtype, f"({darr_all})[{dnum_all}]")
    else:
        return result


def is_integer_array(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    """
    Check if the input is an integer array-like object

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    inhomogeneity: bool, optional, default False
        If True, return positive match for inhomogeneous array like variables
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """

    if is_array_like(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        try:
            x_arr = np.asarray(x)
            if (x_arr.dtype in dint_all):
                result = True
                xtype = f"({type(x)})[{x_arr.dtype}]"
            else:
                result = False
                xtype = f"({type(x)})[{x_arr.dtype}]"
        except (ValueError, TypeError):
            result = False
            xtype = f"({type(x)})[{type(x[0])}]"
        pass
    else:
        result = False
        xtype = f"{type(x)}"

    if verbose:
        return (result, xtype, f"({darr_all})[{dint_all}]")
    else:
        return result


def is_string_array(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    """
    Check if the input can be a string array-like object

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    inhomogeneity: bool, optional, default False
        If True, return positive match for inhomogeneous array like variables
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """

    if is_array_like(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        try:
            x_arr = np.asarray(x)
            if x_arr.dtype.char == 'U':
                result = True
                xtype = f"({type(x)})[{x_arr.dtype}]"
            else:
                result = False
                xtype = f"({type(x)})[{x_arr.dtype}]"
        except (ValueError, TypeError):
            result = False
            xtype = f"({type(x)})[{type(x[0])}]"
        pass
    else:
        result = False
        xtype = f"{type(x)}"

    if verbose:
        return (result, xtype, f"({darr_all})[str]")
    else:
        return result


def is_string_array_inhomogeneous(
    x: Any,
    verbose: Optional[bool] = False,
) -> bool:
    return is_string_array(x, inhomogeneity=True, verbose=verbose)


def is_bool_array(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    """
    Check if the input is a boolean array-like object

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    inhomogeneity: bool, optional, default False
        If True, return positive match for inhomogeneous array like variables
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """
    
    if is_array_like(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        try:
            x_arr = np.asarray(x)
            if (x_arr.dtype in dbool_all):
                result = True
                xtype = f"({type(x)})[{x_arr.dtype}]"
            else:
                result = False
                xtype = f"({type(x)})[{x_arr.dtype}]"
        except (ValueError, TypeError):
            result = False
            xtype = f"({type(x)})[{type(x[0])}]"
        pass
    else:
        result = False
        xtype = f"{type(x)}"

    if verbose:
        return (result, xtype, f"({darr_all})[{dbool_all}]")
    else:
        return result


def is_boolean_array(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    return is_bool_array(x, inhomogeneity=inhomogeneity, verbose=verbose)


def is_None_array(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    """
    Check if the input is a None array-like object.

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    inhomogeneity: bool, optional, default False
        If True, return positive match for inhomogeneous array like variables
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match dtype, else False.

    """

    if is_array_like(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        try:
            x_arr = np.asarray(x)
            if (np.asarray(x_arr) == None).all():
                result = True
                xtype = f"({type(x)})[{x_arr.dtype}]"
            else:
                result = False
                xtype = f"({type(x)})[{x_arr.dtype}]"
        except (ValueError, TypeError):
            result = False
            xtype = f"({type(x)})[{type(x[0])}]"
        pass
    else:
        result = False
        xtype = f"{type(x)}"

    if verbose:
        return (result, xtype, f"({darr_all})[NoneType]")
    else:
        return result


def is_ase_atoms(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is an ASE atoms object.

    Parameters
    ----------
    x: Any
        Input variable of which to check object type
    verbose: bool, optional, default False
        If True, return the object type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match object type, else False.

    """
    if verbose:
        return isinstance(x, ase.atoms.Atoms), type(x), "ase.Atoms"
    else:
        return isinstance(x, ase.atoms.Atoms)


def is_ase_atoms_array(
    x: Any,
    inhomogeneity: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> bool:
    """
    Check if the input is an ASE atoms object.

    Parameters
    ----------
    x: Any
        Input variable of which to check object type
    inhomogeneity: bool, optional, default False
        If True, return positive match for inhomogeneous array like variables
    verbose: bool, optional, default False
        If True, return the object type of the input and the expected type

    Returns
    -------
    bool
        True, if input variable match object type, else False.

    """

    if is_array_like(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        if all([is_ase_atoms(xi) for xi in x]):
            result = True
            xtype = f"({type(x)})[{type(x[0])}]"
        else:
            result = False
            xtype = f"({type(x)})[{type(x[0])}]"
    else:
        result = False
        xtype = f"{type(x)}"

    if verbose:
        return (result, xtype, f"({darr_all})[ase.Atoms]")
    else:
        return result

# --------------- ** Checking torch data options ** ---------------


def is_grad_enabled(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is a torch tensor with gradient enabled

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if option of torch input matches, else False.

    """
    if verbose:
        return (
            isinstance(x, torch.autograd.Variable),
            type(x),
            f"Gradient for {x} is active")
    else:
        return isinstance(x, torch.autograd.Variable)


def in_cuda(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is a torch tensor in CUDA.

    Parameters
    ----------
    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if option of torch input matches, else False.

    """
    val_cuda = x.is_cuda

    if val_cuda:
        message = "Tensor is in CUDA"
    else:
        message = "Tensor is in CPU"

    if verbose:
        return (
            val_cuda,
            type(x),
            message)
    else:
        return val_cuda


def is_attached(x: Any, verbose: Optional[bool] = False) -> bool:
    """
    Check if the input is a torch tensor attached to the computational graph

    Parameters
    ----------

    x: Any
        Input variable of which to check dtype
    verbose: bool, optional, default False
        If True, return the type of the input and the expected type

    Returns
    -------
    bool
        True, if option of torch input matches, else False.

    """
    x_attached = x.is_leaf

    if x_attached:
        message = "Tensor is attached"
    else:
        message = "Tensor is detached"

    if verbose:
        return (
            x_attached,
            type(x),
            message)
    else:
        return x_attached
