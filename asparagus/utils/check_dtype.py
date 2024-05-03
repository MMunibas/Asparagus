
import numpy as np

import logging
from typing import Callable

import torch
import ase

from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Bool data types
dbool_all = (bool, np.bool_)

def is_None(x, verbose=False):
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

    if verbose:
        return (x is None), type(x), "NoneType"
    else:
        return (x is None)

def is_string(x, verbose=False):
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
    if verbose:
        return isinstance(x, str), type(x), "str"
    else:
        return isinstance(x, str)

def is_bool(x, verbose=False):
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
    if verbose:
        return isinstance(x, dbool_all), type(x), "bool"
    else:
        return isinstance(x, dbool_all)

def is_numeric(x, verbose=False):
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
    if isinstance(x, (torch.Tensor, np.ndarray)):
        if len(x.shape)==0:
            result = ((x.dtype in dnum_all), type(x), str(dnum_all))
        else:
            result = (False, type(x), str(dnum_all))
    else:
        result = (
            (type(x) in dnum_all and not is_bool(x)), type(x), str(dnum_all))

    if verbose:
        return result
    else:
        return result[0]

def is_integer(x, verbose=False):
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
    if isinstance(x, (torch.Tensor, np.ndarray)):
        if len(x.shape)==0:
            result = ((x.dtype in dint_all), type(x), str(dint_all))
        else:
            result = (False, type(x), str(dint_all))
    else:
        result = (
            (type(x) in dint_all and not is_bool(x)), type(x), str(dint_all))

    if verbose:
        return result
    else:
        return result[0]

def is_callable(x, verbose=False):
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

def is_object(x, verbose=False):
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

def is_dictionary(x, verbose=False):
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

def is_array_like(x, inhomogeneity=False, verbose=False):
    """
    Check if the input is an array-like object. 
    If requested, positive check allows inhomogeneity of the array object.

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
    # Test for inhomogeneity of the array object
    if inhomogeneity:
        x = [xi for xi in utils.flatten_array_like(x)]
    if verbose:
        try:
            _ = np.asarray(x)
        except ValueError as error:
            logger.warning(error)
        return isinstance(x, darr_all), type(x), str(darr_all)
    else:
        return isinstance(x, darr_all)

def is_numeric_array(x, inhomogeneity=False, verbose=False):
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
    if is_array_like(x, inhomogeneity=inhomogeneity) and len(x):
        try:
            if inhomogeneity:
                x = [xi for xi in utils.flatten_array_like(x)]            
            result = (
                True,
                f"({type(x)})[{np.asarray(x).dtype}]",
                f"({darr_all})[{dnum_all}]")
        except (ValueError, TypeError):
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[{dnum_all}]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]", 
            f"({darr_all})[{dnum_all}]")
    else:
        result = (
            False, 
            f"{type(x)}",
            f"({darr_all})[{dnum_all}]")

    if verbose:
        return result
    else:
        return result[0]

def is_integer_array(x, inhomogeneity=False, verbose=False):
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
    if is_numeric_array(x, inhomogeneity=inhomogeneity):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        if (np.asarray(x).dtype in dint_all):
            result = (
                True,
                f"{type(x)}[{np.asarray(x).dtype}]",
                f"({darr_all})[{dint_all}]")
        else:
            result = (
                False,
                f"{type(x)}[{type(x[0])}]",
                f"({darr_all})[{dint_all}]")
    elif is_array_like(x):
        result = (
            True, 
            f"({type(x)})[empty]",
            f"({darr_all})[{dint_all}]")
    else:
        result = (
            False, 
            f"({type(x)})",
            f"({darr_all})[{dint_all}]")

    if verbose:
        return result
    else:
        return result[0]

def is_string_array(x, inhomogeneity=False, verbose=False):
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
    if is_array_like(x, inhomogeneity=inhomogeneity) and len(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        try:
            np.asarray(x, dtype=str)
            result = (
                True,
                f"({type(x)})[{np.asarray(x).dtype}]",
                f"({darr_all})[str]")
        except (ValueError, TypeError):
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[str]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]",
            f"({darr_all})[str]")
    else:
        result = (
            False,
            f"{type(x)}",
            f"({darr_all})[str]")

    if verbose:
        return result
    else:
        return result[0]

def is_bool_array(x, inhomogeneity=False, verbose=False):
    """
    Redirection to is_boolean_array
    """
    return is_boolean_array(x, inhomogeneity=inhomogeneity, verbose=verbose)

def is_boolean_array(x, inhomogeneity=False, verbose=False):
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
    if is_array_like(x, inhomogeneity=inhomogeneity) and len(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        if (np.asarray(x).dtype in dbool_all):
            result = (
                True,
                f"({type(x)})[{np.asarray(x).dtype}]",
                f"({darr_all})[bool]")
        else:
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[bool]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]",
            f"({darr_all})[bool]")
    else:
        result = (
            False,
            f"{type(x)}",
            f"({darr_all})[bool]")

    if verbose:
        return result
    else:
        return result[0]

def is_None_array(x, inhomogeneity=False, verbose=False):
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
    if is_array_like(x, inhomogeneity=inhomogeneity) and len(x):
        if inhomogeneity:
            x = [xi for xi in utils.flatten_array_like(x)]
        if (np.asarray(x)==None).all():
            result = (
                True,
                f"({type(x)})[None]",
                f"({darr_all})[None]")
        else:
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[None]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]",
            f"({darr_all})[None]")
    else:
        result = (
            False,
            f"{type(x)}",
            f"({darr_all})[None]")

    if verbose:
        return result
    else:
        return result[0]

def is_ase_atoms(x, verbose=False):
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

def is_ase_atoms_array(x, inhomogeneity=False, verbose=False):
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
    if is_array_like(x, inhomogeneity=inhomogeneity) and len(x):
        if all([is_ase_atoms(xi) for xi in utils.flatten_array_like(x)]):
            result = (
                True,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[ase.Atoms]")
        else:
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[ase.Atoms]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]",
            f"({darr_all})[ase.Atoms]")
    else:
        result = (
            False,
            f"{type(x)}",
            f"({darr_all})[ase.Atoms]")

    if verbose:
        return result
    else:
        return result[0]

# --------------- ** Checking torch data options ** ---------------

def is_grad_enabled(x, verbose=False):
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

def in_cuda(x, verbose=False):
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

def is_attached(x, verbose=False):
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
