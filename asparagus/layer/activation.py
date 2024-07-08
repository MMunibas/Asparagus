from typing import Union, Callable

import torch

from .. import utils

__all__ = [
    'get_activation_fn', 'swish', 'softplus', 'shifted_softplus',
    'scaled_shifted_softplus', 'self_normalizing_shifted_softplus',
    'smooth_ELU', 'self_normalizing_smooth_ELU', 'self_normalizing_asinh',
    'self_normalizing_tanh', 'linear']

#======================================
# Activation functions
#======================================


# Google's swish function
@torch.jit.script
def swish(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Google's swish function

    .. math:: f(x) = x * sigmoid(x)

    Parameters
    ----------
    x : torch.Tensor
    
    Returns
    -------
    torch.Tensor

    """

    return x * torch.nn.functional.sigmoid(x)

@torch.jit.script
def _softplus(
    x: torch.Tensor
) -> torch.Tensor:
    return torch.log1p(torch.exp(x))

@torch.jit.script
def softplus(
    x: torch.Tensor
) -> torch.Tensor:
    """
    This definition is for numerical stability for x larger than 15 (single
    precision) or x larger than 34 (double precision), there is no numerical
    difference anymore between the softplus and a linear function.

    *Fun fact*: First time softplus was used as activation function: "Incorporating
    Second-Order Functional Knowledge for Better Option Pricing"
    (https://papers.nips.cc/paper/1920-incorporating-second-order-functional-
    knowledge-for-better-option-pricing.pdf)

    .. math:: f(x) = ln(1+exp(x))

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor

    """

    return torch.where(
        x < 15.0, 
        _softplus(torch.where(x < 15.0, x, torch.zeros_like(x))), 
        x)


@torch.jit.script
def shifted_softplus(
    x: torch.Tensor
) -> torch.Tensor:
    """

    Shifted softplus function

    .. math:: f(x) = ln(1+exp(x)) - ln(2)

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0))



@torch.jit.script
def scaled_shifted_softplus(
    x: torch.Tensor
) -> torch.Tensor:
    """
    This ensures that the function is close to linear near the origin!

    .. math:: f(x) = 2 * (ln(1+exp(x)) - ln(2))

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return 2.0 * shifted_softplus(x)

@torch.jit.script
def self_normalizing_shifted_softplus(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Attempt to make the shifted softplus function self-normalizing.
    Is not really self-normalizing sadly...

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return 1.875596256135042 * shifted_softplus(x)


# General: ln((exp(alpha)-1)*exp(x)+1)-alpha
@torch.jit.script
def smooth_ELU(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Smooth ELU function

    .. math:: (e-1) = 1.718281828459045
    .. math:: f(x) = ln((exp(1)-1)*exp(x)+1)-1

    Parameters
    ----------
    x: torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return torch.log1p(1.718281828459045 * torch.exp(x)) - 1.0


@torch.jit.script
def self_normalizing_smooth_ELU(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Attempt to make the smooth ELU function self-normalizing.

    .. math:: f(x) = 1.574030675714671 * smoothELU(x)

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return 1.574030675714671 * smooth_ELU(x)


@torch.jit.script
def self_normalizing_asinh(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Self normalizing asinh function

    .. math:: f(x) = 1.256734802399369 * sinh^{-1}(x)

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return 1.256734802399369 * torch.asinh(x)


@torch.jit.script
def self_normalizing_tanh(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Self normalizing tanh function

    .. math:: f(x) = 1.592537419722831 * tanh(x)

    Parameters
    ----------
    x: torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return 1.592537419722831 * torch.tanh(x)


@torch.jit.script
def linear(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Linear activation function

    .. math:: f(x) = x

    Parameters
    ----------
    x: torch.Tensor

    Returns
    -------
    torch.Tensor

    """
    return x


#======================================
# Function Assignment
#======================================

functions_avaiable = {
    'swish'.lower(): swish,
    'silu'.lower(): torch.nn.functional.silu,
    'softplus'.lower(): softplus,
    'shifted_softplus'.lower(): shifted_softplus,
    'scaled_shifted_softplus'.lower(): scaled_shifted_softplus,
    'self_normalizing_shifted_softplus.lower()':
        self_normalizing_shifted_softplus,
    'smooth_ELU'.lower(): smooth_ELU,
    'self_normalizing_smooth_ELU'.lower(): self_normalizing_smooth_ELU,
    'self_normalizing_asinh'.lower(): self_normalizing_asinh,
    'self_normalizing_tanh'.lower(): self_normalizing_tanh,
    'linear'.lower(): linear,
    }


def get_activation_fn(
    name: Union[str, Callable]
) -> Callable:
    """
    Get activation function by defined name.

    Parameters
    ----------
    name: (str, callable)
        If name is a str than it checks for the corresponding activation
        function and return the function object.
        The input will be given if it is already a function object.

    Returns
    -------
    callable
        Activation function

    """

    # Check for default option
    if name is None:
        return functions_avaiable['linear']

    # Get activation function
    elif utils.is_callable(name):
        return name

    elif utils.is_string(name):
        if name.lower() in [key.lower() for key in functions_avaiable.keys()]:
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Activation function input '{name}' is not valid! "
                + "Choose from:\n"
                + str(functions_avaiable.keys()))

    else:
        raise ValueError(
            f"Activation function input of type '{type(name)}' " +
            "is not valid! Input 'name' has to be an object or 'str' from;\n"
            + str(functions_avaiable.keys()))
