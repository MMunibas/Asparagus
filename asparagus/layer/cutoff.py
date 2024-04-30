
import math
from typing import Optional, Union, Callable

import torch

from .. import utils

__all__ = [
    'get_cutoff_fn', 'Poly6Cutoff', 'Poly6Cutoff_range', 'CosineCutoff',
    'CosineCutoff_range']

#======================================
# Cutoff functions
#======================================


class Poly6Cutoff(torch.nn.Module):
    """
    2nd derivative smooth polynomial cutoff function of 6th order.

    $$f(x) = 1 - 6x^{5} + 15x^{4} - 10x^{3}$$
    with $$x = distance/cutoff$$

    Parameters
    ----------
    cutoff: float
        Cutoff distance
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(Poly6Cutoff, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))

        return

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff values of input distance tensor

        """

        x = distance/self.cutoff

        return torch.where(
            x < 1.0,
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,
            torch.zeros_like(distance))


class Poly6Cutoff_range(torch.nn.Module):
    """
    2nd derivative smooth polynomial cutoff function of 6th order,
    within the range cuton < x < cutoff

    $$f(x) = 1 - 6x^{5} + 15x^{4} - 10x^{3}$$
    with $$(x - cutoff - width) = distance/width$$

    **Note**: This function is for CHARMM potential.

    Parameters
    ----------
    cutoff: float
        Upper Cutoff distance of cutoff range
    cuton: float
        Lower Cutoff distance of cutoff range
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        cuton: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(Poly6Cutoff_Width, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))
        self.register_buffer("cuton", torch.tensor([cuton], dtype=dtype))
        self.width = cutoff - cuton

        return

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff function values of input distance tensor

        """

        x = (distance - self.cuton)/self.width

        switch = torch.where(
            distance < self.cutoff,
            torch.ones_like(distance),
            torch.zeros_like(distance)
            )

        switch = torch.where(
            torch.logical_and(
                (distance > self.cuton),
                (distance < self.cutoff)
            ),
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,
            switch)

        return switch 


class CosineCutoff(torch.nn.Module):
    """
    Behler-style cosine cutoff module, 
    within the range 0 < x < cutoff
    [source https://github.com/atomistic-machine-learning/schnetpack/blob/
        master/src/schnetpack/nn/cutoff.py]

    Parameters
    ----------
    cutoff: float
        Cutoff distance
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(CosineCutoff, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))

        return

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff values of input distance tensor

        """

        x = distance/self.cutoff

        return torch.where(
            x < 1.0,
            0.5*(torch.cos(x*math.pi) + 1.0),
            torch.zeros_like(distance))


class CosineCutoff_range(torch.nn.Module):
    """
    Behler-style cosine cutoff module.
    [source https://github.com/atomistic-machine-learning/schnetpack/blob/
        master/src/schnetpack/nn/cutoff.py]

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)
          \right] & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Parameters
    ----------
    cutoff: float
        Upper Cutoff distance of cutoff range
    cuton: float
        Lower Cutoff distance of cutoff range
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(CosineCutoff, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))
        self.register_buffer("cuton", torch.tensor([cuton], dtype=dtype))
        self.width = cutoff - cuton

        return
        
    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff function values of input distance tensor

        """
        
        x = (distance - self.cuton)/self.width
        
        switch = torch.where(
            distance < self.cutoff,
            torch.ones_like(distance),
            torch.zeros_like(distance)
            )

        switch = torch.where(
            torch.logical_and(
                (distance > self.cuton),
                (distance < self.cutoff)
            ),
            0.5*(torch.cos(x*math.pi) + 1.0),
            switch)

        return switch

#======================================
# Function assignment
#======================================

functions_avaiable = {
    'default'.lower(): Poly6Cutoff,
    'Poly6'.lower(): Poly6Cutoff,
    'Poly6_range'.lower(): Poly6Cutoff_range,
    'Cosine'.lower(): CosineCutoff,
    'Cosine_range'.lower(): CosineCutoff_range,
    }


def get_cutoff_fn(
    name: Optional[Union[Callable, str]] = None,
) -> torch.nn.Module:
    """
    Get cutoff function by defined name.

    Parameters
    ----------
    name: (str, object), optional, default None
        If name is a str than it checks for the corresponding cutoff function
        and return the function object.
        The input will be given if it is already a function object.
        If None, then default cutoff function is used.

    Returns
    -------
    torch.nn.Module
        Cutoff function layer

    """

    # Check for default option
    if name is None:
        name = 'default'

    # Get cutoff function
    elif utils.is_callable(name):
        return name

    elif utils.is_string(name):
        if name.lower() in functions_avaiable:
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Cutoff function input '{name}' is not valid! "
                + "Choose from:\n"
                + str(functions_avaiable.keys()))

    else:
        raise ValueError(
            f"Cutoff function input of type '{type(name)}' "
            + "is not valid! Input 'name' has to be an object or 'str' from;\n"
            + str(functions_avaiable.keys()))
