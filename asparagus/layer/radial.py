import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import torch

from .. import utils

__all__ = ['get_radial_fn', 'GaussianRBF', 'GaussianRBF_PhysNet']

#======================================
# Radial basis functions
#======================================


class GaussianRBF(torch.nn.Module):
    """
    Gaussian type radial basis functions.

    Parameters
    ----------
    rbf_n_basis: int
        Number of RBF center
    rbf_center_start: float
        Initial lower RBF center range
    rbf_center_end: float
        Initial upper RBF center range
    rbf_trainable: bool
        Trainable RBF center positions
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        rbf_n_basis: int,
        rbf_center_start: float,
        rbf_center_end: float,
        rbf_trainable: bool,
        device: str,
        dtype: object,
    ):
        """
        Initialize Gaussian Radial Basis Function.
        """

        super(GaussianRBF, self).__init__()

        self.rbf_n_basis = rbf_n_basis
        self.rbf_trainable = rbf_trainable

        # Initialize RBF centers and widths
        centers = torch.linspace(
            rbf_center_start,
            rbf_center_end,
            rbf_n_basis,
            device=device,
            dtype=dtype)
        widths = torch.ones_like(centers, device=device, dtype=dtype)

        if rbf_trainable:
            self.centers = torch.nn.Parameter(centers)
            self.widths = torch.nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

        return

    def __str__(self):
        return 'GaussianRBF'        

    def forward(
        self,
        d: torch.Tensor
    ) -> torch.Tensor:

        x = torch.unsqueeze(d, -1)
        rbf = torch.exp(-0.5*((x - self.centers)/self.widths)**2)

        return rbf


class GaussianRBF_PhysNet(torch.nn.Module):
    """
    Original PhysNet type radial basis functions (RBFs) with double exponential
    expression.

    Parameters
    ----------
    rbf_n_basis: int
        Number of RBFs
    rbf_center_start: float
        Initial lower RBF center range
    rbf_center_end: float
        Initial upper RBF center range
    rbf_trainable: bool
        Trainable RBF center positions
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        rbf_n_basis: int,
        rbf_center_start: float,
        rbf_center_end: float,
        rbf_trainable: bool,
        device: str,
        dtype: object,
    ):
        """
        Initialize original PhysNet type Gaussian Radial Basis Function.
        """

        super(GaussianRBF_PhysNet, self).__init__()

        self.rbf_n_basis = rbf_n_basis
        self.rbf_cutoff_fn = rbf_cutoff_fn
        self.rbf_trainable = rbf_trainable

        # Initialize RBF centers and widths
        centers = torch.linspace(
            rbf_center_start, np.exp(-rbf_center_end), rbf_n_basis)
        softp = torch.nn.Softplus()
        width_val = self.softplus_inverse(
            (0.5 / ((1.0 - np.exp(-rbf_center_end)) / rbf_n_basis)) ** 2)
        widths = torch.empty(
            rbf_n_basis, dtype=torch.float64).new_full(
                (rbf_n_basis,), softp(width_val * rbf_n_basis))

        if rbf_trainable:
            self.centers = torch.nn.Parameter(centers)
            self.widths = torch.nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

        return

    def __str__(self):
        return 'GaussianRBF_PhysNet'        

    @staticmethod
    def softplus_inverse(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return torch.log(torch.exp(x) - 1)

    def forward(
        self,
        d: torch.Tensor
    ) -> torch.Tensor:

        x = torch.unsqueeze(d, -1)
        rbf = torch.exp(-self.widths*(torch.exp(-x) - self.centers)**2)

        return rbf


#======================================
# Function assignment
#======================================

functions_avaiable = {
    'default'.lower(): GaussianRBF,
    'GaussianRBF'.lower(): GaussianRBF,
    'GaussianRBF_PhysNet'.lower(): GaussianRBF_PhysNet,
    }


def get_radial_fn(
    name: Union[str, Callable],
) -> torch.nn.Module:
    """
    Get radial basis function by defined name.

    Parameters
    ----------
    name: (str, callable)
        If name is a str than it checks for the corresponding radial basis
        function and return the function object.
        The input will be given if it is already a function object.

    Returns
    -------
    torch.nn.Module
        Radial basis function object

    """

    # Check for default option
    if utils.is_callable(name):
        return name

    # Get radial function
    elif utils.is_string(name):
        if name.lower() in [key.lower() for key in functions_avaiable.keys()]:
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Radial basis function input '{name}' is not valid!" +
                "Choose from:\n" +
                str(functions_avaiable.keys()))

    else:
        raise ValueError(
            f"Radial basis function input of type '{type(name)}' " +
            "is not valid! Input 'name' has to be an object or 'str' from;\n" +
            str(functions_avaiable.keys()))
