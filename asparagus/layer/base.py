from typing import Optional, Union, Callable

import torch

__all__ = ['DenseLayer', 'ResidualLayer']

#======================================
# Neural Network Layers
#======================================


class DenseLayer(torch.nn.Linear):
    """
    Dense layer: wrapper for torch.nn.Linear

    Parameters
    ----------
    n_input: int
        Number of input features.
    n_output: int
        Number of output features.
    activation_fn: callable, optional, default None
        Activation function. If None, identity is used.
    bias: bool, optional, default True
        If True, apply bias shift on neuron output.
    weight_init: callable, optional, default 'torch.nn.init.xavier_normal_'
        By Default, use Xavier initialization for neuron weight else zero
        weights are used.
    bias_init: callable, optional, default 'torch.nn.init.zeros_'
        By Default, zero bias values are initialized.
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        activation_fn: Optional[Union[Callable, torch.nn.Module]] = None,
        bias: Optional[bool] = True,
        weight_init: Optional[Callable] = torch.nn.init.xavier_normal_,
        bias_init: Optional[Callable] = torch.nn.init.zeros_,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize dense layer.

        """

        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(
            n_input, n_output, bias=bias, device=device, dtype=dtype)

        # Assign activation function
        if activation_fn is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation_fn

        return

    def reset_parameters(
        self
    ):
        """
        Initialize dense layer variables.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)
        
        return

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:

        output = torch.nn.functional.linear(input, self.weight, self.bias)
        output = self.activation(output)

        return output


class ResidualLayer(torch.nn.Module):
    """
    Residual layer

    Parameters
    ----------
    n_input: int
        Number of input features.
    activation_fn: callable, optional, default None
        Activation function passed to dense layer
    bias: bool, optional, default True
        If True, apply bias shift for dense layers.
    weight_1_init: callable, optional, default 'torch.nn.init.orthogonal_'
        By Default, use orthogonal initialization for first dense layer 
        weights. If None, use zero initialization.
    weight_2_init: callable, optional, default 'torch.nn.init.zeros_'
        By Default, use zero initialization for second dense layer weights.
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        n_input: int,
        activation_fn: Optional[Union[Callable, torch.nn.Module]] = None,
        bias: Optional[bool] = True,
        weight_1_init: Optional[Callable] = torch.nn.init.orthogonal_,
        weight_2_init: Optional[Callable] = torch.nn.init.zeros_,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(ResidualLayer, self).__init__()

        # Assign initial activation function
        if activation_fn is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation_fn

        # Assign first dense layer
        self.dense1 = DenseLayer(
            n_input,
            n_input,
            activation_fn=activation_fn,
            bias=bias,
            weight_init=weight_1_init,
            device=device,
            dtype=dtype)
        
        # Assign second dense layer
        self.dense2 = DenseLayer(
            n_input,
            n_input,
            activation_fn=None,
            bias=bias,
            weight_init=weight_2_init,
            device=device,
            dtype=dtype)

        return

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:

        output = self.activation(input)     # Activation
        output = self.dense1(output)        # Linear + Activation
        output = self.dense2(output)        # Linear 

        return input + output               # Add Input + Residual output
