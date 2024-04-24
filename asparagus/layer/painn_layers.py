from typing import Optional, Union, List, Dict, Callable

import torch

from .base import DenseLayer

from .. import utils

__all__ = [
    'PaiNNInteraction', 'PaiNNMixing', 
    'PaiNNOutput_scalar', 'PaiNNOutput_tensor']

class PaiNNInteraction(torch.nn.Module):
    """
    Interaction block for PaiNN.

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    activation_fn: callable, optional, default None
        Residual layer activation function.
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        n_atombasis: int,
        activation_fn: Optional[Callable] = None,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PaiNN interaction block.
        
        """

        super(PaiNNInteraction, self).__init__()

        # Initialize context layer
        self.context = torch.nn.Sequential(
            DenseLayer(
                n_atombasis, 
                n_atombasis,
                bias=True,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
                ),
            DenseLayer(
                n_atombasis, 
                3*n_atombasis,
                bias=True,
                activation_fn=None,
                device=device,
                dtype=dtype
                ),
            )

        return

    def forward(
        self,
        sfeatures: torch.Tensor,
        efeatures: torch.Tensor,
        descriptors: torch.Tensor,
        vectors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
        n_features: int
    ) -> (torch.Tensor, torch.Tensor):

        # Apply context layer on scalar features
        cfea = self.context(sfeatures)

        # Apply conformational information on context of scalar features
        cfea = descriptors*cfea[idx_j]

        # Split update information
        ds, dev, dee = torch.split(
            cfea, n_features, dim=-1)

        # Reduce from pair information to atom i
        ds = torch.zeros(
            (n_atoms, 1, n_features),
            device=ds.device, dtype=ds.dtype
            ).index_add(0, idx_i, ds)

        # Compute equivariant feature vector update and reduce to atom i
        de = dev*vectors[..., None] + dee*efeatures[idx_j]
        de = torch.zeros(
            (n_atoms, 3, n_features),
            device=de.device, dtype=de.dtype
            ).index_add(0, idx_i, de)

        # Update scalar and equivariant feature vector
        sfeatures = sfeatures + ds
        efeatures = efeatures + de

        return sfeatures, efeatures


class PaiNNMixing(torch.nn.Module):
    """
    Mixing block of scalar and equivariant feature vectors in PaiNN.

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    activation_fn: callable, optional, default None
        Residual layer activation function.
    stability_constant: float, optional, default 1e-8
        Numerical stability added constant
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type
    
    """
    
    def __init__(
        self,
        n_atombasis: int,
        activation_fn: Optional[Callable] = None,
        stability_constant: Optional[float] = 1.e-8,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PaiNN mixing block.
        
        """

        super(PaiNNMixing, self).__init__()

        # Assign class parameter
        self.stability_constant = stability_constant
        
        # Initialize context and mixing layer
        self.context = DenseLayer(
            n_atombasis, 
            2*n_atombasis,
            activation_fn=activation_fn,
            bias=False,
            device=device,
            dtype=dtype,
            )
        self.mixing = torch.nn.Sequential(
            DenseLayer(
                2*n_atombasis, 
                n_atombasis,
                bias=True,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
                ),
            DenseLayer(
                n_atombasis, 
                3*n_atombasis,
                activation_fn=None,
                bias=False,
                device=device,
                dtype=dtype
                ),
            )

        return

    def forward(
        self,
        sfeatures: torch.Tensor,
        vfeatures: torch.Tensor,
        n_features: int,
    ) -> (torch.Tensor, torch.Tensor):
        
        # Apply context layer on vectorial feature vector and split 
        # information
        cfea = self.context(vfeatures)
        U, V = torch.split(cfea, n_features, dim=-1)
        
        # Mix scalar and vectorial feature vector
        U = torch.sqrt(
            torch.sum(U**2, dim=-2, keepdim=True) 
            + self.stability_constant)
        mix = torch.cat([sfeatures, U], dim=-1)
        mix = self.mixing(mix)
        
        # Split update information
        ds, dv, dsv = torch.split(mix, n_features, dim=-1)
        
        # Compute scalar and vectorial feature vector update
        ds = ds + dsv*torch.sum(U*V, dim=1, keepdim=True)
        dv = dv*V
        
        # Update feature and message vector 
        sfeatures = sfeatures + ds
        vfeatures = vfeatures + dv
        
        return sfeatures, vfeatures


class PaiNNGatedEquivarience(torch.nn.Module):
    """
    Gated equivariant block of the PaiNN model.
    
    Parameters
    ----------
    scalar_n_input: int
        Number of input scalar representation features.
    vector_n_input: int
        Number of input vector representation features.
    scalar_n_output: int
        Number of output scalar representation features.
    vector_n_output: int
        Number of output vector representation features.
    hidden_n_neurons: int
        Number of hidden neurons
    scalar_activation_fn: callable, optional, default None
        Activation function on scalar features. If None, identity is used.
    hidden_activation_fn: callable, optional, default None
        Activation function on hidden layer . If None, identity is used.
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
        scalar_n_input: int,
        vector_n_input: int,
        scalar_n_output: int,
        vector_n_output: int,
        hidden_n_neurons: int,
        scalar_activation_fn: 
            Optional[Union[Callable, torch.nn.Module]] = None,
        hidden_activation_fn: 
            Optional[Union[Callable, torch.nn.Module]] = None,
        bias: Optional[bool] = True,
        weight_init: Optional[Callable] = torch.nn.init.xavier_normal_,
        bias_init: Optional[Callable] = torch.nn.init.zeros_,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize gated equivariant block.
        
        """
        
        super(PaiNNGatedEquivarience, self).__init__()
        
        # Input and output feature vector dimensions
        self.scalar_n_input = scalar_n_input
        self.vector_n_input = vector_n_input
        self.scalar_n_output = scalar_n_output
        self.vector_n_output = vector_n_output
        
        # Number of hidden layer
        self.hidden_n_neurons = hidden_n_neurons
        
        # Dense module to branch vector features for normalization to a scalar
        # feature vector and keeping a vector representation for scaling
        self.branch_vector = DenseLayer(
            vector_n_input,
            2*vector_n_output,
            activation_fn=None,
            bias=False,
            weight_init=weight_init,
            device=device,
            dtype=dtype)

        # Mixed scalar/norm(vector) representation network
        self.mixed_scalars = torch.nn.Sequential(
            DenseLayer(
                scalar_n_input + vector_n_output,
                hidden_n_neurons,
                activation_fn=hidden_activation_fn,
                bias=bias,
                weight_init=weight_init,
                bias_init=bias_init,
                device=device,
                dtype=dtype),
            DenseLayer(
                hidden_n_neurons,
                scalar_n_output + vector_n_output,
                activation_fn=None,
                bias=bias,
                weight_init=weight_init,
                bias_init=bias_init,
                device=device,
                dtype=dtype),
        )

        # Scalar feature activation function
        self.scalar_activation_fn = scalar_activation_fn

        return

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Forward pass of the gated equivariant block.
        
        Parameter
        ---------
        features: list(torch.tensor)
            List of scalar and vector feature vectors:
            sfeatures: torch.tensor(N_atoms, n_atombasis)
                Scalar atomic feature vectors
            vfeatures: torch.tensor(N_atoms, 3, n_atombasis)
                Vector atomic feature vectors
        
        Returns
        -------
        sfeatures: torch.tensor(N_atoms, n_atombasis)
            Modified scalar atomic feature vectors
        vfeatures: torch.tensor(N_atoms, 3, n_atombasis)
            Modified vector atomic feature vectors
        
        """
        
        # Extract scalar and vector features
        sfeatures, vfeatures = features
        
        # Branch vector features
        vmix = self.branch_vector(vfeatures)
        vfeatures_norm, vfeatures_keep = torch.split(
            vmix, self.vector_n_output, dim=-1)
        
        # Normalize vector features
        vfeatures_norm = torch.norm(vfeatures_norm, dim=-2)
        
        # Pass mixed features to network
        output = self.mixed_scalars(
            torch.cat([sfeatures, vfeatures_norm], dim=-1))
        
        # Split output into scalar features and vector scaling features
        sfeatures_out, vfeatures_scale = torch.split(
            output, [self.scalar_n_output, self.vector_n_output], dim=-1)
        
        # Scale vector features
        vfeatures_out = vfeatures_keep*vfeatures_scale[:, None, :]

        # Apply scalar activation function
        sfeatures_out = self.scalar_activation_fn(sfeatures_out)

        return (sfeatures_out, vfeatures_out)


class PaiNNOutput_scalar(torch.nn.Module):
    """
    Deep neural network output block for scalar property predictions from 
    atom-wise representations in PaiNN.

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_property: int
        Dimension of the predicted property.
    n_layer: int, optional, default 2
        Number of hidden layer in the output block.
    n_neurons: (int list(int)), optional, default None
        Number of neurons of the hidden layers (int) or per hidden layer (list)
    activation_fn: callable, optional, default None
        Residual layer activation function.
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """
    
    _default_output_scalar = {
        'n_layer':              2,
        'n_neurons':            None,
        'activation_fn':        None,
        'bias_layer':           True,
        'bias_last':            True,
        'weight_init_layer':    torch.nn.init.zeros_,
        'weight_init_last':     torch.nn.init.zeros_,
        'bias_init_layer':      torch.nn.init.zeros_,
        'bias_init_last':       torch.nn.init.zeros_,
        }
    
    def __init__(
        self,
        n_atombasis: int,
        n_property: int,
        n_layer: Optional[int] = None,
        n_neurons: Optional[Union[int, List[int]]] = None,
        activation_fn: Optional[Callable] = None,
        bias_layer: Optional[bool] = None,
        bias_last: Optional[bool] = None,
        weight_init_layer: Optional[Callable] = None,
        weight_init_last: Optional[Callable] = None,
        bias_init_layer: Optional[Callable] = None,
        bias_init_last: Optional[Callable] = None,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PaiNN scalar output block.
        
        """
        
        super(PaiNNOutput_scalar, self).__init__()

        # Check input
        _ = utils.check_input_args(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=self._default_output_scalar,
            check_dtype=None)

        # Check hidden layer neuron options
        if self.n_layer:
            n_neurons_list = [self.n_atombasis]
            if utils.is_integer(self.n_neurons):
                n_neurons_list += [self.n_neurons]*(self.n_layer - 1)
            elif utils.is_integer_array(self.n_neurons):
                n_neurons_list += list(self.n_neurons)[:(self.n_layer - 1)]
            else:
                # Half number of hidden layer neurons with each layer
                ni = self.n_atombasis
                for ii in range(self.n_layer - 1):
                    ni = max(self.n_property, ni//2)
                    n_neurons_list.append(ni)
            n_neurons_list.append(self.n_property)
        else:
            # If no hidden layer, set hidden neurons to property neuron number
            n_neurons_list = [self.n_atombasis, self.n_property]

        # Initialize output module
        self.output = torch.nn.Sequential()

        # Append hidden layers
        for ii in range(n_layer - 1):
            self.output.append(
                DenseLayer(
                    n_neurons_list[ii], 
                    n_neurons_list[ii + 1],
                    activation_fn=self.activation_fn,
                    bias=self.bias_layer,
                    weight_init=self.weight_init_layer,
                    bias_init=self.bias_init_layer,
                    device=device,
                    dtype=dtype
                    ),
                )
        
        # Append output layer
        self.output.append(
            DenseLayer(
                n_neurons_list[-2], 
                n_neurons_list[-1],
                activation_fn=None,
                bias=self.bias_last,
                weight_init=self.weight_init_last,
                bias_init=self.bias_init_last,
                device=device,
                dtype=dtype
                ),
            )
        
        return

    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply interaction block.
        
        Parameters
        ----------
        features: torch.Tensor(N_atoms, n_atombasis)
            Atomic feature vectors

        Returns
        -------
        torch.Tensor(N_atoms, n_property)
            Transformed atomic feature vector to result property vector
        
        """
        
        # Transform to result properties
        result = self.output(features)

        return result


class PaiNNOutput_tensor(torch.nn.Module):
    """
    Gated equivariant output block for tensor like property predictions from 
    atom-wise representations in PaiNN.

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_property: int
        Dimension of the predicted property.
    n_layer: int, optional, default 2
        Number of hidden layer in the output block.
    n_neurons: (int list(int)), optional, default None
        Number of neurons of the input and output neurons of the layer in the
        gated equivariant block.
    hidden_n_neurons: (int list(int)), optional, default None
        Number of neurons of the hidden layers (int) or per tensor
        hidden layer (list)
    scalar_activation_fn: callable, optional, default None
        Scalar feature activation function.
    hidden_activation_fn: callable, optional, default None
        Hidden layer activation function.
    bias_layer: bool, optional, default True
        Add bias parameter for hidden layer neurons
    bias_last: bool, optional, default True
        Add bias parameter for last layer neuron(s)
    weight_init_layer: callable, optional, default 'torch.nn.init.zeros_'
        Weight parameter initialization function of the hidden layer
    weight_init_last: callable, optional, default 'torch.nn.init.zeros_'
        Weight parameter initialization function of the last layer
    bias_init_layer: callable, optional, default 'torch.nn.init.zeros_'
        Bias parameter initialization function of the hidden layer
    bias_init_last: callable, optional, default 'torch.nn.init.zeros_'
        Bias parameter initialization function of the last layer
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """
    
    _default_output_tensor = {
        'n_layer':              2,
        'n_neurons':            None,
        'hidden_n_neurons':     None,
        'scalar_activation_fn': None,
        'hidden_activation_fn': None,
        'bias_layer':           True,
        'bias_last':            True,
        'weight_init_layer':    torch.nn.init.zeros_,
        'weight_init_last':     torch.nn.init.zeros_,
        'bias_init_layer':      torch.nn.init.zeros_,
        'bias_init_last':       torch.nn.init.zeros_,
        }
    
    def __init__(
        self,
        n_atombasis: int,
        n_property: int,
        n_layer: Optional[int] = None,
        n_neurons: Optional[Union[int, List[int]]] = None,
        hidden_n_neurons: Optional[Union[int, List[int]]] = None,
        scalar_activation_fn: Optional[Callable] = None,
        hidden_activation_fn: Optional[Callable] = None,
        bias_layer: Optional[bool] = True,
        bias_last: Optional[bool] = True,
        weight_init_layer: Optional[Callable] = torch.nn.init.zeros_,
        weight_init_last: Optional[Callable] = torch.nn.init.zeros_,
        bias_init_layer: Optional[Callable] = torch.nn.init.zeros_,
        bias_init_last: Optional[Callable] = torch.nn.init.zeros_,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PaiNN tensor output block.
        
        """
        
        super(PaiNNOutput_tensor, self).__init__()

        # Check input
        _ = utils.check_input_args(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=self._default_output_tensor,
            check_dtype=None)

        # Check hidden scalar layer neuron options
        if self.n_layer:
            n_neurons_list = [self.n_atombasis]
            if utils.is_integer(self.n_neurons):
                n_neurons_list += [self.n_neurons]*(
                    self.n_layer - 1)
            elif utils.is_integer_array(self.n_neurons):
                n_neurons_list += list(self.n_neurons)[
                    :(self.n_layer - 1)]
            else:
                # Half number of hidden layer neurons with each layer
                ni = self.n_atombasis
                for ii in range(self.n_layer - 1):
                    ni = max(self.n_property, ni//2)
                    n_neurons_list.append(ni)
            n_neurons_list.append(self.n_property)
        else:
            # If no hidden layer, set hidden neurons to property neuron number
            n_neurons_list = [self.n_atombasis, self.n_property]

        # Check hidden layer neuron options
        if self.hidden_n_neurons is None:
            hidden_n_neurons_list = n_neurons_list[:-1]
        elif utils.is_integer(self.hidden_n_neurons):
            hidden_n_neurons_list = [self.hidden_n_neurons]*self.n_layer
        else:
            hidden_n_neurons_list = list(self.hidden_n_neurons)[:self.n_layer]

        # Initialize output module
        self.output = torch.nn.Sequential()

        # Append hidden layers
        for ii in range(n_layer - 1):
            self.output.append(
                PaiNNGatedEquivarience(
                    n_neurons_list[ii],
                    n_neurons_list[ii],
                    n_neurons_list[ii + 1],
                    n_neurons_list[ii + 1],
                    hidden_n_neurons_list[ii],
                    scalar_activation_fn=self.scalar_activation_fn,
                    hidden_activation_fn=self.hidden_activation_fn,
                    bias=self.bias_layer,
                    weight_init=self.weight_init_layer,
                    bias_init=self.bias_init_layer,
                    device=device,
                    dtype=dtype
                    ),
                )
        
        # Append output layer
        self.output.append(
            PaiNNGatedEquivarience(
                n_neurons_list[-2],
                n_neurons_list[-2],
                n_neurons_list[-1],
                n_neurons_list[-1],
                hidden_n_neurons_list[-1],
                scalar_activation_fn=self.scalar_activation_fn,
                hidden_activation_fn=None,
                bias=self.bias_last,
                weight_init=self.weight_init_last,
                bias_init=self.bias_init_last,
                device=device,
                dtype=dtype
                ),
            )

        return

    def forward(
        self,
        sfeatures: torch.Tensor,
        vfeatures: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Apply scalar output block.
        
        Parameters
        ----------
        sfeatures: torch.tensor(N_atoms, n_atombasis)
            Scalar atomic feature vectors
        vfeatures: torch.tensor(N_atoms, 3, n_atombasis)
            Vector atomic feature vectors
        
        Returns
        -------
        torch.Tensor(N_atoms, n_property)
            Resulting atomic scalar property vector
        torch.Tensor(N_atoms, 3, n_property)
            Resulting atomic vector properties.
        
        Parameter
        ---------
        
        Returns
        -------
        scalar_result: torch.tensor(N_atoms, n_atombasis)
            Modified scalar atomic feature vectors
        vector_result: torch.tensor(N_atoms, 3, n_atombasis)
            Modified vector atomic feature vectors
        
        """
        
        # Transform to scalar and vector results
        scalar_result, vector_result = self.output(
            (sfeatures, vfeatures)
            )

        return scalar_result, vector_result
        
