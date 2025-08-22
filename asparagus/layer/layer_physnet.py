from typing import Optional, Tuple, Callable

import torch

from .base import DenseLayer, ResidualLayer

from asparagus import utils

__all__ = ['InteractionBlock', 'InteractionLayer', 'OutputBlock']

#======================================
# PhysNet NN Blocks
#======================================


class InteractionBlock(torch.nn.Module):
    """
    Interaction block for PhysNet

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_radialbasis: int
        Number of input radial basis centers
    n_residual_interaction: int
        Number of residual layers for atomic feature and radial basis vector
        interaction.
    n_residual_features: int
        Number of residual layers for atomic feature interactions.
    activation_fn: callable
        Residual layer activation function.
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type

    """

    def __init__(
        self,
        n_atombasis: int,
        n_radialbasis: int,
        n_residual_interaction: int,
        n_residual_features: int,
        activation_fn: Callable,
        device: str,
        dtype: 'dtype',
    ):
        """
        Initialize PhysNet interaction block.
        """

        super(InteractionBlock, self).__init__()

        # Atomic features and radial basis vector interaction layer
        self.interaction = InteractionLayer(
            n_atombasis,
            n_radialbasis,
            n_residual_interaction,
            activation_fn,
            device,
            dtype)

        # Atomic feature interaction layers
        self.residuals = torch.nn.ModuleList([
            ResidualLayer(
                n_atombasis,
                activation_fn,
                True,
                device,
                dtype)
            for _ in range(n_residual_features)])

        return

    def forward(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply interaction block.
        
        Parameters
        ----------
        features: torch.Tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        descriptors: torch.Tensor(N_atoms, n_atombasis, n_radialbasis)
            Atom pair radial distribution vectors
        idx_i: torch.Tensor(N_pairs)
            Atom i pair index
        idx_j: torch.Tensor(N_pairs)
            Atom j pair index

        Returns
        -------
        torch.Tensor(N_atoms, n_atombasis)
            Modified atom feature vectors

        """

        # Apply interaction layer
        x = self.interaction(features, descriptors, idx_i, idx_j)

        # Iterate through atomic feature interaction layers
        for residual in self.residuals:
            x = residual(x)

        return x


class InteractionLayer(torch.nn.Module):
    """
    Atomic features and radial basis vector interaction layer for PhysNet

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_radialbasis: int
        Number of input radial basis centers
    n_residual_interaction: int
        Number of residual layers for atomic feature and radial basis vector
        interaction.
    activation_fn: callable
        Residual layer activation function.
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type

    """

    def __init__(
        self,
        n_atombasis: int,
        n_radialbasis: int,
        n_residual_interaction: int,
        activation_fn: Callable,
        device: str,
        dtype: 'dtype',
    ):

        super(InteractionLayer, self).__init__()

        # Assign activation function
        if activation_fn is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation_fn

        # Dense layer for the conversion from radial basis vector to atomic 
        # feature vector length
        self.radial2atom = DenseLayer(
            n_radialbasis,
            n_atombasis,
            None,
            False,
            device,
            dtype,
            weight_init=torch.nn.init.xavier_normal_)
            

        # Dense layer for atomic feature vector for atom i
        self.dense_i = DenseLayer(
            n_atombasis,
            n_atombasis,
            activation_fn,
            True,
            device,
            dtype)
        
        # Dense layer for atomic feature vector for atom j
        self.dense_j = DenseLayer(
            n_atombasis,
            n_atombasis,
            activation_fn,
            True,
            device,
            dtype)

        # Residual layers for atomic feature vector pair interaction modifying 
        # the message vector
        self.residuals_ij = torch.nn.ModuleList([
            ResidualLayer(
                n_atombasis,
                activation_fn,
                True,
                device,
                dtype)
            for _ in range(n_residual_interaction)])

        # Dense layer for message vector interaction
        self.dense_out = DenseLayer(
            n_atombasis,
            n_atombasis,
            None,
            True,
            device,
            dtype)
        
        # Scaling vector for mixing of initial atomic feature vector with
        # message vector
        self.scaling = torch.nn.Parameter(
            torch.ones([n_atombasis], device=device, dtype=dtype))
        
        # Special case flag for variable assignment on CPU's
        if device.lower() == 'cpu':
            self.cpu = True
        else:
            self.cpu = False

        return

    def forward(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply interaction layer.
        
        Parameters
        ----------
        features: torch.Tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        descriptors: torch.Tensor(N_atoms, n_atombasis, n_radialbasis)
            Atom pair radial distribution vectors
        idx_i: torch.Tensor(N_pairs)
            Atom i pair index
        idx_j: torch.Tensor(N_pairs)
            Atom j pair index

        Returns
        -------
        torch.Tensor(N_atoms, n_atombasis)
            Modified atom feature vectors
        
        """

        # Apply initial activation function on atomic features
        x = self.activation(features)

        # Apply radial basis (descriptor) to feature vector layer
        g = self.radial2atom(descriptors)

        # Calculate contribution of central atom i and neighbor atoms j
        xi = self.dense_i(x)
        if self.cpu:
            gxj = g*self.dense_j(x)[idx_j]
        else:
            j = idx_j.view(-1, 1).expand(-1, features.shape[-1])
            gxj = g * torch.gather(self.dense_j(x), 0, j)

        # Combine descriptor weighted neighbor atoms feature vector for each
        # central atom i
        xj = torch.zeros_like(xi).scatter_add_(
            0, idx_i.unsqueeze(-1).repeat(1, xi.shape[1]), gxj)

        # Combine features to message vector
        message = xi + xj

        # Apply residual layers and activation function for message vector
        # interaction
        for residual in self.residuals_ij:
            message = residual(message)
        message = self.activation(message)

        # Mix initial atomic feature vector with message vector
        x = self.scaling*x + self.dense_out(message)

        return x


class OutputBlock(torch.nn.Module):
    """
    Output block for PhysNet

    Parameters
    ----------
    n_blocks: int
        Number of information processing cycles and sets of feature vectors
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_maxatom: int
        Maximum Number of atomic elements for atomic property scaling
    n_property: int
        Number of output vector features.
    n_residual: int
        Number of residual layers for transformation from atomic feature vector
        to output results.
    activation_fn: callable
        Residual layer activation function.
    atomic_scaling: bool
        Prediction shift term and scaling factor per atom type (True) or
        universal (False)
    trainable_scaling: bool
        If True, the scaling factor and shift term parameters are trainable.
    split_scaling: bool
        If True, each property are scaled by an individual set of scaling
        factors and shift terms. This option has no effect for n_property=1.
    device: str
        Device type for model variable allocation
    dtype: dtype object
        Model variables data type

    """

    def __init__(
        self,
        n_blocks: int,
        n_atombasis: int,
        n_maxatom: int,
        n_property: int,
        n_residual: int,
        activation_fn: Callable,
        atomic_scaling: bool,
        trainable_scaling: bool,
        split_scaling: bool,
        device: str,
        dtype: 'dtype',
    ):

        super(OutputBlock, self).__init__()

        # Assign activation function
        if activation_fn is None:
            self.activation_fn = torch.nn.Identity()
        else:
            self.activation_fn = activation_fn
        
        # Assign module variable parameters from configuration
        self.device = device
        self.dtype = dtype

        # Residual layer for atomic feature modification
        self.residuals = torch.nn.ModuleList([
                torch.nn.Sequential(
                    *[
                        ResidualLayer(
                            n_atombasis,
                            self.activation_fn,
                            True,
                            self.device,
                            self.dtype)
                        for _ in range(n_residual)
                    ])
                for _ in range(n_blocks)])

        # Dense layer for transforming atomic feature vector to result vector
        self.output = torch.nn.ModuleList(
            [
                DenseLayer(
                    n_atombasis,
                    n_property,
                    self.activation_fn,
                    False,
                    self.device,
                    self.dtype,
                    weight_init=torch.nn.init.xavier_normal_,
                    kwargs_weight_init={'gain': 1.E-1}
                    )
                for _ in range(n_blocks)
            ])

        # Assign property dimension to register
        self.register_buffer(
            "n_property",
            torch.tensor(n_property, device=self.device, dtype=torch.int64))

        # Assign output scaling parameter
        self.atomic_scaling = atomic_scaling
        self.trainable_scaling = trainable_scaling
        self.split_scaling = split_scaling
        if self.atomic_scaling:
            if self.split_scaling and self.n_property > 1:
                scaling = torch.tensor(
                    [
                        [[0.0, 1.0]]*n_property
                        for _ in torch.arange(0, n_maxatom + 1)
                    ],
                    device=self.device, dtype=self.dtype)
            else:
                scaling = torch.tensor(
                    [[0.0, 1.0] for _ in torch.arange(0, n_maxatom + 1)],
                    device=self.device, dtype=self.dtype)
        else:
            if self.split_scaling and self.n_property > 1:
                scaling = torch.tensor(
                    [[0.0, 1.0]]*n_property,
                    device=self.device, dtype=self.dtype)
            else:
                scaling = torch.tensor(
                    [0.0, 1.0],
                    device=self.device, dtype=self.dtype)
        self.set_scaling(scaling)

        return

    def set_scaling(
        self,
        scaling: torch.Tensor,
    ):
        """
        Set scaling parameter

        Parameters
        ----------
        scaling: torch.tensor
            Scaling parameter

        """
        if self.trainable_scaling:
            self.scaling = torch.nn.Parameter(scaling)
        else:
            self.scaling = self.register_buffer("scaling", scaling)

        return

    def forward(
        self,
        features_list: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply output block.
        
        Parameters
        ----------
        features_list: torch.Tensor(N_blocks, N_atoms, n_atombasis)
            List of atomic feature vectors per message passing cycle
        atomic_numbers: torch.Tensor(N_atoms)
            List of atomic numbers
        
        Returns
        -------
        torch.Tensor(N_atoms, n_results)
            Transformed atomic feature vector to result vector
        torch.Tensor()
            Loss value to benfit decreasing result amplitudes with each cycle
        
        """

        # Initialize results variable
        results = torch.zeros(
            self.n_property, device=self.device, dtype=self.dtype)

        # Initialize training variables
        nhloss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        last_results2 = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Iterate over feature vector cylces
        for icycle, (residual_i, output_i) in enumerate(
            zip(self.residuals, self.output)
        ):
            
            # Grep feature list
            features_i = features_list[icycle]
            
            # Apply residual layers on atomic features
            features_i = residual_i(features_i)
        
            # Apply last activation function
            features_i = self.activation_fn(features_i)

            # Transform to result vector
            results_i = output_i(features_i)

            # Flatten result
            if self.n_property == 1:
                results_i = torch.flatten(results_i, start_dim=0)
            
            # Add result
            results = results + results_i

            # If training mode is active, compute nhloss contribution
            if self.training:
                results2 = results_i**2
                if icycle:
                    nhloss = nhloss + torch.mean(
                        results2/(results2 + last_results2 + 1.0e-7))
                last_results2 = results2

        # Apply scaling
        if self.atomic_scaling:
            if self.split_scaling and self.n_property > 1:
                shifts = self.scaling[atomic_numbers][:, :, 0]
                scales = self.scaling[atomic_numbers][:, :, 1]
            else:
                shifts = self.scaling[atomic_numbers][:, 0]
                scales = self.scaling[atomic_numbers][:, 1]
        else:
            if self.split_scaling and self.n_property > 1:
                shifts = self.scaling[:, 0].unsqueeze(0)
                scales = self.scaling[:, 1].unsqueeze(0)
            else:
                shifts = self.scaling[0]
                scales = self.scaling[1]
        results = results*scales + shifts

        return results, nhloss
