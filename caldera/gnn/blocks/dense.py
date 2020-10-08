from collections import OrderedDict
from typing import Callable
from typing import List

from torch import nn

from .sequential_module_dict import SequentialModuleDict
from caldera.defaults import CalderaDefaults as D
from caldera.utils import pairwise


class DenseBlock(nn.Module):
    """A dense block comprised of linear and activation functions.

    .. versionchanged: 0.1.0a0

        Renamed from MLPBlock to DenseBlock
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = None,
        layer_norm: bool = True,
        dropout: float = None,
        activation: Callable = D.activation,
    ):
        """Initialize a single layer of a dense block.

        Linear -> Activation -> [Optional]LayerNorm -> [Optional]Dropout

        :param input_size: input data size
        :param output_size: output data size
        :param layer_norm: whether to apply layer normalization to each layer
        :param dropout: optional dropout to apply to each layer
        :param activation: activation function to use after each linear function
        """
        super().__init__()
        if output_size is None:
            output_size = input_size
        dense = SequentialModuleDict({})
        dense["linear"] = nn.Linear(input_size, output_size)
        dense["activation"] = activation()
        if layer_norm:
            dense["layer_norm"] = nn.LayerNorm(output_size, elementwise_affine=False)
        if dropout is not None:
            dense["dropout"] = nn.Dropout(dropout)
        self.dense = dense

    def forward(self, x):
        return self.dense(x)


class Dense(nn.Module):
    """A dense module comprised of multiple linear and activation layers.

    .. versionchanged:: 0.1.0a0     Renamed from MLP to Dense
    """

    def __init__(
        self,
        *latent_sizes: List[int],
        layer_norm: bool = True,
        dropout: float = None,
        activation: Callable = D.activation
    ):
        """Initialize a Dense neural network. For each layer, implements:

        Linear -> Activation -> [Optional]LayerNorm -> [Optional]Dropout

        :param latent_sizes: list of latent sizes to use for the linear layers
        :param layer_norm: whether to apply layer normalization to each layer
        :param dropout: optional dropout to apply to each layer
        :param activation: activation function to use after each linear function
        """
        super().__init__()
        self.dense_layers = nn.Sequential(
            *[
                DenseBlock(
                    n1,
                    n2,
                    layer_norm=layer_norm,
                    dropout=dropout,
                    activation=activation,
                )
                for n1, n2 in pairwise(latent_sizes)
            ]
        )

    def forward(self, x):
        return self.dense_layers(x)
