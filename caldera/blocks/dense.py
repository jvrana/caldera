from typing import Callable
from typing import List

from torch import nn

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
        layers = [nn.Linear(input_size, output_size), activation()]
        if layer_norm:
            layers.append(nn.LayerNorm(output_size))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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
        self.layers = nn.Sequential(
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
        return self.layers(x)
