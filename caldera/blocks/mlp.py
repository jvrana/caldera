from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from torch import nn

from caldera.defaults import CalderaDefaults as D
from caldera.utils import pairwise


class MLPBlock(nn.Module):
    """A multilayer perceptron block."""

    def __init__(
        self,
        input_size: int,
        output_size: int = None,
        layer_norm: bool = True,
        dropout: float = None,
        activation: Callable = D.activation,
    ):
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


class MLP(nn.Module):
    """A multilayer perceptron."""

    def __init__(
        self,
        *latent_sizes: List[int],
        layer_norm: bool = True,
        dropout: float = None,
        activation: Callable = D.activation
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                MLPBlock(
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
