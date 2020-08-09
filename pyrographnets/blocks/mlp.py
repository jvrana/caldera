from typing import List

from torch import nn

from pyrographnets.utils import pairwise
from typing import Union, Optional

class MLPBlock(nn.Module):
    """A multilayer perceptron block."""

    def __init__(self, input_size: int,
                 output_size: int = None,
                 layer_normalization: bool = True,
                 dropout: float = None
        ):
        super().__init__()
        if output_size is None:
            output_size = input_size
        layers = [nn.Linear(input_size, output_size), nn.ReLU()]
        if layer_normalization:
            layers.append(nn.LayerNorm(output_size))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    """A multilayer perceptron."""

    def __init__(self, *latent_sizes: List[int], layer_norm: bool = True, dropout: float = None):
        super().__init__()
        self.layers = nn.Sequential(
            *[MLPBlock(n1, n2, layer_norm, dropout) for n1, n2 in pairwise(latent_sizes)]
        )

    def forward(self, x):
        return self.layers(x)
