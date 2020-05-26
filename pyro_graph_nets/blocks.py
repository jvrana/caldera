import torch
from typing import *
import itertools
from torch_scatter import scatter_mean
import torch_scatter
from torch import nn
from pyro_graph_nets.utils import pairwise


class MLPBlock(nn.Module):
    """
    A multilayer perceptron block
    """
    def __init__(self, input_size: int, output_size: int = None):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.blocks = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.LayerNorm(output_size)
        )

    def forward(self, x):
        return self.blocks(x)


class MLP(nn.Module):
    """
    A multilayer perceptron

    """
    def __init__(self, *latent_sizes: List[int]):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MLPBlock(n1, n2) for n1, n2 in pairwise(latent_sizes)]
        )

    def forward(self, x):
        return self.blocks(x)

# TODO: have the NN select the appropriate aggregation!
class Aggregator(nn.Module):
    """Aggregation layer"""

    valid_aggregators = {
        'mean': torch_scatter.scatter_mean,
        'max': torch_scatter.scatter_max,
        'min': torch_scatter.scatter_min,
        'add': torch_scatter.scatter_add
    }

    def __init__(self, aggregator: str, dim: int = None, dim_size: int = None):
        super().__init__()
        if aggregator not in self.valid_aggregators:
            raise ValueError(
                "Aggregator '{}' not not one of the valid aggregators {}".format(
                    aggregator, self.valid_aggregators))
        self.aggregator = aggregator
        self.kwargs = dict(
            dim=dim,
            dim_size=dim_size
        )

    def forward(self, x, indices, **kwargs):
        func_kwargs = dict(self.kwargs)
        func_kwargs.update(kwargs)
        func = self.valid_aggregators[self.aggregator]
        return func(x, indices, **func_kwargs)


class Block(nn.Module):

    def __init__(self, module_dict: Dict[str, nn.Module], independent: bool):
        self._independent = independent
        self.block_dict = nn.ModuleDict(dict(module_dict))


class EdgeModel(Block):

    def __init__(self, input_size: int, layers: List[int], independent: bool):
        super().__init__(
            {
                'mlp': MLP(input_size, *layers)
            },
            independent=independent
        )

    def forward(self, src, dest, edge_attr, u, batch):
        if not self._independent:
            out = torch.cat([src, dest, edge_attr], 1)
        else:
            out = edge_attr
        return self.block_dict['mlp'](out)

# TODO: add global features
class NodeModel(Block):

    def __init__(self, input_size: int, layers: List[int], independent: bool):
        super().__init__({
            'aggregator': Aggregator('mean'),
            'mlp': MLP(input_size, *layers)
        }, independent=independent)

    def forward(self, v, edge_index, edge_attr, u, barch):
        if not self._independent:
            row, col = edge_index
            aggregated = self.blocks['aggregator'](edge_attr, col, dim=0,
                                                   dim_size=v.size(0))
            out = torch.cat([aggregated, v], dim=1)
        else:
            out = v
        return self.block_dict['mlp'](out)


class GlobalModel(Block):
    def __init__(self, input_size: int, layers: List[int], independent: bool):
        super().__init__({
            'node_aggregator': Aggregator('mean'),
            'mlp': MLP(input_size, *layers)
        }, independent=independent)

    def forward(self, x, edge_index, edge_attr, u, batch):
        if not self._independent:
            out = torch.cat([u, self.blocks['node_aggregator'](x, batch, dim=0)], dim=1)
        else:
            out = u
        return torch.block_dict['mlp'](out)
