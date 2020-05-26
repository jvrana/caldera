import torch
import torch_scatter
from torch import nn
from pyro_graph_nets.utils import pairwise
from typing import List, Dict

# TODO: rename arguments to v, e, u
# TODO: incoporate aggregation of global attributes
# TODO: edge aggregation at global block
# TODO: clean interface for removing or adding various blocks
# TODO: have the NN select the appropriate aggregation!
# TODO: demonstration of Tensorboard
# TODO: different types of aggregations (source and node?)

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
        super().__init__()
        self._independent = independent
        self.block_dict = nn.ModuleDict({name: mod for name, mod in module_dict.items() if mod is not None})


class EdgeBlock(Block):

    def __init__(self, input_size: int, layers: List[int], independent: bool):
        super().__init__(
            {
                'mlp': MLP(input_size, *layers)
            },
            independent=independent
        )

    def forward(self, src, dest, edge_attr, u, node_idx, edge_idx):
        if not self._independent:
            out = torch.cat([src, dest, edge_attr], 1)
        else:
            out = edge_attr
        results = self.block_dict['mlp'](out)
        return results


# TODO: add global features
class NodeBlock(Block):

    def __init__(self, input_size: int, layers: List[int], independent: bool, edge_aggregator: Aggregator = None):
        """

        :param input_size:
        :param layers:
        :param edge_aggregator:
        :param independent:
        """
        super().__init__({
            'edge_aggregator': edge_aggregator,
            'mlp': MLP(input_size, *layers)
        }, independent=independent)

    def forward(self, v, edge_index, edge_attr, u, node_idx, edge_idx):
        if not self._independent:
            row, col = edge_index
            aggregator_fn = self.block_dict['edge_aggregator']
            if aggregator_fn:
                aggregated = self.block_dict['edge_aggregator'](edge_attr, col, dim=0,
                                                       dim_size=v.size(0))
                out = torch.cat([aggregated, v], dim=1)
            else:
                out = torch.cat([v], dim=1)
        else:
            out = v
        return self.block_dict['mlp'](out)


class GlobalBlock(Block):
    def __init__(self, input_size: int, layers: List[int], independent: bool,
                 node_aggregator: Aggregator = None, edge_aggregator: Aggregator = None):
        super().__init__({
            'node_aggregator': node_aggregator,
            'edge_aggregator': edge_aggregator,
            'mlp': MLP(input_size, *layers)
        }, independent=independent)

    def forward(self, node_attr, edge_index, edge_attr, u, node_idx, edge_idx):
        if not self._independent:
            node_agg = self.block_dict['node_aggregator']
            edge_agg = self.block_dict['edge_aggregator']
            to_cat = [u]
            if node_agg is not None:
                to_cat.append(node_agg(node_attr, node_idx, dim=0))
            if edge_agg is not None:
                to_cat.append(edge_agg(edge_attr, edge_idx, dim=0))
            out = torch.cat(to_cat, dim=1)
        else:
            out = u
        return self.block_dict['mlp'](out)
