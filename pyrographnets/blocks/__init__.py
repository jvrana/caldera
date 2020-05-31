from torch import nn
import torch
from typing import Dict
from typing import List
from pyrographnets.utils import pairwise

class MLPBlock(nn.Module):
    """A multilayer perceptron block."""

    def __init__(self, input_size: int, output_size: int = None):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.blocks = nn.Sequential(
            nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size)
        )

    def forward(self, x):
        return self.blocks(x)


class MLP(nn.Module):
    """A multilayer perceptron."""

    def __init__(self, *latent_sizes: List[int]):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MLPBlock(n1, n2) for n1, n2 in pairwise(latent_sizes)]
        )

    def forward(self, x):
        return self.blocks(x)

class Block(nn.Module):
    def __init__(self, module_dict: Dict[str, nn.Module], independent: bool):
        super().__init__()
        self._independent = independent
        self.block_dict = nn.ModuleDict(
            {name: mod for name, mod in module_dict.items() if mod is not None}
        )

    @property
    def independent(self):
        return self._independent


class EdgeBlock(Block):
    def __init__(self, mlp: nn.Module):
        super().__init__({"mlp": mlp}, independent=True)

    def forward(self, edge_attr: torch.tensor, node_attr: torch.tensor = None, edges: torch.tensor = None):
        results = self.block_dict["mlp"](edge_attr)
        return results

    def forward_from_data(self, data: GraphData):
        return self(data.e, data.x, data.edges)


class AggregatingEdgeBlock(EdgeBlock):

    def __init__(self, mlp: nn.Module):
        super().__init__(mlp)
        self._independent = False

    def forward(self, edge_attr: torch.tensor, node_attr: torch.tensor, edges: torch.tensor):
        out = torch.cat([node_attr[edges[0]], node_attr[edges[1]], edge_attr], 1)
        return self.block_dict['mlp'](out)

    def forward_from_data(self, data: GraphData):
        return self(data.e, data.x.data.edges)


# TODO: this is redundent with EdgeBlock
class NodeBlock(Block):

    def __init__(self, mlp: nn.Module):
        super().__init__({
            'mlp': mlp
        }, independent=True)

    def forward(self, node_attr):
        return self.block_dict['mlp'](node_attr)

    def forward_from_data(self, data: GraphBatch):
        return self(data.x)


class AggregatingNodeBlock(NodeBlock):

    def __init__(self, mlp: nn.Module, edge_aggregator: Aggregator):
        super().__init__(mlp)
        self.block_dict['edge_aggregator'] = edge_aggregator
        self._independent = False

    # TODO: source_to_target, target_to_source
    def forward(self, node_attr, edge_attr, edges):
        aggregated = self.block_dict['edge_aggregator'](edge_attr, edges[1], dim=0, dim_size=node_attr.size(0))
        out = torch.cat([node_attr, aggregated], dim=1)
        return self.block_dict['mlp'](out)

    def forward_from_data(self, data: GraphBatch):
        return self(data.x, data.e, data.edges)


class GlobalBlock(Block):

    def __init__(self, mlp):
        super().__init__({'mlp': mlp}, independent=True)

    def forward(self, global_attr):
        return self.block_dict['mlp'](global_attr)

    def forward_from_data(self, data: GraphBatch):
        return self(data.g)


class AggregatingGlobalBlock(GlobalBlock):

    def __init__(self, mlp, edge_aggregator=None, node_aggregator=None):
        super().__init__(mlp)
        self.block_dict['edge_aggregator'] = edge_aggregator
        self.block_dict['node_aggregator'] = node_aggregator
        self._independent = False

    def forward(self, global_attr, node_attr, edge_attr, edges, node_idx, edge_idx):
        aggregated = [global_attr]
        if 'node_aggregator' in self.block_dict:
            aggregated.append(
                self.block_dict['node_aggregator'](node_attr, node_idx, dim=0, dim_size=global_attr.shape[0]))
        if 'edge_aggregator' in self.block_dict:
            aggregated.append(
                self.block_dict['edge_aggregator'](edge_attr, edge_idx, dim=0, dim_size=global_attr.shape[0]))

        out = torch.cat(aggregated, dim=1)
        return self.block_dict['mlp'](out)

    def forward_from_data(self, data: GraphBatch):
        return self(data.g, data.x, data.e, data.edges, data.node_idx, data.edge_idx)