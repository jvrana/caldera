import torch
from torch import nn

from pyrographnets.blocks.block import Block
from pyrographnets.data import GraphData


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