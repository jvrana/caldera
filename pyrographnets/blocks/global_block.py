import torch

from pyrographnets.blocks.block import Block
from pyrographnets.data import GraphBatch


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