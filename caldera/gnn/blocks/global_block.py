from typing import Optional

import torch
from torch import nn

from caldera.data import GraphBatch
from caldera.gnn import Aggregator
from caldera.gnn.blocks.block import Block


class GlobalBlock(Block):
    def __init__(self, module: nn.Module):
        super().__init__({"module": module}, independent=True)

    def forward(self, global_attr):
        return self.block_dict["module"](global_attr)

    def forward_from_data(self, data: GraphBatch):
        return self(data.g)


# TODO: determine which aggregator to use during training (some function of attributes -> one-hot)
# TODO: better documentation
class AggregatingGlobalBlock(GlobalBlock):
    def __init__(
        self,
        module,
        edge_aggregator: Optional[Aggregator] = None,
        node_aggregator: Optional[Aggregator] = None,
    ):
        """Aggregating global block.

        :param module: any torch.nn.Module
        :param edge_aggregator: edge aggregation module
        :param node_aggregator: node aggregation module
        """
        super().__init__(module)
        self.block_dict["edge_aggregator"] = edge_aggregator
        self.block_dict["node_aggregator"] = node_aggregator
        self._independent = False

    def forward(self, *, global_attr, node_attr, edge_attr, edges, node_idx, edge_idx):
        aggregated = [global_attr]
        if "node_aggregator" in self.block_dict:
            aggregated.append(
                self.block_dict["node_aggregator"](
                    node_attr, node_idx, dim=0, dim_size=global_attr.shape[0]
                )
            )
        if "edge_aggregator" in self.block_dict:
            aggregated.append(
                self.block_dict["edge_aggregator"](
                    edge_attr, edge_idx, dim=0, dim_size=global_attr.shape[0]
                )
            )

        out = torch.cat(aggregated, dim=1)
        return self.block_dict["module"](out)

    def forward_from_data(self, data: GraphBatch):
        return self(data.g, data.x, data.e, data.edges, data.node_idx, data.edge_idx)
