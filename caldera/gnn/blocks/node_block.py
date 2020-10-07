from typing import Optional

import torch
from torch import nn

from caldera.data import GraphBatch
from caldera.gnn.blocks.aggregator import Aggregator
from caldera.gnn.blocks.block import Block


class NodeBlock(Block):
    def __init__(self, module: nn.Module):
        super().__init__({"module": module}, independent=True)

    def forward(self, node_attr):
        return self.block_dict["module"](node_attr)

    def forward_from_data(self, data: GraphBatch):
        return self(data.x)


class AggregatingNodeBlock(NodeBlock):
    def __init__(self, module: nn.Module, edge_aggregator: Optional[Aggregator] = None):
        """Aggregating version of the NodeBlock.

        :param module: any torch.nn.Module
        :param edge_aggregator: edge aggregation function
        """
        super().__init__(module)
        self.block_dict["edge_aggregator"] = edge_aggregator
        self._independent = False

    # TODO: source_to_dest or dest_to_source (isn't this just reversing the graph?)
    def forward(
        self,
        *,
        node_attr,
        edge_attr,
        edges,
        global_attr: torch.Tensor = None,
        node_idx: torch.Tensor = None,
    ):
        aggregated = (
            self.block_dict["edge_aggregator"](
                edge_attr, edges[1], dim=0, dim_size=node_attr.size(0)
            ),
        )
        if global_attr is not None:
            if node_idx is None:
                raise RuntimeError(
                    "Must provide `node_index` if providing `global_attr`"
                )
            aggregated += (global_attr[node_idx],)
        out = torch.cat([node_attr, *aggregated], dim=1)
        return self.block_dict["module"](out)

    def forward_from_data(self, data: GraphBatch):
        return self(data.x, data.e, data.edges)
