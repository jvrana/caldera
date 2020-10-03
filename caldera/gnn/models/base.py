from typing import Union

import torch

from caldera.data import GraphBatch
from caldera.data import GraphTuple


class GraphNetworkBase(torch.nn.Module):
    """Base class for GraphNetwork modules."""

    # def reset_parameters(self):
    #     for child in self.children():
    #         if hasattr(child, 'reset_parameters'):
    #             child.reset_parameters()

    def gt_to_batch(self, gt: GraphTuple, batch: GraphBatch) -> GraphBatch:
        return GraphBatch(
            node_attr=gt.x,
            edge_attr=gt.e,
            global_attr=gt.g,
            edges=batch.edges,
            node_idx=batch.node_idx,
            edge_idx=batch.edge_idx,
        )

    def forward(self, data: GraphBatch) -> Union[GraphTuple, GraphBatch]:
        pass
