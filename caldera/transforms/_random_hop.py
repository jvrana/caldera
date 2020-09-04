"""_random_hop.py.

Choose random nodes, and induce the graph on those nodes to "h" hopes.
"""
from typing import Dict
from typing import Optional
from typing import overload

import torch

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils import induce


class RandomHop(TransformBase):
    def __init__(self, n_nodes: int, n_hops: int):
        """Choose subgraphs from a graph.

        :param n_nodes:
        :param n_hops:
        """
        self.n_nodes = n_nodes
        self.n_hops = n_hops

    @overload
    def transform_each(self, data: GraphData, edge_dict: ...) -> GraphData:
        ...

    def transform(
        self, data: GraphBatch, edge_dict: Optional[Dict] = None
    ) -> GraphBatch:
        if data.num_nodes == 0:
            return data.clone()
        src = torch.randint(data.num_nodes, (self.n_nodes,))
        nodes = induce(data, src, self.n_hops, edge_dict=edge_dict)
        node_mask = torch.BoolTensor([False] * data.num_nodes)
        node_mask[nodes] = True
        return data.apply_node_mask(node_mask)
