"""
random_hop.py

Choose random nodes, and induce the graph on those nodes to "h" hopes.
"""

from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload, Dict, Optional
from caldera.data.utils import tensor_induce, induce
import torch


class RandomHop(TransformBase):
    def __init__(self, n_nodes: int, n_hops: int):
        self.n_nodes = n_nodes
        self.n_hops = n_hops

    @overload
    def __call__(self, data: GraphData, edge_dict: ...) -> GraphData:
        ...

    def __call__(
        self, data: GraphBatch, edge_dict: Optional[Dict] = None
    ) -> GraphBatch:
        if data.num_nodes == 0:
            return data.clone()
        src = torch.randint(data.num_nodes, (self.n_nodes,))
        nodes = induce(data, src, self.n_hops, edge_dict=edge_dict)
        node_mask = torch.BoolTensor([False] * data.num_nodes)
        node_mask[nodes] = True
        return data.apply_node_mask(node_mask)
