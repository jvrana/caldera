"""
random_hop.py

Choose random nodes, and induce the graph on those nodes to "h" hopes.
"""

from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload


class RandomHop(TransformBase):

    def __init__(self, n_nodes: int, n_hops: int):
        self.n_nodes = n_nodes
        self.n_hops = n_hops

    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        pass
