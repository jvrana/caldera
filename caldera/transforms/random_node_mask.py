"""
random_node_mask.py

Apply random node mask to the graph, removing relevant nodes.
"""
from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload


class RandomNodeMask(TransformBase):
    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        pass
