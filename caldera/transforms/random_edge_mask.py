"""
random_edge_mask.py

Apply random edge mask to the graph.
"""
from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload


class RandomEdgeMask(TransformBase):

    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        pass
