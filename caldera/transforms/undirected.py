"""
undirected.py

Convert graph to undirected graph, copying the edge attributes.
"""
from .base import TransformBase
from caldera.data import GraphBatch, GraphData
from typing import overload


class ToUndirected(TransformBase):

    @overload
    def __call__(self, data: GraphData):
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        pass
