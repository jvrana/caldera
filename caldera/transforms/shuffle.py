"""
shuffle.py

Shuffle the node, edge, and graph labels.
"""
from .base import TransformBase
from caldera.data import GraphBatch, GraphData
from typing import overload


class Shuffle(TransformBase):
    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        return data.shuffle()
