"""
reverse.py

Reverse the direction of the edges.
"""
from .base import TransformBase
from caldera.data import GraphBatch, GraphData
from typing import overload


class Reverse(TransformBase):

    @overload
    def __call__(self, data: GraphData):
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        pass