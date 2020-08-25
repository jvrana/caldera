"""_shuffle.py.

Shuffle the node, edge, and graph labels.
"""
from typing import overload

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData


class Shuffle(TransformBase):
    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        return data.shuffle()
