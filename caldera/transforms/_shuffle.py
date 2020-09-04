"""_shuffle.py.

Shuffle the node, edge, and graph labels.
"""
from typing import overload

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData


class Shuffle(TransformBase):
    @overload
    def transform_each(self, data: GraphData) -> GraphData:
        ...

    def transform(self, data: GraphBatch) -> GraphBatch:
        return data.shuffle()
