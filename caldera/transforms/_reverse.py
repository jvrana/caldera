"""_reverse.py.

Reverse the direction of the edges.
"""
from typing import overload

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData


class Reverse(TransformBase):
    @overload
    def transform_each(self, data: GraphData) -> GraphData:
        ...

    def transform(self, data: GraphBatch) -> GraphBatch:
        return data.reverse()
