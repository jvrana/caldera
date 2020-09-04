"""_fully_connected.py.

Make graph fully connected
"""
from typing import overload

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils import add_edges


class Undirected(TransformBase):
    def __init__(self, fill_value: float = 0.0):
        super().__init__()
        self.fill_value = fill_value

    @overload
    def transform_each(self, data: GraphData) -> GraphData:
        ...

    def transform(self, data: GraphBatch) -> GraphBatch:
        return add_edges(data, self.fill_value, kind="undirected")
