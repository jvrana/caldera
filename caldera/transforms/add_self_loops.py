"""
add_self_loops.py

Add self loops to graph.
"""


from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload
from caldera.data.utils import add_missing_edges


class AddSelfLoops(TransformBase):
    def __init__(self, fill_value: float = 0.0):
        super().__init__()
        self.fill_value = fill_value

    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        return add_missing_edges(data, self.fill_value, kind="self")
