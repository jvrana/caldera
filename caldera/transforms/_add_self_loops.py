"""_add_self_loops.py.

Add self loops to graph.
"""
from typing import overload

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils import add_edges


class AddSelfLoops(TransformBase):
    def __init__(self, fill_value: float = 0.0):
        super().__init__()
        self.fill_value = fill_value

    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:  # noqa: E811
        return add_edges(data, self.fill_value, kind="self")
