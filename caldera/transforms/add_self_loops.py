"""
add_self_loops.py

Add self loops to graph.
"""


from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload


class AddSelfLoops(TransformBase):

    @overload
    def __call__(self, data: GraphData) -> GraphData:
        ...

    def __call__(self, data: GraphBatch) -> GraphBatch:
        pass
