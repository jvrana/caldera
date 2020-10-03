r"""
This module provide objects for representing graphs as `torch` tensors.

.. autosummary::
    :toctree: _generated/

    GraphBatch
    GraphData
    GraphTuple
    GraphDataLoader
"""
from ._graph_batch import GraphBatch
from ._graph_data import GraphData
from ._graph_tuple import GraphTuple
from ._loader import GraphDataLoader

__all__ = [
    "GraphBatch",
    "GraphData",
    "GraphTuple",
    "GraphDataLoader",
]


def method():
    """This is bullshit."""


this_is_my_attribute = "a tribute"
