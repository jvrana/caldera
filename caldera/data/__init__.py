r"""
Data (:mod:`caldera.data`)
==============================

.. currentmodule:: caldera.data

This module provide objects for representing graphs as :module:`torch` tensors.

.. autosummary::
    :toctree: generated/
"""


from ._graph_batch import GraphBatch
from ._graph_data import GraphData
from ._graph_dataset import GraphBatchDataset
from ._graph_dataset import GraphDataset
from ._graph_tuple import GraphTuple
from ._loader import GraphDataLoader

__all__ = [
    'GraphBatch',
    'GraphData',
    'GraphBatchDataset',
    'GraphDataset',
    'GraphTuple',
    'GraphDataLoader',
]
