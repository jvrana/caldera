r"""
Dataset (:mod:`caldera.dataset`)
==============================

.. currentmodule:: caldera.dataset

Graphdata sets

.. autosummary::
    :toctree: generated

    GraphDataset
    GraphBatchDataset
"""
from ._graph_dataset import GraphBatchDataset
from ._graph_dataset import GraphDataset

__all__ = ["GraphDataset", "GraphBatchDataset"]
