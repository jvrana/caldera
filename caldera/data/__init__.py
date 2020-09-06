r"""
Data (:mod:`caldera.data`)
==============================

.. currentmodule:: caldera.data

This module provide objects for representing graphs as `torch` tensors.

.. autosummary::
    :toctree: generated/

    GraphData
    GraphBatch
    GraphDataset
    GraphBatchDataset
    GraphTuple
    GraphDataLoader

Utilities
---------

Utility functions for :class:`GraphData` and :class:`GraphBatch`

.. autosummary::
    :toctree: generated/

    utils.add_edges
    utils.floyd_warshall
    utils.adj_matrix
    utils.adj_matrix_from_edges
    utils.graph_matrix
    utils.in_degree
    utils.in_degree_matrix_from_edges
    utils.out_degree
    utils.out_degree_matrix_from_edges
    utils.to_sparse_coo_matrix
    utils.bfs_nodes
    utils.induce
    utils.neighbors
    utils.tensor_induce
    utils.get_edge_dict
"""
from ._graph_batch import GraphBatch
from ._graph_data import GraphData
from ._graph_dataset import GraphBatchDataset
from ._graph_dataset import GraphDataset
from ._graph_tuple import GraphTuple
from ._loader import GraphDataLoader

__all__ = [
    "GraphBatch",
    "GraphData",
    "GraphBatchDataset",
    "GraphDataset",
    "GraphTuple",
    "GraphDataLoader",
]
