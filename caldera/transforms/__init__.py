r"""
Transforms (:mod:`caldera.transform`)
=====================================

.. currentmodule:: caldera.transforms

Methods for transforming GraphData and GraphBatch

GraphData and GraphBatch
------------------------

Transformation classes for :class:`caldera.data.GraphData` and :class:`caldera.data.GraphBatch`.

.. autosummary::
    :toctree: generated/

   Shuffle
   Reverse
   Undirected
   FullyConnected
   RandomEdgeMask
   RandomNodeMask
   RandomHop

Preprocessing Transforms
------------------------

Networkx
________

Transforms on :class:`networkx.Graph` instances

.. autosummary::

   networkx
"""
from ._add_self_loops import AddSelfLoops
from ._base import TransformCallable
from ._compose import Compose
from ._fully_connected import FullyConnected
from ._random_edge_mask import RandomEdgeMask
from ._random_hop import RandomHop
from ._random_node_mask import RandomNodeMask
from ._reverse import Reverse
from ._shuffle import Shuffle
from ._undirected import Undirected

__all__ = [
    "AddSelfLoops",
    "TransformCallable",
    "Compose",
    "FullyConnected",
    "RandomHop",
    "RandomEdgeMask",
    "RandomNodeMask",
    "Reverse",
    "Shuffle",
    "Undirected",
]
