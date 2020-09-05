"""transforms.py.

Methods for transforming GraphData and GraphBatch
"""
from ._add_self_loops import AddSelfLoops
from ._base import TransformCallable
from ._compose import Compose

r"""
Data (:mod:`caldera.transform`)
==============================

.. currentmodule:: caldera.transforms

Data transforms.

.. autosummary::
    :toctree: generated/
"""


from ._fully_connected import FullyConnected
from ._random_edge_mask import RandomEdgeMask
from ._random_hop import RandomHop
from ._random_node_mask import RandomNodeMask
from ._reverse import Reverse
from ._shuffle import Shuffle
from ._undirected import Undirected

__all__ = [
    'AddSelfLoops',
    'TransformCallable',
    'Compose',
    'FullyConnected',
    'RandomHop',
    'RandomEdgeMask',
    'RandomNodeMask',
    'Reverse',
    'Shuffle',
    'Undirected',
]
