r"""
Transformation classes for :class:`caldera.data.GraphData` and :class:`caldera.data.GraphBatch`.

.. autosummary::
    :toctree: _generated/

   Shuffle
   Reverse
   Undirected
   FullyConnected
   RandomEdgeMask
   RandomNodeMask
   RandomHop
"""
from ._add_self_loops import AddSelfLoops
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
    "Compose",
    "FullyConnected",
    "RandomHop",
    "RandomEdgeMask",
    "RandomNodeMask",
    "Reverse",
    "Shuffle",
    "Undirected",
]
