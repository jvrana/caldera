"""transforms.py.

Methods for transforming GraphData and GraphBatch
"""
from .fully_connected import FullyConnected
from .undirected import Undirected
from .reverse import Reverse
from .add_self_loops import AddSelfLoops
from .compose import Compose
from .random_hop import RandomHop
from .random_node_mask import RandomNodeMask
from .random_edge_mask import RandomEdgeMask
from .shuffle import Shuffle
