import networkx as nx

from ._nx_apply_to_graph import NetworkxApply


class NetworkxFreeze(NetworkxApply):
    def __init__(self):
        super().__init__(func=nx.freeze)
