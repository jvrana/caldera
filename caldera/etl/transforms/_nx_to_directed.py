from typing import Optional
from typing import Type

from ._base import NetworkxTransformBase
from caldera.utils.nx import nx_to_directed


class NetworkxToDirected(NetworkxTransformBase):
    def __init__(self, graph_class: Optional[Type] = None):
        self.graph_class = graph_class

    def transform(self, graphs):
        for g in graphs:
            yield nx_to_directed(g)
