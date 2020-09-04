from typing import Optional
from typing import Type

from ._base import NetworkxTransformBase
from caldera.utils.nx import nx_to_undirected


class NetworkxToUndirected(NetworkxTransformBase):
    def __init__(self, graph_class: Optional[Type] = None):
        self.graph_class = graph_class

    def transform(self, g):
        if self.graph_class is None:
            kwargs = {}
        else:
            kwargs = dict(graph_class=self.graph_class)
        return nx_to_undirected(g, **kwargs)
