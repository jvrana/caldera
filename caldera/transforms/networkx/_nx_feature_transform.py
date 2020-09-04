from typing import Callable
from typing import Generator
from typing import Tuple
from typing import TypeVar

import networkx as nx

from ._base import NetworkxTransformBase
from caldera.utils import functional as Fn
from caldera.utils.nx import nx_copy

T = TypeVar("T")
TupleGen = Generator[Tuple, None, None]
GraphGen = Generator[nx.Graph, None, None]


class NetworkxTransformFeatures(NetworkxTransformBase):
    def __init__(
        self,
        node_transform: Callable[[TupleGen], TupleGen] = None,
        edge_transform: Callable[[TupleGen], TupleGen] = None,
        global_transform: Callable[[TupleGen], TupleGen] = None,
    ):
        """.. code-block:: python.

            def only_self_loops(edges):
                for e1, e2, edata in edges:
                    if e1 == e2:
                        yield e1, e2, edata

            transform = nx_transform(edge_transform=only_self_loops)
            transform(graphs)

        Alternatively, using the functional programming module:

        .. code-block:: python
            from caldera.utils.functional import Functional

            only_self_loops = Fn.filter_each(lambda x: x[0] == x[1])
            transform = nx_transform(edge_transform=only_self_loops)
            transform(graphs)

        :param node_transform:
        :param edge_transform:
        :param global_transform:
        :param kwargs:
        :return:
        """
        self.node_transform = node_transform
        self.edge_transform = edge_transform
        self.global_transform = global_transform

    def transform(self, g):
        return nx_copy(
            g,
            None,
            node_transform=self.node_transform,
            edge_transform=self.edge_transform,
            global_transform=self.global_transform,
        )


class NetworkxTransformFeatureData(NetworkxTransformFeatures):
    def __init__(
        self,
        node_transform: Callable[[TupleGen], TupleGen] = None,
        edge_transform: Callable[[TupleGen], TupleGen] = None,
        global_transform: Callable[[TupleGen], TupleGen] = None,
    ):
        super().__init__(
            node_transform=Fn.map_each(node_transform),
            edge_transform=Fn.map_each(edge_transform),
            global_transform=Fn.map_each(global_transform),
        )
