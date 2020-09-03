import functools
from typing import Any
from typing import Callable
from typing import Generator
from typing import Optional
from typing import Tuple

import networkx as nx
import numpy as np

from ._utils import merge_update
from ._utils import values_to_one_hot
from caldera.utils import functional as Fn
from caldera.utils.nx import nx_copy

TupleGen = Generator[Tuple, None, None]
GraphGen = Generator[nx.Graph, None, None]


def nx_transform(
    node_transform: Callable[[TupleGen], TupleGen] = None,
    edge_transform: Callable[[TupleGen], TupleGen] = None,
    global_transform: Callable[[TupleGen], TupleGen] = None,
    **kwargs
) -> Callable[[GraphGen], GraphGen]:
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

    def _nx_transform(graphs):
        yield from Fn.map_each(
            lambda g: nx_copy(
                g,
                None,
                node_transform=node_transform,
                edge_transform=edge_transform,
                global_transform=global_transform,
                **kwargs
            )
        )(graphs)

    return _nx_transform


def nx_transform_each(
    node_transform: Callable[[Tuple], Tuple] = None,
    edge_transform: Callable[[Tuple], Tuple] = None,
    global_transform: Callable[[Tuple], Tuple] = None,
    **kwargs
) -> Callable[[GraphGen], GraphGen]:
    """.. code-block:: python.

        node_keys_to_str = nx_transform_each(node_transform=lambda x: (str(x[0]), x[1]))
        new_graphs = list(node_keys_to_str(graphs))

    :param node_transform:
    :param edge_transform:
    :param global_transform:
    :param kwargs:
    :return:
    """

    def _nx_transform_each(graphs):
        yield from Fn.map_each(
            lambda g: nx_copy(
                g,
                None,
                node_Transform=Fn.map_each(node_transform),
                edge_transform=Fn.map_each(edge_transform),
                global_transform=Fn.map_each(global_transform),
                **kwargs
            )
        )(graphs)

    return _nx_transform_each


def nx_collect_np_features_in_place(
    feature: str,
    from_key: str,
    to_key: str,
    *,
    default: Any = ...,
    encoding: Optional[str] = None,
    processing_func=None,
    processing_kwargs=None,
    join_func: str = "hstack",
    global_key: str = None,
    **kwargs
) -> Callable[[GraphGen], GraphGen]:
    if processing_kwargs is None:
        processing_kwargs = {}

    if encoding is None:
        processing_func = Fn.compose(list, np.array)
    elif encoding == "onehot":
        processing_func = Fn.compose(
            list, functools.partial(values_to_one_hot, **kwargs)
        )

    if join_func == "hstack":

        def join_func(a, b):
            return np.hstack([a, b])

    elif join_func == "vstack":

        def join_func(a, b):
            return np.vstack([a, b])

    _NODE, _EDGE, _GLOBAL = "node", "edge", "global"

    def _in_place(g: nx.Graph) -> nx.Graph:
        if feature == _NODE:
            data = g.nodes(data=True)
        elif feature == _EDGE:
            data = g.edges(data=True)
        elif feature == _GLOBAL:
            data = [(0, g.get_global(global_key))]
        else:
            raise ValueError(
                "feature '{}' not recognized. Select from {}.".format(
                    feature, [_NODE, _EDGE, _GLOBAL]
                )
            )

        merge_update(
            data,
            from_key,
            to_key,
            join_fn=join_func,
            process_fn=processing_func,
            default=default,
        )
        return g

    def _nx_collect_np_features_in_place(graphs):
        yield from Fn.map_each(_in_place)(graphs)

    return _nx_collect_np_features_in_place
