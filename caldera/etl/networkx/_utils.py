import functools
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import networkx as nx
import numpy as np

from caldera.defaults import CalderaDefaults
from caldera.utils import dict_join
from caldera.utils import functional as Fn
from caldera.utils.nx.types import Graph
from caldera.utils.tensor import to_one_hot


collect_values_by_key = Fn.compose(
    Fn.map_each(lambda x: x.items()),
    Fn.chain_each(),
    Fn.group_each_by_key(lambda x: x[0]),
    Fn.map_each(lambda x: (x[0], [_x[1] for _x in x[1]])),
    dict,
)  # from a list of dictionaries, List[Dict] -> Dict[str, List]


unique_value_types = Fn.compose(
    Fn.index_each(-1),
    collect_values_by_key,
    lambda x: {k: {_v.__class__ for _v in v} for k, v in x.items()},
)


def feature_info(g, global_key: str = None):
    return {
        "node": {"keys": unique_value_types(g.nodes(data=True))},
        "edge": {"keys": unique_value_types(g.edges(data=True))},
        "global": {"keys": unique_value_types([(g.get_global(global_key),)])},
    }


def _raise_if_not_graph(x):
    if not issubclass(x.__class__, nx.Graph):
        raise TypeError(
            "`graphs` must be an iterable of type `nx.Graph`. Found {}".format(
                x.__class__
            )
        )


T = TypeVar("T")
K = TypeVar("K")
S = TypeVar("S")
V = TypeVar("V")


def update_left_inplace(
    data: Dict[K, T], new_data: Dict[K, S], join_fn: Callable[[T, S], K]
) -> Dict[K, Union[T, S, V]]:
    """Updates the left dictionary, joining values with the provided
    function."""
    return dict_join(data, new_data, data, mode="right", join_fn=join_fn)


def values_to_one_hot(
    values: Iterable[T],
    classes: Union[List[T], Tuple[T, ...]],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Convert an iterable of values to a one-hot encoded `np.ndarray`.

    :param values: iterable of values
    :param classes: valid classes of values. Will appear in one-hot array in order they appear here.
    :param num_classes: Number of classes for one-hot encoder. If not provided, number of provided classes
        will be used.
    :return: one-hot encoded array of values.
    """
    assert len(set(classes)) == len(classes)
    d = {k: i for i, k in enumerate(classes)}
    _values = []
    for v in values:
        try:
            _values.append(d[v])
        except KeyError:
            raise KeyError(
                "Value '{}' not found in list of available one-hot classes: {}".format(
                    v, d
                )
            )
    if num_classes is None:
        num_classes = len(d)
    return to_one_hot(np.array(_values), mx=num_classes)


def merge_update(data, key, to_key, join_fn, process_fn, default=...):
    update_fn = functools.partial(update_left_inplace, join_fn=join_fn)

    data1 = Fn.compose(Fn.index_each(-1))(data)

    getter = Fn.get_each(key)
    if default is not ...:
        getter = Fn.get_each(key, default=default)

    data2 = Fn.compose(
        Fn.index_each(-1),
        getter,
        process_fn,
        Fn.map_each(lambda x: {to_key: x}),
    )(data)

    for d1, d2 in zip(data1, data2):
        _ = update_fn(d1, d2)


# def fill_defaults(graph, keys, default: ..., global_key: Optional[str] = None):
#     if default is ...:
#         default = np.array([0.0])
#
#     keys = set(keys)
#
#     x = {
#         "node": graph.nodes(data=True),
#         "edge": graph.edges(data=True),
#         "global": [(graph.get_global(global_key),)],
#     }
#
#     for n_e_or_g, datalist in x.items():
#         for k in keys.difference(_get_unique_keys(datalist)):
#             nx_collect_np_features_in_place(graph, n_e_or_g, None, k, default=default)


_get_unique_keys = Fn.compose(
    Fn.index_each(-1), Fn.chain_each(), Fn.iter_each_unique(), set
)


def setdefault_inplace(d1: Dict[K, T], d2: Dict[K, S]) -> Dict[K, Union[T, S]]:
    return dict_join(d1, d2, d1, join_fn=lambda a, b: a, mode="right")


def add_default_node_data(g: Graph, data: Dict):
    """Update set default node data.

    Will not update if key exists in data.
    """
    for _, ndata in g.nodes(data=True):
        setdefault_inplace(ndata, data)


def add_default_edge_data(g: Graph, data: Dict):
    """Update set default edge data.

    Will not update if key exists in data.
    """
    for _, _, edata in g.edges(data=True):
        setdefault_inplace(edata, data)


def add_default_global_data(
    g: Graph, data: Dict, global_key: str = CalderaDefaults.nx_global_key
):
    """Update set default glboal data.

    Will not update if key exists in data.
    """
    if not g.get_global(global_key):
        g.set_global(data, global_key)


def add_default(
    g: Graph,
    *,
    node_data: Optional[Dict] = None,
    edge_data: Optional[Dict] = None,
    global_data: Optional[Dict] = None,
    global_key: str = None
):
    if node_data:
        add_default_node_data(g, node_data)
    if edge_data:
        add_default_edge_data(g, edge_data)
    if global_data:
        add_default_global_data(g, global_data, global_key)
