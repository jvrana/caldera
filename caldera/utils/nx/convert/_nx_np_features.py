"""_nx_np_features.py.

Functional methods to convert networkx graphs to
:class:`caldera.data.GraphData` instances.
"""
import functools
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import networkx as nx
import numpy as np

from caldera.utils import dict_join
from caldera.utils.functional import Functional as Fn
from caldera.utils.nx.types import Graph
from caldera.utils.tensor import to_one_hot

T = TypeVar("T")
K = TypeVar("K")
S = TypeVar("S")
V = TypeVar("V")
GLOBAL = "data"


def _raise_if_not_graph(x):
    if not issubclass(x.__class__, nx.Graph):
        raise TypeError(
            "`graphs` must be an iterable of type `nx.Graph`. Found {}".format(
                x.__class__
            )
        )


def get_global_data(graphs, global_key: str = GLOBAL):
    return Fn.compose(
        Fn.apply_each(_raise_if_not_graph),
        Fn.getattr_each(global_key),
        Fn.enumerate_each(),
        list,
    )(graphs)


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
    v = [d[_v] for _v in values]
    if num_classes is None:
        num_classes = len(d)
    return to_one_hot(np.array(v), mx=num_classes)


def merge_update(data, key, to_key, join_fn, process_fn):
    update_fn = functools.partial(update_left_inplace, join_fn=join_fn)

    data1 = Fn.compose(Fn.index_each(-1))(data)

    data2 = Fn.compose(
        Fn.index_each(-1),
        Fn.get_each(key),
        process_fn,
        Fn.map_each(lambda x: {to_key: x}),
    )(data)

    for d1, d2 in zip(data1, data2):
        _ = update_fn(d1, d2)


def nx_collect_features(
    g: Graph,
    feature: str,
    from_key: str,
    to_key: str,
    encoding: Optional[str] = None,
    processing_func=None,
    processing_kwargs=None,
    join_func: str = "hstack",
    global_key: str = GLOBAL,
    **kwargs
):
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

    if feature == "node":
        data = g.nodes(data=True)
    elif feature == "edge":
        data = g.edges(data=True)
    elif feature == "global":
        data = get_global_data([g], global_key)
    else:
        raise ValueError("feature '{}' not recognized.".format(feature))

    merge_update(data, from_key, to_key, join_fn=join_func, process_fn=processing_func)
