import functools
import itertools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import networkx as nx
import numpy as np

from ._base import NetworkxTransformBase
from caldera.utils import dict_join
from caldera.utils import functional
from caldera.utils.tensor import to_one_hot


GraphGen = Generator[nx.Graph, None, None]

T = TypeVar("T")
K = TypeVar("K")
S = TypeVar("S")
V = TypeVar("V")


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


def _get_join_fn(join_func):
    if join_func == "hstack":

        def join_func(a, b):
            return np.hstack([a, b])

    elif join_func == "vstack":

        def join_func(a, b):
            return np.vstack([a, b])

    return join_func


def _get_processing_func(encoding):
    if encoding is None:
        processing_func = functional.compose(list, np.array)
    elif encoding == "onehot":
        processing_func = functional.compose(list, functools.partial(values_to_one_hot))
    else:
        processing_func = None
    return processing_func


def _dispatch_nx_iterator(g, x):
    if x == "node":
        return g.nodes(data=True)
    elif x == "edge":
        return g.edges(data=True)
    elif x == "global":
        return g.globals(data=True)
    else:
        raise ValueError("choose from {}".format(["node", "edge", "global"]))


def _update_left_inplace(
    data: Dict[K, T], new_data: Dict[K, S], join_fn: Callable[[T, S], K]
) -> Dict[K, Union[T, S, V]]:
    """Updates the left dictionary, joining values with the provided
    function."""
    return dict_join(data, new_data, data, mode="right", join_fn=join_fn)


def _merge_update(data, key, to_key, join_fn, process_fn, default=...):
    gen1, gen2 = itertools.tee(data)
    update_fn = functools.partial(_update_left_inplace, join_fn=join_fn)

    data1 = functional.compose(functional.index_each(-1))(gen1)

    getter = functional.get_each(key)
    if default is not ...:
        getter = functional.get_each(key, default=default)

    data2 = functional.compose(
        functional.index_each(-1),
        getter,
        process_fn,
        functional.map_each(lambda x: {to_key: x}),
    )(gen2)
    for d1, d2 in zip(data1, data2):
        _ = update_fn(d1, d2)


class NetworkxAttachNumpyFeatures(NetworkxTransformBase):
    def __init__(
        self,
        x: str,
        from_key: str,
        to_key: str,
        *,
        default: Any = ...,
        encoding: Optional[str] = None,
        join_func: str = "hstack",
        global_key: str = None,
        **processing_kwargs
    ):

        if encoding is None:
            processing_func = functional.compose(list, np.array)
        elif encoding == "onehot":
            processing_func = functional.compose(
                list, functools.partial(values_to_one_hot, **processing_kwargs)
            )

        if join_func == "hstack":

            def join_func(a, b):
                return np.hstack([a, b])

        elif join_func == "vstack":

            def join_func(a, b):
                return np.vstack([a, b])

        self.x = x
        self.from_key = from_key
        self.to_key = to_key
        self.global_key = global_key
        self.default = default
        self.join_fn = join_func
        self.processing_func = processing_func
        self.processing_kwargs = processing_kwargs

    def transform(self, g):
        iterator = list(_dispatch_nx_iterator(g, self.x))

        _merge_update(
            iterator,
            self.from_key,
            self.to_key,
            join_fn=self.join_fn,
            process_fn=self.processing_func,
            default=self.default,
        )
        return g


class NetworkxAttachNumpyOneHot(NetworkxAttachNumpyFeatures):
    def __init__(
        self,
        x: str,
        from_key: str,
        to_key: str,
        *,
        default: Any = ...,
        join_func: str = "hstack",
        global_key: str = None,
        classes: List[str] = None,
        num_classes: int = None
    ):
        super().__init__(
            x,
            from_key,
            to_key,
            default=default,
            encoding="onehot",
            join_func=join_func,
            global_key=global_key,
            classes=classes,
            num_classes=num_classes,
        )
