import functools
import itertools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import networkx as nx
import numpy as np
from networkx.classes.reportviews import EdgeView
from networkx.classes.reportviews import NodeView

from ._base import NetworkxTransformBase
from ._types import _G
from caldera.utils import dict_join
from caldera.utils import functional as fn
from caldera.utils.tensor import to_one_hot

GraphGen = Generator[_G, None, None]

_T = TypeVar("T")
_K = TypeVar("K")
_S = TypeVar("S")
_V = TypeVar("V")


def values_to_one_hot(
    values: Iterable[_T],
    classes: Union[List[_T], Tuple[_T, ...]],
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


# def _get_join_fn(join_func):
#     if join_func == "hstack":
#
#         def join_func(a, b):
#             return np.hstack([a, b])
#
#     elif join_func == "vstack":
#
#         def join_func(a, b):
#             return np.vstack([a, b])
#
#     return join_func


# def _get_processing_func(encoding):
#     if encoding is None:
#         processing_func = fn.compose(list, np.array)
#     elif encoding == "onehot":
#         processing_func = fn.compose(list, functools.partial(values_to_one_hot))
#     else:
#         processing_func = None
#     return processing_func


def _dispatch_nx_iterator(
    g: nx.DiGraph, x: str
) -> Union[NodeView, EdgeView, Generator[Tuple[Hashable, Any], None, None]]:
    if x == "node":
        return g.nodes(data=True)
    elif x == "edge":
        return g.edges(data=True)
    elif x == "global":
        return g.globals(data=True)
    else:
        raise ValueError("choose from {}".format(["node", "edge", "global"]))


def _update_left_inplace(
    data: Dict[_K, _T], new_data: Dict[_K, _S], join_fn: Callable[[_T, _S], _K]
) -> Dict[_K, Union[_T, _S, _V]]:
    """Updates the left dictionary, joining values with the provided
    function."""
    return dict_join(data, new_data, data, mode="right", join_fn=join_fn)


def _merge_update(data, key, to_key, join_fn, process_fn, default=...):
    """Update dictionary by applying join_fn and process_fn. For example, the
    processing function may be converting values to np.ndarray, while the
    join_fn might be to apply hstack.

    :param data:
    :param key:
    :param to_key:
    :param join_fn:
    :param process_fn:
    :param default:
    :return:
    """

    select_data = fn.index_each(-1)

    getter = fn.get_each(key)
    if default is not ...:
        getter = fn.get_each(key, default=default)

    select_process_and_send_to_key = fn.compose(
        fn.index_each(-1),
        getter,
        process_fn,
        fn.map_each(lambda x: {to_key: x}),
    )
    merge_and_join = functools.partial(_update_left_inplace, join_fn=join_fn)

    teed = itertools.tee(data)
    original_data = select_data(teed[0])
    processed_data = list(select_process_and_send_to_key(teed[1]))

    for d1, d2 in zip(original_data, processed_data):
        merge_and_join(d1, d2)


class NetworkxAttachNumpyFeatures(NetworkxTransformBase):
    def __init__(
        self,
        x: str,
        from_key: str,
        to_key: str,
        *,
        default: Any = ...,
        encoding: Optional[Union[str, Callable[[Iterable[_T]], Iterable[_T]]]] = None,
        join_func: Union[
            str, Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = "hstack",
        global_key: str = None,
        **processing_kwargs
    ):
        """Initialize transform that converts networkx features to a
        :class:`np.ndarray`

        :param x: 'edge', 'node', or 'global'
        :param from_key: dictionary key to find feature
        :param to_key: new key to attach feature
        :param default:
        :param encoding:
        :param join_func: select from 'hstack', 'vstack' or provide a new join function
        :param global_key:
        :param processing_kwargs:
        """

        if encoding is None:
            processing_func = fn.compose(list, np.array)
        elif encoding == "onehot":
            processing_func = fn.compose(
                list, functools.partial(values_to_one_hot, **processing_kwargs)
            )
        elif encoding == "bool":
            processing_func = fn.compose(
                fn.map_each(lambda x: np.array([int(bool(x))]))
            )
        elif callable(encoding):
            processing_func = encoding
        else:
            raise ValueError(
                "Encoding {} is not a valid encoding. Select from 'onehot', 'bool', a callable, or None".format(
                    encoding
                )
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

    def transform(self, g: _G) -> _G:
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
        """Initialize transform that converts encodes networkx features into
        one-hot encodings.

        :param x: 'edge', 'node', 'global'
        :param from_key: dictionary key to find feature
        :param to_key: new key to attach feature
        :param default:
        :param join_func:
        :param global_key:
        :param classes:
        :param num_classes:
        """
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


class NetworkxAttachNumpyBool(NetworkxAttachNumpyFeatures):
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
        """Initialize transform that converts encodes networkx into a boolean
        represented as :class:`np.ndarray` of size 1.

         E.g. `True` gets encoded as `np.ndarray([1.])`

        :param x: 'edge', 'node', 'global'
        :param from_key: dictionary key to find feature
        :param to_key: new key to attach feature
        :param default:
        :param join_func:
        :param global_key:
        :param classes:
        :param num_classes:
        """
        super().__init__(
            x,
            from_key,
            to_key,
            default=default,
            encoding="bool",
            join_func=join_func,
            global_key=global_key,
            classes=classes,
            num_classes=num_classes,
        )
