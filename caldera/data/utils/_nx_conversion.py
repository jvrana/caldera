"""_nx_conversion.py.

Functional methods to convert networkx graphs to
:class:`caldera.data.GraphData` instances.
"""
import numpy as np
from caldera.utils import dict_join
from caldera.utils.functional import Functional as Fn
from caldera.utils.tensor import to_one_hot
import functools
from typing import Generator, Dict


def _enumerate_unique_values(key):
    def _wrapped_counts_by_value(datalist: Generator[Dict, None, None]):
        return Fn.compose(
            Fn.get_each(key),
            Fn.iter_each_unique(),
            lambda x: sorted(x),
            Fn.enumerate_each(reverse=True),
            dict
        )(datalist)
    return _wrapped_counts_by_value


# TODO: this needs to be deterministic or sorted
def _index_features(key):
    def _wrapped_index_features(datalist: Generator[Dict, None, None]):
        return Fn.compose(
            Fn.tee_pipe_yield(
                _enumerate_unique_values(key)
            ),
            Fn.star(
                lambda counts, gen: Fn.map_each(lambda x: counts[x[key]])(gen)
            )
        )(datalist)
    return _wrapped_index_features


def features_to_one_hot(key, num_classes):
    def _features_to_one_hot(datalist: Generator[Dict, None, None]):
        return Fn.compose(
            _index_features(key),
            list,
            np.array,
            functools.partial(to_one_hot, mx=num_classes)
        )(datalist)
    return _features_to_one_hot


def _one_hot_graph_features(getter, key, to_key, num_classes):
    return Fn.compose(
        Fn.tee_pipe_yield(
            Fn.map_each(getter),
            features_to_one_hot(key, num_classes),
            Fn.map_each(lambda x: {to_key: x})
        ),
        Fn.zip_all(reverse=True),
        dict
    )


def _dict_join_and_hstack(data, new_data):
    """Joins two dictionaries. If keys are shared, `np.hstack` is applied."""
    return dict_join(data, new_data, data, mode='right', join_fn=lambda a, b: np.hstack([a, b]))


def _update_dict(dict_to_update, new_data):
    dict_join(dict_to_update, new_data, join_fn=_dict_join_and_hstack, mode='union')


def _nx_collect_one_hot(getter, _list, key, to_key, num_classes):
    data1 = {k: getter(k) for k in _list}
    data2 = _one_hot_graph_features(getter, key, to_key, num_classes)(_list)
    _update_dict(data1, data2)


def nx_collect_one_hot_nodes(g, key, to_key, num_classes):
    return _nx_collect_one_hot(lambda k: g.nodes[k], list(g.nodes()), key, to_key, num_classes)


def nx_collect_one_hot_edges(g, key, to_key, num_classes):
    return _nx_collect_one_hot(lambda k: g.edges[k], list(g.edges()), key, to_key, num_classes)


def _setdefault(d1, d2):
    return dict_join(d1, d2, d1, join_fn=lambda a, b: a, mode='right')



def add_default_node_data(g, data):
    for _, _, edata in g.edges(data=True):
        _setdefault(edata, data)


def add_default_edge_data(g, data):
    for _, _, edata in g.edges(data=True):
        _setdefault(edata, data)