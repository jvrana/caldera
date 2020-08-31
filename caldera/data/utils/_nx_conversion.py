"""_nx_conversion.py.

Functional methods to convert networkx graphs to
:class:`caldera.data.GraphData` instances.
"""
import numpy as np

from caldera.utils import dict_join
from caldera.utils.functional import Functional as Fn
from caldera.utils.functional.curry import curry

from caldera.utils.tensor import to_one_hot

import functools

# TODO: expose this method


def counts_by_value(key):
    return Fn.compose(
        Fn.group_each_by_key(lambda x: x[key]),
        Fn.index_each(0),
        Fn.enumerate_each(),
        Fn.map_each(lambda x: (x[1], x[0])),
        dict
    )


def index_features(key):
    return Fn.compose(
        Fn.tee_pipe_yield(
            counts_by_value(key)
        ),
        Fn.star(
            lambda counts, gen: Fn.map_each(lambda x: counts[x[key]])(gen)
        ),
        list,
        np.array
    )


def features_to_one_hot(key, num_classes):
    return Fn.compose(
        index_features(key),
        list,
        np.array,
        functools.partial(to_one_hot, mx=num_classes)
    )
#
# def dict_list_to_one_hot(key, num_classes):
#     return to_one_hot(index_features(datalist), num_classes)

collect_node_data = Fn.compose(
    lambda g: g.nodes(data=True),
    Fn.index_each(1)
)


def dict_join_and_hstack(data, new_data):
    return dict_join(data, new_data, data, mode='right', join_fn=lambda a, b: np.hstack([a, b]))


def ndata_getter(g):
    return Fn.map_each(lambda n: g.nodes[n])


def edata_getter(g):
    return Fn.map_each(lambda e: g.nodes[e[0]][e[1]])


def one_hot_graph_features(getter, g, key, to_key, num_classes):
    return Fn.compose(
            getter(g),
            features_to_one_hot(key, num_classes),
            Fn.map_each(lambda x: {to_key: x})
    )


def update_graph_data(graph_data, new_data, joiner):
    return Fn.compose(
        Fn.zip_all(),
        Fn.map_each(joiner),
        list
    )([graph_data, new_data])


def zip_dict(d1, d2):
    return Fn.compose(
        Fn.tee_pipe_yield(
            Fn.yield_all(
                Fn.map_each(lambda x: d1[x]),
                Fn.map_each(lambda x: d2[x])
            ),
            Fn.zip_all()
        ),
        Fn.zip_all(),
        list
    )


def nx_collect_one_hot_nodes(g, key, to_key, num_classes):

    nodelist = list(g.nodes())
    ndata = list(Fn.map_each(g.nodes.__getitem__)(nodelist))
    new_ndata = list(one_hot_graph_features(ndata_getter, g, key, to_key, num_classes)(nodelist))
    result = update_graph_data(ndata, new_ndata, Fn.star(dict_join_and_hstack))
    return result

def nx_collect_one_hot_edges(g, key, to_key, num_classes):
    edges_to_one_hot = Fn.compose(
        Fn.map_each(lambda x: g[x[0]][x[1]]),
        features_to_one_hot(key, num_classes),
        Fn.map_each(lambda x: {to_key: x})
    )




# data = [{"features": True}, {"features": True}, {"features": False}]
# f = Fn.compose(
#         lambda g: g.nodes(data=True),
#         Fn.index_each(1),
#         features_to_one_hot(
# , num_classes)
# )
# print(f(data))
#
# def collate_to_one_hot(datalist, num_classes):
#     print(datalist)
#     print(list(fn.group_each_by_key(lambda x: x['features'])(datalist)))
#     d = counts_by_value(datalist)
#     print(d)
#     print()
#     print(datalist)
#     print(list(fn.group_each_by_key(lambda x: x['features'])(datalist)))
#     d = counts_by_value(datalist)
#     print(d)
#
#     to_long = Fn.compose(
#         fn.tee_pipe_yield(
#             counts_by_value
#         ),
#
#         # fn.asterisk(
#         #     lambda c, g: fn.map_each(lambda x: c[x['features']])(g)
#         # ),
#         list,
#         # np.array,
#         # functools.partial(to_one_hot, mx=num_classes)
#     )
#     return to_long(datalist)