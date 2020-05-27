# GraphTuple
from collections import namedtuple
from functools import partial
from typing import Callable
from typing import List
from typing import Tuple

import networkx as nx
import numpy as np
import torch

GraphTuple = namedtuple(
    "GraphTuple",
    [
        "node_attr",  # node level attributes
        "edge_attr",  # edge level attributes
        "global_attr",  # global level attributes
        "edges",  # node-to-node connectivity
        "node_indices",  # tensor where each element indicates the index of the graph the node_attr belongs to
        "edge_indices",
        # tensor where each element indicates the index of the graph that the edge_attr and edges belong to.
    ],
)


def pick_edge(g):
    for x in g.edges(data=True):
        return x


def pick_node(g):
    for n in g.nodes(data=True):
        return n


def to_graph_tuple(
    graphs: List[nx.DiGraph],
    feature_key: str = "features",
    global_attr_key: str = "data",
    device: str = None,
) -> GraphTuple:
    """Convert a list og networkx graphs into a GraphTuple.

    :param graphs: list of graphs
    :param feature_key: key to find the node, edge, and global features
    :param global_attr_key: attribute on the NetworkX graph to find the global data (default: 'data')
    :return: GraphTuple, a namedtuple of ['node_attr', 'edge_attr', 'global_attr',
        'edges', 'node_inices', 'edge_indices']
    """
    n_edges = 0
    n_nodes = 0
    for graph in graphs:
        n_edges += graph.number_of_edges()
        n_nodes += graph.number_of_nodes()

    n = len(graphs)
    node_idx = np.empty(n_nodes)
    edge_idx = np.empty(n_edges)

    edata = pick_edge(graph)[-1][feature_key]
    vdata = pick_node(graph)[-1][feature_key]
    udata = getattr(graph, global_attr_key)[feature_key]
    connectivity = np.empty((n_edges, 2))

    v = np.empty((n_nodes, *tuple(vdata.shape)))
    e = np.empty((n_edges, *tuple(edata.shape)))
    u = np.empty((n, *tuple(udata.shape)))

    _v = 0
    _e = 0

    ndict = {}

    for gidx, graph in enumerate(graphs):
        for node, ndata in graph.nodes(data=True):
            v[_v] = ndata[feature_key]
            ndict[node] = _v
            node_idx[_v] = gidx
            _v += 1

        for n1, n2, edata in graph.edges(data=True):
            e[_e] = edata[feature_key]
            edge_idx[_e] = gidx
            connectivity[_e] = [ndict[n1], ndict[n2]]
            _e += 1

        u[gidx] = getattr(graph, global_attr_key)[feature_key]

    result = GraphTuple(
        torch.tensor(v, dtype=torch.float),
        torch.tensor(e, dtype=torch.float),
        torch.tensor(u, dtype=torch.float),
        torch.tensor(connectivity, dtype=torch.long),
        torch.tensor(node_idx, dtype=torch.long),
        torch.tensor(edge_idx, dtype=torch.long),
    )
    if device:
        return GraphTuple(*[x.to(device) for x in result])
    return result


# def batch(a, batch_size):
#     """Batches the tensor according to the batch size.
#
#     .. code-block::
#
#         batch(torch.rand(20, 8, 10), batch_size=5).shape
#         # >>> torch.Size([5, 4, 8, 10])
#     """
#     b = torch.unsqueeze(a, 1).reshape(batch_size, -1, *a.shape[1:])
#     return b


def collate_tuples(tuples: List[Tuple], func: Callable[[List[Tuple]], Tuple]):
    """Collate elements of many tuples using a function. All of the first
    elements for all the tuples will be collated using the function, then all
    of the second elements of all tuples, and so on.

    :param tuples: list of tuples
    :param func: callable
    :return: tuple of same type
    """

    def f(x):
        if x is None:
            return None
        else:
            return func(x)

    t = type(tuples[0])
    if t is tuple:

        def t(*args):
            return tuple(args)

    return t(*[f(x) for x in zip(*tuples)])


def replace_key(graph_tuple, data: dict):
    """Replace the values of the graph tuple.

    DOES NOT REPLACE IN PLACE.
    """
    values = []
    for k, v in zip(graph_tuple._fields, graph_tuple):
        if k in data:
            v = data[k]
        values.append(v)
    return GraphTuple(*values)


def apply_to_tuple(x, func: Callable[[List[Tuple]], Tuple]):
    """Apply function to each element of the tuple."""
    return type(x)(*[func(x) for x in x])


def print_graph_tuple_shape(graph_tuple):
    for field, x in zip(graph_tuple._fields, graph_tuple):
        print(field, "  ", x.shape)


def cat_gt(*gts: Tuple[GraphTuple, ...]) -> GraphTuple:
    """Concatenate graph tuples along dimension=1.

    Edges, node idx and edge idx are simply copied over.
    """
    cat = partial(torch.cat, dim=1)
    return GraphTuple(
        cat([gt.node_attr for gt in gts]),
        cat([gt.edge_attr for gt in gts]),
        cat([gt.global_attr for gt in gts]),
        gts[0].edges,
        gts[0].node_indices,
        gts[0].edge_indices,
    )


def gt_to_device(x: Tuple, device):
    return GraphTuple(*[v.to(device) for v in x])


class InvalidGraphTuple(Exception):
    pass


def validate_gt(gt: GraphTuple):
    if not isinstance(gt, GraphTuple):
        raise InvalidGraphTuple("{} is not a {}".format(gt, GraphTuple))

    if not gt.edge_attr.shape[0] == gt.edges.shape[0]:
        raise InvalidGraphTuple(
            "Edge attribute shape {} does not match edges shape {}".format(
                gt.edge_attr.shape, gt.edges.shape
            )
        )

    if not gt.edge_attr.shape[0] == gt.edge_indices.shape[0]:
        raise InvalidGraphTuple(
            "Edge attribute shape {} does not match edge idx shape {}".format(
                gt.edge_attr.shape, gt.edge_indices.shape
            )
        )

    if not gt.node_attr.shape[0] == gt.node_indices.shape[0]:
        raise InvalidGraphTuple(
            "Node attribute shape {} does not match node idx shape {}".format(
                gt.node_attr.shape, gt.node_indices.shape
            )
        )

    # edges cannot refer to non-existent nodes
    if not gt.edges.max() < gt.node_attr.shape[0]:
        raise InvalidGraphTuple(
            "Edges reference node {} which does not exist nodes of size {}".format(
                gt.edges.max(), gt.node_attr.shape[0]
            )
        )

    if not gt.edges.min() >= 0:
        raise InvalidGraphTuple(
            "Node index must be greater than 0, not {}".format(gt.edges.min())
        )
