# GraphTuple
from collections import namedtuple
from typing import Callable, List, Tuple
import torch
import numpy as np



GraphTuple = namedtuple('GraphTuple', [
    'node_attr',
    'edge_attr',
    'global_attr',
    'edges',
    'node_indices',
    'edge_indices'
])


def to_graph_tuple(graphs):
    senders = []
    receivers = []
    edge_attributes = []
    node_attributes = []
    global_attributes = []
    n_nodes = []
    n_edges = []
    node_indices = []
    edge_indices = []

    for index, graph in enumerate(graphs):
        n_nodes.append(graph.number_of_nodes())
        n_edges.append(graph.number_of_edges())
        if not hasattr(graph, 'global'):
            global_attributes.append([1.])
        for node, ndata in sorted(graph.nodes(data=True)):
            node_attributes.append(ndata['features'])
            node_indices.append(index)
        for n1, n2, edata in graph.edges(data=True):
            senders.append(n1)
            receivers.append(n2)
            edge_attributes.append(edata['features'])
            edge_indices.append(index)

    node_attr = torch.tensor(np.vstack(node_attributes), dtype=torch.float)
    edge_attr = torch.tensor(np.vstack(edge_attributes), dtype=torch.float)
    edges = torch.tensor(np.vstack([senders, receivers]).T, dtype=torch.long)
    global_attr = torch.tensor(global_attributes, dtype=torch.float).detach()
    node_indices = torch.tensor(node_indices, dtype=torch.long).detach()
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).detach()
    return GraphTuple(node_attr, edge_attr, global_attr, edges, node_indices,
                      edge_indices)


def batch(a, batch_size):
    """Batches the tensor according to the batch size.

    .. code-block::

        batch(torch.rand(20, 8, 10), batch_size=5).shape
        # >>> torch.Size([5, 4, 8, 10])
    """
    b = torch.unsqueeze(a, 1).reshape(batch_size, -1, *a.shape[1:])
    return b


def combine_tuples(tuples: List[Tuple], func: Callable[[List[Tuple]], Tuple]):
    def f(x):
        if x is None:
            return None
        else:
            return func(x)

    t = type(tuples[0])
    if t is tuple:
        def t(*args):
            return tuple(args)

    return t(
        *[f(x) for x in zip(*tuples)]
    )


def replace_key(graph_tuple, data: dict):
    values = []
    for k, v in zip(graph_tuple._fields, graph_tuple):
        if k in data:
            v = data[k]
        values.append(v)
    return GraphTuple(*values)


def apply_to_tuple(x, func: Callable[[List[Tuple]], Tuple]):
    return type(x)(
        *[func(x) for x in x]
    )