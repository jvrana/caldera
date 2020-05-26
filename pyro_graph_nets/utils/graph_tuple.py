# GraphTuple
from collections import namedtuple
from typing import Callable, List, Tuple
import torch
import numpy as np
import networkx as nx


GraphTuple = namedtuple('GraphTuple', [
    'node_attr',    # node level attributes
    'edge_attr',    # edge level attributes
    'global_attr',  # global level attributes
    'edges',        # node-to-node connectivity
    'node_indices', # tensor where each element indicates the index of the graph the node_attr belongs to
    'edge_indices'  # tensor where each element indicates the index of the graph that the edge_attr and edges belong to.
])


def to_graph_tuple(graphs: List[nx.DiGraph], feature_key: str = 'features', global_attr_key: str = 'data') -> GraphTuple:
    """
    Convert a list og networkx graphs into a GraphTuple. 
    
    :param graphs: list of graphs 
    :param feature_key: key to find the node, edge, and global features
    :param global_attr_key: attribute on the NetworkX graph to find the global data (default: 'data')
    :return: GraphTuple, a namedtuple of ['node_attr', 'edge_attr', 'global_attr', 'edges', 'node_inices', 'edge_indices']
    """
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
        if not hasattr(graph, global_attr_key):
            global_attributes.append([1.])
        else:
            global_attributes.append(graph.data[feature_key])
        for node, ndata in sorted(graph.nodes(data=True)):
            node_attributes.append(ndata[feature_key])
            node_indices.append(index)
        for n1, n2, edata in graph.edges(data=True):
            senders.append(n1)
            receivers.append(n2)
            edge_attributes.append(edata[feature_key])
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


def collate_tuples(tuples: List[Tuple], func: Callable[[List[Tuple]], Tuple]):
    """
    Collate elements of many tuples using a function. All of the first elements
    for all the tuples will be collated using the function, then all of the second
    elements of all tuples, and so on.

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

    return t(
        *[f(x) for x in zip(*tuples)]
    )


def replace_key(graph_tuple, data: dict):
    """Replace the values of the graph tuple. DOES NOT REPLACE IN PLACE."""
    values = []
    for k, v in zip(graph_tuple._fields, graph_tuple):
        if k in data:
            v = data[k]
        values.append(v)
    return GraphTuple(*values)


def apply_to_tuple(x, func: Callable[[List[Tuple]], Tuple]):
    """Apply function to each element of the tuple"""
    return type(x)(
        *[func(x) for x in x]
    )


def print_graph_tuple_shape(graph_tuple):
    for field, x in zip(graph_tuple._fields, graph_tuple):
        print(field, '  ', x.shape)