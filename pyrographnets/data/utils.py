from typing import Optional

import networkx as nx
import numpy as np
import torch

from pyrographnets.data import GraphData, GraphBatch
from pyrographnets.utils import _first, scatter_group


def to_graph_data(g: nx.DiGraph,
                  n_node_feat: Optional[int] = None,
                  n_edge_feat: Optional[int] = None,
                  n_glob_feat: Optional[int] = None):
    if hasattr(g, 'data'):
        gdata = g.data
    else:
        gdata = {}

    if n_node_feat is None:
        _, ndata = _first(g.nodes(data=True))
        n_node_feat = ndata['features'].shape[0]

    if n_edge_feat is None:
        _, _, edata = _first(g.edges(data=True))
        n_edge_feat = edata['features'].shape[0]

    if n_glob_feat is None:
        n_glob_feat = gdata['features'].shape[0]

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    node_attr = np.empty((n_nodes, n_node_feat))
    edge_attr = np.empty((n_edges, n_edge_feat))
    glob_attr = np.empty((1, n_glob_feat))

    nodes = sorted(list(g.nodes(data=True)))
    ndict = {}
    for i, (n, ndata) in enumerate(nodes):
        node_attr[i] = ndata['features']
        ndict[n] = i

    edges = np.empty((2, n_edges))
    for i, (n1, n2, edata) in enumerate(g.edges(data=True)):
        edges[:, i] = np.array([ndict[n1], ndict[n2]])
        edge_attr[i] = edata['features']

    glob_attr[0] = g.data['features']

    return GraphData(torch.tensor(node_attr, dtype=torch.float),
                     torch.tensor(edge_attr, dtype=torch.float),
                     torch.tensor(glob_attr, dtype=torch.float),
                     torch.tensor(edges, dtype=torch.long))


def graph_batch_to_data_list(batch: GraphBatch):
    assert issubclass(type(batch), GraphBatch)
    gidx_n, node_attr = scatter_group(batch.x, batch.node_idx)
    gidx_e, edge_attr = scatter_group(batch.e, batch.edge_idx)
    gidx_edge, edges = scatter_group(batch.edges.T, batch.edge_idx)

    def to_dict(a, b):
        return dict(zip([x.item() for x in a], b))

    ndict = to_dict(gidx_n, node_attr)
    edict = to_dict(gidx_e, edge_attr)
    edgesdict = to_dict(gidx_edge, edges)
    datalist = []
    for k in ndict:
        _edges = edgesdict[k].T - edgesdict[k].min()

        data = GraphData(ndict[k], edict[k], batch.g[k], _edges)
        datalist.append(data)
    return datalist


def graph_data_to_nx(data: GraphData):
    g = nx.DiGraph()
    for n, ndata in enumerate(data.x):
        g.add_node(n, **{'features': ndata})
    for i, e in enumerate(data.edges.T):
        g.add_edge(e[0], e[1], **{'features': data.e[i]})
    g.data = {'features': data.g}
    return g


def random_data(n_features, e_features, g_features):
    n_nodes = torch.randint(1, 10, torch.Size([])).item()
    n_edges = torch.randint(1, 20, torch.Size([])).item()
    return GraphData(
        torch.randn(n_nodes, n_features),
        torch.randn(n_edges, e_features),
        torch.randn(1, g_features),
        torch.randint(0, n_nodes, torch.Size([2, n_edges]))
    )