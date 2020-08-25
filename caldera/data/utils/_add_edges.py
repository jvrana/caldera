from typing import Union

import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils._utils import edges_difference


def add_edges(data: GraphData, fill_value: Union[float, int], kind: str) -> GraphData:
    """Adds edges to the :class:`caldera.data.GraphData`.

    :param data: :class:`caldera.data.GraphData` instance.
    :param fill_value: fill value for edge attribute tensor.
    :param kind: Choose from "self" (for self edges), "complete" (for complete graph) or "undirected" (undirected edges)
    :return:
    """
    UNDIRECTED = "undirected"
    COMPLETE = "complete"
    SELF = "self"
    VALID_KIND = [UNDIRECTED, COMPLETE, SELF]
    if kind not in VALID_KIND:
        raise ValueError("'kind' must be one of {}".format(VALID_KIND))

    data_cls = data.__class__
    if issubclass(data_cls, GraphBatch):
        node_idx = data.node_idx
        edge_idx = data.edge_idx
    elif issubclass(data_cls, GraphData):
        node_idx = torch.zeros(data.x.shape[0], dtype=torch.long)
        edge_idx = torch.zeros(data.e.shape[0], dtype=torch.long)
    else:
        raise ValueError(
            "data must be a subclass of {} or {}".format(
                GraphBatch.__class__.__name__, GraphData.__class__.__name__
            )
        )

    with torch.no_grad():
        # we count the number of nodes in each graph using node_idx
        gidx, n_nodes = torch.unique(node_idx, return_counts=True, sorted=True)
        _, n_edges = torch.unique(edge_idx, return_counts=True, sorted=True)

        eidx = 0
        nidx = 0
        graph_edges_list = []
        new_edges_lengths = torch.zeros(gidx.shape[0], dtype=torch.long)

        for _gidx, _n_nodes, _n_edges in zip(gidx, n_nodes, n_edges):
            graph_edges = data.edges[:, eidx : eidx + _n_edges]
            if kind == UNDIRECTED:
                missing_edges = edges_difference(graph_edges.flip(0), graph_edges)
            elif kind == COMPLETE:
                all_graph_edges = _create_all_edges(nidx, _n_nodes)
                missing_edges = edges_difference(all_graph_edges, graph_edges)
            elif kind == SELF:
                self_edges = torch.cat(
                    [torch.arange(nidx, nidx + _n_nodes).expand(1, -1)] * 2
                )
                missing_edges = edges_difference(self_edges, graph_edges)
            graph_edges_list.append(missing_edges)

            if not missing_edges.shape[0]:
                new_edges_lengths[_gidx] = 0
            else:
                new_edges_lengths[_gidx] = missing_edges.shape[1]

            nidx += _n_nodes
            eidx += _n_edges

        new_edges = torch.cat(graph_edges_list, axis=1)
        new_edge_idx = gidx.repeat_interleave(new_edges_lengths)
        new_edge_attr = torch.full(
            (new_edges.shape[1], data.e.shape[1]), fill_value=fill_value
        )

        edges = torch.cat([data.edges, new_edges], axis=1)
        edge_idx = torch.cat([edge_idx, new_edge_idx])
        edge_attr = torch.cat([data.e, new_edge_attr], axis=0)

        idx = edge_idx.argsort()

    if issubclass(data_cls, GraphBatch):
        return GraphBatch(
            node_attr=data.x.detach().clone(),
            edge_attr=edge_attr[idx],
            global_attr=data.g.detach().clone(),
            edges=edges[:, idx],
            node_idx=data.node_idx.detach().clone(),
            edge_idx=edge_idx[idx],
        )
    elif issubclass(data_cls, GraphData):
        return data_cls(
            node_attr=data.x.detach().clone(),
            edge_attr=edge_attr[idx],
            global_attr=data.g.detach().clone(),
            edges=edges[:, idx],
        )


def _create_all_edges(start: int, n_nodes: int) -> torch.LongTensor:
    """Create all edges starting from node index 'start' for 'n_nodes'
    additional nodes.

    E.g. _all_edges(5, 4) would create all edges of nodes 5, 6, 7, 8
    with shape (2, 4)
    """
    a = torch.arange(start, start + n_nodes, dtype=torch.long)
    b = a.repeat_interleave(torch.ones(n_nodes, dtype=torch.long) * n_nodes)
    c = torch.arange(start, start + n_nodes, dtype=torch.long).repeat(n_nodes)
    return torch.stack([b, c], axis=0)
