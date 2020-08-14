"""
fully_connected.py

Make graph fully connected
"""

from .base import TransformBase
from caldera.data import GraphData, GraphBatch
from typing import overload
import torch
from caldera.data.utils import edges_difference, _create_all_edges


class FullyConnected(TransformBase):

    def __init__(self, fill_value: float = 0.):
        super().__init__()
        self.fill_value = fill_value


    # def __call__(self, data: GraphData) -> GraphData:
    #     if issubclass(data.__class__, GraphBatch):
    #         return data.__class__.from_data_list([self(d) for d in data.to_data_list()])
    #     with torch.no_grad():
    #         all_graph_edges = _create_all_edges(0, data.num_nodes)
    #         missing_edges = edges_difference(all_graph_edges, data.edges)
    #         edges = torch.cat([data.edges, missing_edges], axis=1)
    #         edge_attr = torch.full((edges.shape[1], data.e.shape[1]), fill_value=self.fill_value)
    #     return GraphData(
    #         node_attr=data.x.detach().clone(),
    #         edge_attr=edge_attr,
    #         global_attr=data.g.detach().clone(),
    #         edges=edges
    #     )

        #             graph_edges_list.append(missing_edges)
    def __call__(self, data: GraphBatch) -> GraphBatch:

        data_cls = data.__class__
        if issubclass(data_cls, GraphBatch):
            node_idx = data.node_idx
            edge_idx = data.edge_idx
        elif issubclass(data_cls, GraphData):
            node_idx = torch.zeros(data.x.shape[0], dtype=torch.long)
            edge_idx = torch.zeros(data.e.shape[1], dtype=torch.long)
        else:
            raise ValueError("data must be a subclass of {} or {}".format(
                GraphBatch.__class__.__name__, GraphData.__class__.__name__))

        with torch.no_grad():
            # we count the number of nodes in each graph using node_idx
            gidx, n_nodes = torch.unique(node_idx, return_counts=True, sorted=True)
            _, n_edges = torch.unique(edge_idx, return_counts=True, sorted=True)

            eidx = 0
            nidx = 0
            graph_edges_list = []
            new_edges_lengths = torch.zeros(gidx.shape[0], dtype=torch.long)
            for _gidx, _n_nodes, _n_edges in zip(gidx, n_nodes, n_edges):
                graph_edges = data.edges[:, eidx:eidx+_n_edges]
                all_graph_edges = _create_all_edges(nidx, _n_nodes)
                missing_edges = edges_difference(all_graph_edges, graph_edges)
                graph_edges_list.append(missing_edges)
                if not missing_edges.shape[0]:
                    new_edges_lengths[_gidx] = 0
                else:
                    new_edges_lengths[_gidx] = missing_edges.shape[1]

                nidx += _n_nodes
                eidx += _n_edges

            new_edges = torch.cat(graph_edges_list, axis=1)
            new_edge_idx = gidx.repeat_interleave(new_edges_lengths)
            new_edge_attr = torch.full((new_edges.shape[1], data.e.shape[1]), fill_value=self.fill_value)

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
                edge_idx=edge_idx[idx]
            )
        elif issubclass(data_cls, GraphData):
            return data_cls(
                node_attr=data.x.detach().clone(),
                edge_attr=edge_attr[idx],
                global_attr=data.g.detach().clone(),
                edges=edges[:, idx],
            )