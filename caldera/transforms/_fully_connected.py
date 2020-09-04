"""_fully_connected.py.

Make graph fully connected
"""
from typing import overload

from ._base import TransformBase
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils import add_edges


class FullyConnected(TransformBase):
    def __init__(self, fill_value: float = 0.0):
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

    @overload
    def transform_each(self, data: GraphData) -> GraphData:
        ...

    def transform(self, data: GraphBatch) -> GraphBatch:
        return add_edges(data, self.fill_value, kind="complete")
