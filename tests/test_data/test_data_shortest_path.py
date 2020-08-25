from caldera.data.utils import adj_matrix
from caldera.data import GraphData, GraphBatch
import torch


def test_():
    data = GraphData.random(5, 4, 3, min_nodes=1000, min_edges=1000)
    M = adj_matrix(data)
    m2 = torch.zeros_like(M)
    m2[M > 0] = 1


# W = nx.to_numpy_matrix(
#     g,
#     nodelist=nodelist,
#     multigraph_weight=self.multigraph_weight,
#     weight=self.name,
#     nonedge=self.nonedge_fill,
#     dtype=dtype,
# )
