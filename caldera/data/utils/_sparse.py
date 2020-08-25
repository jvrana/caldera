from typing import Union

import torch

from caldera.data import GraphBatch
from caldera.data import GraphData


def graph_data_to_coo_matrix(
    data: Union[GraphData, GraphBatch],
    fill_value=1,
    tensor_type=torch.sparse.FloatTensor,
):
    ij = data.edges
    v = torch.full(data.edges[0].shape, fill_value=fill_value, dtype=torch.float)
    size = torch.Size([data.num_nodes] * 2)
    return tensor_type(ij, v, size)
