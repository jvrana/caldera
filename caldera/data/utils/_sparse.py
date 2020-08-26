from typing import Optional
from typing import Union

import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils.sparse import scatter_coo


def to_coo_matrix(
    data: Union[GraphData, GraphBatch],
    fill_value: Optional[Union[int, float, torch.Tensor]] = None,
    dtype=torch.float,
):
    if fill_value is not None:
        return scatter_coo(
            data.edges,
            fill_value,
            expand=True,
            dtype=dtype,
            size=(data.num_nodes, data.num_nodes, ...),
        )
    else:
        return scatter_coo(
            data.edges,
            data.e,
            expand=False,
            dtype=dtype,
            size=(data.num_nodes, data.num_nodes, ...),
        )
