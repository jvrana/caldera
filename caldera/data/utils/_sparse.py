from typing import Optional
from typing import Union

import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.utils.sparse import scatter_coo


def to_sparse_coo_matrix(
    data: Union[GraphData, GraphBatch],
    fill_value: Optional[Union[int, float, torch.Tensor]] = None,
    dtype=torch.float,
):
    """Return the sparse coo representation of the edge attribute data of the.

    :class:`caldera.data.GraphData` instance. If provided with the optional
    fill value, will return a sparse matrix filled with the value instead of
    the edge attribute data.

    .. code-block:: python

        to_sparse_coo_matrix(data)
        to_sparse_coo_matrix(data, fill_value=1)

    :param data:
    :param fill_value:
    :param dtype:
    :return:
    """
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
