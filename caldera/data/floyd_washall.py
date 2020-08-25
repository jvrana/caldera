from caldera.data import GraphData, GraphBatch
from typing import Union
from caldera.data.utils import graph_matrix
import torch


def torch_floyd_warshall(data: Union[GraphData, GraphBatch],):
    """
    Run the floyd-warshall algorithm (all pairs shortest path) with arbitrary
    cost functions.

    .. code-block:: python

        W = floyd_warshall2(g, symbols=[
                PathSymbol("A", SumPath),
                PathSymbol("B", MulPath)
            ], func: lambda a, b: a / b
        )

    .. code-block:: python

        W = floyd_warshall2(g, key="weight")

    :param g:
    :param symbols:
    :param func:
    :param nodelist:
    :param return_all:
    :param dtype:
    :return:
    """

    A = graph_matrix(data, dtype=torch.float, fill_value=torch.inf, edge_value=1)

    n, m = A.shape

    I = torch.eye(n)
    A[I == 1] = 0  # diagonal elements should be zero
    for i in range(n):
        B = A[0, :].expand(1, -1) + A[:, 0].expand(1, -1).T
        idx = torch.where(B < A)
        A[idx] = B[idx]
    return A
