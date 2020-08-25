import numpy as np
import networkx as nx
from typing import Union, Callable, Type, List, Hashable
from collections import OrderedDict
from caldera.utils.np import replace_nan_with_inf
from .path_utils import PathSymbol, PathSum


def floyd_warshall(
    g: Union[nx.DiGraph, nx.Graph],
    symbols: List[PathSymbol],
    func: Callable = None,
    nodelist: List[Hashable] = None,
    return_all: bool = False,
    dtype: Type = np.float64,
):
    """
    Usage:

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

    _symbols = symbols[:]
    symbols = []
    for s in _symbols:
        if isinstance(s, str):
            s = PathSymbol(s, PathSum)
        symbols.append(s)

    if func is None:
        func = lambda x: x

    if nodelist is None:
        nodelist = list(g.nodes())

    # initialize weight matrices
    matrix_dict = OrderedDict()
    for symbol in symbols:
        matrix_dict[symbol.name] = symbol.initialize_matrix(g, dtype, nodelist)

    if return_all:
        ori_matrix_dict = {k: v.copy() for k, v in matrix_dict.items()}

    n, m = list(matrix_dict.values())[0].shape

    for i in np.arange(n):
        # get costs if using node 'i'
        parts_dict = OrderedDict()
        for symbol in symbols:
            M = matrix_dict[symbol.name]
            parts_dict[symbol.name] = symbol.accumulator(M[i, :], M[:, i])

        # get total cost
        m_arr = [np.asarray(m) for m in matrix_dict.values()]
        p_arr = [np.asarray(m) for m in parts_dict.values()]
        C = replace_nan_with_inf(func(*m_arr))
        C_part = func(*p_arr)

        # update
        for key, M in matrix_dict.items():
            part = parts_dict[key]
            c = C > C_part

            if np.any(c):
                # assert M.shape == part.shape
                np.putmask(M, c, part)

    m_arr = [np.asarray(m) for m in matrix_dict.values()]
    C = replace_nan_with_inf(func(*m_arr))
    if return_all:
        return C, matrix_dict, ori_matrix_dict
    return C
