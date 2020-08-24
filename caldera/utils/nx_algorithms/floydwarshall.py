import numpy as np
import networkx as nx
from typing import Union, Callable, Any, Dict, Type, Tuple, List
from collections import OrderedDict


def sympy_multisource_dijkstras(
    g, sources, f, accumulators=None, init=None, target=None, cutoff=None
):
    if sources is None or not len(sources):
        raise ValueError("sources must not be empty")
    if target in sources:
        return (0, [target])
    paths = {source: [source] for source in sources}  # dictionary of paths
    dist = _multisource_dijkstra(
        g,
        sources,
        f,
        target=target,
        accumulators=accumulators,
        init=init,
        paths=paths,
        cutoff=cutoff,
    )
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError:
        raise nx.NetworkXNoPath("No path to {}.".format(target))


def sympy_dijkstras(
    g, source, f, target=None, accumulators=None, init=None, cutoff=None
):
    """Computes the shortest path distance and path for a graph using an
    arbitrary function.

    :param g:
    :param source:
    :param f:
    :param target:
    :param accumulators:
    :param init:
    :param cutoff:
    :return:
    """
    dist, path = sympy_multisource_dijkstras(
        g,
        [source],
        f,
        target=target,
        accumulators=accumulators,
        init=init,
        cutoff=cutoff,
    )
    return dist, path


PRODUCT = "product"
SUM = "sum"
MAX = "max"
MIN = "min"


# TODO: implement MIN and MAX
def accumulate_helper(key, m1, m2):
    if key == SUM:
        return m1 + m2
    elif key == PRODUCT:
        return np.multiply(m1, m2)
    else:
        raise ValueError(
            "Key '{}' not in accumulator dictionary. Options are '{}' or '{}'".format(
                key, PRODUCT, SUM
            )
        )


def replace_nan_with_inf(m):
    np.putmask(m, np.isnan(m), np.full_like(m, np.inf))
    return m


def sympy_floyd_warshall(
    g: Union[nx.DiGraph, nx.Graph],
    f: Callable[[Tuple[Union[int, float], ...]], Union[int, float]],
    symbols: Union[Tuple[str, ...], List[str]],
    accumulators: dict,
    nonedge: dict = None,
    nodelist: list = None,
    multigraph_weight: Callable = None,
    identity_subs: Dict[str, Any] = None,
    return_all: bool = False,
    dtype: Type = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.matrix], Dict[str, np.matrix]]]:
    """Implementation of algorithm:

    .. math::

       C = \\frac{\\sum_{i}^{n}{a_i}}{\\prod_{i}^{n}{b_i}}

    Where :math:`a_i` and :math:`b_i` is the weight 'a' and 'b' of the *ith* edge in the path
    respectively.
    :math:`\\sum_{i}^{n}{a_i}` is the accumulated sum of weights 'a' and
    :math:`\\prod_{i}^{n}{b_i}` is the accumulated product of weights 'b'.
    Arbitrarily complex path functions with arbitrary numbers of weights
    (:math:`a, b, c,...`) can be used in the algorithm.

    Because arbitrary functions are used, the shortest path between ij and jk does not
    necessarily mean the shortest path nodes ijk is the concatenation of these two
    paths. In other words, for the shortest path :math:`p_{ik}` between
    nodes :math:`i` :math:`j` and :math:`k`:

    .. math::

        p_{ij} + p_{jk} \\neq p_{ijk}

    :param g: the graph
    :param f: the function string that represents SymPy function to compute the weights
    :param accumulators: diciontary of symbols to accumulator functions (choose from
                         ["PRODUCT" - :math:`\\prod`,
                         "SUM" - :math:`\\sum`]
                         to use for accumulation of weights through
                         a path. If missing "SUM" is used.
    :param nonedge: dictionary of symbol to value to use for nonedges
                    (e.g. {'weight': np.inf})
    :param nodelist: optional nodelist to use
    :param multigraph_weight: optional (default: min) function to use for multigraphs
    :param identity_subs: the dictionary of values to set along the diagonal axis
    :param return_all: if True, return both the resulting weight matrix and the
            individual components broken down by symbol strings.
    :param dtype: the dtype of the resulting np.ndarrays used and returned
    :return: either just the weight matrix or, if return_all is True, the weight_matrix
                and the dictionary of the weight components.
    """
    if dtype is None:
        dtype = np.float64

    if identity_subs is None:
        identity_subs = {}

    if multigraph_weight is None:
        multigraph_weight = {}

    if nonedge is None:
        nonedge = {}

    func = f

    matrix_dict = OrderedDict()

    for name in symbols:
        matrix_dict[name] = nx.to_numpy_matrix(
            g,
            nodelist=nodelist,
            multigraph_weight=multigraph_weight.get(name, min),
            weight=name,
            nonedge=nonedge.get(name, np.inf),
            dtype=dtype,
        )

    if return_all:
        ori_matrix_dict = {k: v.copy() for k, v in matrix_dict.items()}

    n, m = list(matrix_dict.values())[0].shape

    # replace diagonals
    identity = np.identity(n)
    for key, matrix in matrix_dict.items():
        # set accumulators
        if accumulators.get(key, SUM) == SUM:
            d = 0.0
        elif accumulators[key] == PRODUCT:
            d = 1.0
        elif accumulators[key] in [MAX, MIN]:
            d = 0.0
        else:
            raise ValueError(
                "Accumulator key {} must either be '{}' or '{}' or a callable with two "
                "arguments ('M' a numpy matrix and 'i' a node index as an int)".format(
                    key, SUM, PRODUCT
                )
            )

        # set diagonal
        matrix[identity == 1] = identity_subs.get(key, d)

    for i in np.arange(n):
        # get costs if using node 'i'
        parts_dict = OrderedDict()
        for key, M in matrix_dict.items():
            M = matrix_dict[key]
            parts_dict[key] = accumulate_helper(
                accumulators.get(key, SUM), M[i, :], M[:, i]
            )

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
