import numpy as np
import networkx as nx
from typing import Union, Callable, Any, Dict, Type, Tuple, List
from collections import OrderedDict
import inspect


class PathAccumulator(object):

    all = {}

    def __init__(self, name: str, func: Callable[[np.ndarray, np.ndarray], np.ndarray], i_fill: float):
        self.name = name
        self.func = func
        if self.name in self.all:
            raise ValueError("{} already defined".format(name))
        self.all[self.name] = self
        self.i_fill

    def __call__(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        return self.func(arr1, arr2)

    @classmethod
    def get(cls, name, default: str = None):
        if default is None:
            return cls.all[name]
        else:
            if name not in cls.all:
                name = default
            return cls.all.get(name)


Product = PathAccumulator("product", np.multiply, 1.0)
Sum = PathAccumulator("sum", lambda a, b: a + b, 0.0)
Min = PathAccumulator('min', np.minimum, 0.0)
Max = PathAccumulator('max', np.maximum, 0.0)


class PathCostSignature(object):

    def __init__(self, symbols: Union[List[str], Tuple[str, ...]],
                 function: Callable,
                 accumulator_map: Dict[str, Union[Callable, PathAccumulator]]):
        self.symbols = symbols
        self.function = function
        self.accumulator_map = accumulator_map
        self.validate_accumulator_map()
        self.validate_function()

    def validate_function(self):
        spec = inspect.getfullargspec(self.function)
        if len(spec.args) != len(self.symbols):
            raise ValueError("Function signature has {} args, but only {} symbols were provided".format(
                len(spec.args), len(self.symbols)
            ))

    def validate_accumulator_map(self):
        for k, v in self.accumulator_map.items():
            if k not in self.symbols:
                raise ValueError("Symbol '{}' is present in accumulator_map, but abset from list of symbols".format(
                    self.symbols
                ))
            if isinstance(v, str):
                if v not in PathAccumulator.all:
                    raise ValueError("Accumulator '{}' is not a valid name")
            elif not issubclass(v.__class__, PathAccumulator) and not callable(v):
                raise ValueError("Accumulator map values must be either a callable or Accumulator, not {}".format(
                    v.__class__
                ))


def replace_nan_with_inf(m):
    np.putmask(m, np.isnan(m), np.full_like(m, np.inf))
    return m


def init_matrix(g, nodelist, key, multigraph_weight_func, nonedge_fill, dtype)
    W = nx.to_numpy_matrix(
        g,
        nodelist=nodelist,
        multigraph_weight=multigraph_weight_func,
        weight=key,
        nonedge=nonedge_fill,
        dtype=dtype,
    )
    return W


def sympy_floyd_warshall(
    g: Union[nx.DiGraph, nx.Graph],
    f: Callable[[Tuple[Union[int, float], ...]], Union[int, float]],
    symbols: Union[Tuple[str, ...], List[str]],
    accumulator_map: dict,
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
    :param accumulator_map: diciontary of symbols to accumulator functions (choose from
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

    # initialize weight matrices
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
        accumulator_name = accumulator_map.get(key, Sum.name)
        accumulator = PathAccumulator.get(accumulator_name)
        if accumulator.name == 'sum':
            d = 0.0
        elif accumulator.name == 'product':
            d = 1.0
        elif accumulator.name in ['max', 'min']:
            d = 0.0
        else:
            raise ValueError("Accumulator '{}' not recognized".format(key))

        # set diagonal
        matrix[identity == 1] = identity_subs.get(key, d)

    for i in np.arange(n):
        # get costs if using node 'i'
        parts_dict = OrderedDict()
        for key, M in matrix_dict.items():
            M = matrix_dict[key]
            parts_dict[key] = PathAccumulator.get(key, 'sum')(M[i, :], M[:, i])

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
