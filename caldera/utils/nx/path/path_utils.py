from typing import Callable
import numpy as np
import networkx as nx


class PathAccumulator(object):

    all = {}

    def __init__(
        self,
        name: str,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        i_fill: float,
    ):
        self.name = name
        self.func = func
        if self.name in self.all:
            raise ValueError("{} already defined".format(name))
        self.all[self.name] = self
        self.i_fill = i_fill

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.func(*args, **kwargs)

    @classmethod
    def get(cls, name, default: str = None):
        if default is None:
            return cls.all[name]
        else:
            if name not in cls.all:
                name = default
            return cls.all.get(name)


PathMul = PathAccumulator("product", np.multiply, 1.0)
PathSum = PathAccumulator("sum", lambda a, b: a + b, 0.0)
PathMin = PathAccumulator("min", np.minimum, 0.0)
PathMax = PathAccumulator("max", np.maximum, 0.0)
PathNpSum = PathAccumulator("npsum", np.sum, 0.0)
PathNpProduct = PathAccumulator("npprod", np.product, 1.0)


class PathSymbol(object):
    def __init__(
        self,
        name: str,
        accumulator: Callable = PathSum,
        nonedge_fill=np.inf,
        multigraph_weight: Callable = sum,
    ):
        self.name = name
        self.iden_fill = accumulator.i_fill
        self.multigraph_weight = multigraph_weight
        self.nonedge_fill = nonedge_fill
        self.accumulator = accumulator

    def initialize_matrix(self, g, dtype, nodelist=None):
        if nodelist is None:
            nodelist = list(g.nodes)

        W = nx.to_numpy_matrix(
            g,
            nodelist=nodelist,
            multigraph_weight=self.multigraph_weight,
            weight=self.name,
            nonedge=self.nonedge_fill,
            dtype=dtype,
        )

        n, m = W.shape

        # replace diagonals
        identity = np.identity(n)

        # set accumulators
        W[identity == 1] = self.accumulator.i_fill
        return W
