import random

import networkx as nx
import numpy as np
import pytest

from caldera.utils.nx.path import floyd_warshall
from caldera.utils.nx.path import PathMul
from caldera.utils.nx.path import PathSum
from caldera.utils.nx.path import PathSymbol


class TestSympyFloydWarshall:
    @pytest.mark.parametrize("key", ["weight", "cost"])
    @pytest.mark.parametrize(
        "complete_graph", [1, 100, 200, (100, nx.Graph)], indirect=True
    )
    def test_floyd_warshall(self, key, complete_graph):
        g, params, indirect_params = complete_graph
        nodelist = list(g.nodes())
        for n1, n2, edata in g.edges(data=True):
            for k in list(edata):
                del edata[k]
            edata[key] = random.randint(1, 10)
        expected = nx.floyd_warshall_numpy(g, nodelist=nodelist, weight=key)
        x = floyd_warshall(g=g, func=lambda x: x, symbols=(key,))
        assert np.allclose(x, expected)

    def test_floyd_warshall_complex(self, complete_graph):
        g, params, indirect_params = complete_graph
        for n1, n2, edata in g.edges(data=True):
            edata["weight"] = random.randint(1, 10)
            edata["efficiency"] = np.random.uniform(0.7, 0.9, 1)
        x = floyd_warshall(
            g=g,
            func=lambda w, e: w / e,
            symbols=[PathSymbol("weight", PathSum), PathSymbol("efficiency", PathMul)],
        )
        print(x)
