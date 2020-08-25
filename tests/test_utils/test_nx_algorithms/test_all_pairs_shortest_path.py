import random

import networkx as nx
import numpy as np
import pytest

from caldera.utils.nx_algorithms.floydwarshall import sympy_floyd_warshall


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
        x = sympy_floyd_warshall(
            g=g, f=lambda x: x, symbols=(key,), accumulator_map={key: "sum"}
        )
        assert np.allclose(x, expected)
