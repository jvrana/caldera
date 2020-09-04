from pprint import pprint

import networkx as nx

from caldera.data import GraphData
from caldera.testing import annotate_shortest_path
from caldera.transforms import Compose
from caldera.transforms.networkx import NetworkxAttachNumpyOneHot
from caldera.transforms.networkx import NetworkxNodesToStr
from caldera.transforms.networkx import NetworkxSetDefaultFeature
from caldera.transforms.networkx import NetworkxToDirected
from caldera.utils._tools import _resolve_range
from caldera.utils.nx.generators import _uuid_chain
from caldera.utils.nx.generators import chain_graph
from caldera.utils.nx.generators import compose_and_connect
from caldera.utils.nx.generators import random_graph


def generate_shorest_path_example(n_nodes, density, path_length):
    d = _resolve_range(density)
    l = _resolve_range(path_length)
    path = list(_uuid_chain(l))
    h = chain_graph(path, nx.Graph)
    g = random_graph(n_nodes, density=d)
    graph = compose_and_connect(g, h, d)

    annotate_shortest_path(
        graph,
        True,
        True,
        source_key="source",
        target_key="target",
        path_key="shortest_path",
        source=path[0],
        target=path[-1],
    )

    preprocess = Compose(
        [
            NetworkxSetDefaultFeature(
                node_default={"source": False, "target": False, "shortest_path": False},
                edge_default={"shortest_path": False},
            ),
            NetworkxAttachNumpyOneHot(
                "node", "source", "_features", classes=[False, True]
            ),
            NetworkxAttachNumpyOneHot(
                "node", "target", "_features", classes=[False, True]
            ),
            # NetworkxAttachNumpyOneHot(
            #     "edge", "shortest_path", "_target", classes=[False, True]
            # ),
            # NetworkxAttachNumpyOneHot(
            #     "node", "shortest_path", "_target", classes=[False, True]
            # ),
            NetworkxNodesToStr(),
            NetworkxToDirected(),
        ]
    )
    copied_graph = preprocess([graph])[0]

    for n, ndata in copied_graph.nodes(data=True):
        print(ndata)

    # 5. fill in missing features
    GraphData.from_networkx(copied_graph, feature_key="_features")
    GraphData.from_networkx(copied_graph, feature_key="_target")

    return copied_graph


def test_():
    generate_shorest_path_example(100, 0.01, 10)
