from caldera.testing import annotate_shortest_path
from caldera.transforms import Compose
from caldera.transforms.networkx import NetworkxSetDefaultFeature, NetworkxAttachNumpyOneHot, NetworkxNodesToStr, \
    NetworkxToDirected
from caldera.utils._tools import _resolve_range
from caldera.utils.nx.generators import compose_and_connect, random_graph, chain_graph, _uuid_chain
import networkx as nx
import numpy as np


def generate_shorest_path_example(n_nodes, density, path_length, compose_density = None):
    d = _resolve_range(density)
    l = _resolve_range(path_length)
    if compose_density is None:
        cd = d
    else:
        cd = _resolve_range(compose_density)
    path = list(_uuid_chain(l))
    h = chain_graph(path, nx.Graph)
    g = random_graph(n_nodes, density=d)
    graph = compose_and_connect(g, h, cd)

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
            NetworkxAttachNumpyOneHot(
                "edge", "shortest_path", "_target", classes=[False, True]
            ),
            NetworkxAttachNumpyOneHot(
                "node", "shortest_path", "_target", classes=[False, True]
            ),
            NetworkxSetDefaultFeature(
                node_default={"_features": np.array([0.0]), "_target": np.array([0.0])},
                edge_default={"_features": np.array([0.0]), "_target": np.array([0.0])},
                global_default={
                    "_features": np.array([0.0]),
                    "_target": np.array([0.0]),
                },
            ),
            NetworkxNodesToStr(),
            NetworkxToDirected(),
        ]
    )

    return preprocess([graph])[0]
