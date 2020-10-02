# flake8: noqa
##########################################################
# Relative Imports
##########################################################
import argparse
import sys
from os.path import isfile
from os.path import join

from caldera.transforms import Compose
from caldera.transforms.networkx import NetworkxAttachNumpyBool
from caldera.transforms.networkx import NetworkxAttachNumpyFeatures
from caldera.transforms.networkx import NetworkxAttachNumpyOneHot
from caldera.transforms.networkx import NetworkxNodesToStr
from caldera.transforms.networkx import NetworkxSetDefaultFeature
from caldera.transforms.networkx import NetworkxToDirected
from caldera.utils.nx.generators import chain_graph
from caldera.utils.nx.generators import compose_and_connect
from caldera.utils.nx.generators import random_graph
from caldera.utils.nx.generators import uuid_sequence


def find_pkg(name: str, depth: int):
    if depth <= 0:
        ret = None
    else:
        d = [".."] * depth
        path_parts = d + [name, "__init__.py"]

        if isfile(join(*path_parts)):
            ret = d
        else:
            ret = find_pkg(name, depth - 1)
    return ret


def find_and_ins_syspath(name: str, depth: int):
    path_parts = find_pkg(name, depth)
    if path_parts is None:
        raise RuntimeError("Could not find {}. Try increasing depth.".format(name))
    path = join(*path_parts)
    if path not in sys.path:
        sys.path.insert(0, path)


try:
    import caldera
except ImportError:
    find_and_ins_syspath("caldera", 3)

from examples.traversals.data import generate_shortest_path_example
from examples.traversals.configuration.data import Uniform
import networkx as nx
import numpy as np
from caldera.utils import functional as fn

if __name__ == "__main__":
    uniform = Uniform(0.1, 100)

    preprocess = Compose(
        [
            NetworkxSetDefaultFeature(
                node_default={"source": False, "target": False, "shortest_path": False},
                edge_default={"shortest_path": False},
            ),
            NetworkxAttachNumpyBool(
                "node", "source", "_features"
            ),  # label nodes as 'start'
            NetworkxAttachNumpyBool(
                "node", "target", "_features"
            ),  # label nodes as 'end'
            # attached weight
            NetworkxAttachNumpyFeatures(
                "edge",
                "weight",
                "_features",
                encoding=fn.map_each(lambda x: np.array([x])),
            ),
            NetworkxAttachNumpyBool(
                "edge", "shortest_path", "_target"
            ),  # label edge as shortest_path
            NetworkxAttachNumpyBool(
                "node", "shortest_path", "_target"
            ),  # label node as shortest_path
            NetworkxSetDefaultFeature(
                global_default={
                    "_features": np.array([1.0]),
                    "_target": np.array([1.0]),
                },
            ),
            NetworkxNodesToStr(),
            NetworkxToDirected(),
        ]
    )

    for i in range(1):
        g = generate_shortest_path_example(
            100, 0.01, path_length=10, weight=uniform, weight_key="weight"
        )

        for _, _, edata in g.edges(data=True):
            assert "weight" in edata

        sources = []
        targets = []
        for n, ndata in g.nodes(data=True):
            if ndata["source"]:
                sources.append(n)
            if ndata["target"]:
                targets.append(n)
        assert len(sources) == 1
        assert len(targets) == 1

        shortest_paths = set()
        for n1, n2, edata in g.edges(data=True):
            if edata["shortest_path"]:
                shortest_paths.add(n1)
                shortest_paths.add(n2)
        import networkx as nx

        path = nx.shortest_path(
            g, source=sources[0], target=targets[0], weight="weight"
        )
        paths = list(
            nx.all_shortest_paths(
                g, source=sources[0], target=targets[0], weight="weight"
            )
        )
        assert len(paths) == 1
        print(path)
        print(shortest_paths)
        assert set(path) == shortest_paths

        for n, ndata in g.nodes(data=True):
            print(ndata)
        print()
        # g = preprocess([g])[0]
        for n, ndata in g.nodes(data=True):
            print(ndata)
        print()

        for _, _, edata in g.edges(data=True):
            print(edata)

        _path = []
        for _, _, edata in g.edges(data=True):
            if edata["_features"][0].item() == 1.0:
                _path.append(_)

        print(sorted(set(shortest_paths)))
        print(sorted(set(_path)))
