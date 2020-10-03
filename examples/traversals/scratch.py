##########################################################
# Relative Imports
##########################################################
import sys
from functools import partial
from os.path import isfile
from os.path import join


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

##########################################################
# Main
##########################################################

from caldera.transforms import Compose
import numpy as np
from caldera.transforms.networkx import (
    NetworkxSetDefaultFeature,
    NetworkxAttachNumpyOneHot,
    NetworkxAttachNumpyFeatures,
    NetworkxNodesToStr,
    NetworkxToDirected,
)
from caldera.utils import functional as fn
from caldera.utils.nx.generators import random_graph
from torch import nn
from caldera import gnn


preprocess = Compose(
    [
        NetworkxSetDefaultFeature(
            node_default={"source": False, "target": False, "shortest_path": False},
            edge_default={"shortest_path": False},
        ),
        NetworkxAttachNumpyOneHot(
            "node", "source", "_features", classes=[False, True]
        ),  # label nodes as 'start'
        NetworkxAttachNumpyOneHot(
            "node", "target", "_features", classes=[False, True]
        ),  # label nodes as 'end'
        # attached weight
        NetworkxAttachNumpyFeatures(
            "edge", "weight", "_features", encoding=fn.map_each(lambda x: np.array([x]))
        ),
        NetworkxAttachNumpyOneHot(
            "edge", "shortest_path", "_target", classes=[False, True]
        ),  # label edge as shortest_path
        NetworkxAttachNumpyOneHot(
            "node", "shortest_path", "_target", classes=[False, True]
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


class Network(nn.Module):
    def __init__(self):

        dense = partial(
            gnn.Flex(gnn.Dense), layer_norm=True, dropout=0.2, activation=nn.ReLU
        )

        self.encode = gnn.GraphEncoder(
            edge_block=gnn.EdgeBlock(dense(..., 16)),
            node_block=gnn.NodeBlock(dense(..., 16)),
            global_block=gnn.GlobalBlock(dense(..., 16)),
        )
        self.core = gnn.GraphCore(
            edge_block=gnn.AggregatingEdgeBlock(dense(..., 16)),
            node_block=gnn.AggregatingNodeBlock(
                dense(..., 16), edge_aggregator=gnn.Flex(gnn.Aggregator)(..., "add")
            ),
            global_block=gnn.AggregatingGlobalBlock(
                dense(..., 16),
                edge_aggregator=gnn.Flex(gnn.Aggregator)(..., "add"),
                node_aggregator=gnn.Flex(gnn.Aggregator)(..., "add"),
            ),
        )
        self.out = gnn.GraphEncoder(
            edge_block=gnn.EdgeBlock(
                nn.Sequential(gnn.Flex(nn.Linear)(..., 2), nn.Softmax(0))
            ),
            node_block=gnn.NodeBlock(gnn.Flex(nn.Linear)(..., 2), nn.Softmax(0)),
            global_block=gnn.GlobalBlock(gnn.Flex(nn.Linear)(..., 1), nn.Softmax(0)),
        )

    def forward(self, data):
        latent0 = self.encode(data)
