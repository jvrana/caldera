import networkx as nx
import torch

from caldera.blocks import AggregatingEdgeBlock
from caldera.blocks import AggregatingGlobalBlock
from caldera.blocks import AggregatingNodeBlock
from caldera.blocks import EdgeBlock
from caldera.blocks import Flex
from caldera.blocks import GlobalBlock
from caldera.blocks import MLP
from caldera.blocks import MultiAggregator
from caldera.blocks import NodeBlock
from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data import GraphDataLoader
from caldera.defaults import CalderaDefaults as defaults
from caldera.models import GraphCore
from caldera.models import GraphEncoder
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
            NetworkxAttachNumpyOneHot(
                "edge", "shortest_path", "_target", classes=[False, True]
            ),
            NetworkxAttachNumpyOneHot(
                "node", "shortest_path", "_target", classes=[False, True]
            ),
            NetworkxNodesToStr(),
            NetworkxToDirected(),
        ]
    )

    return preprocess([graph])[0]


def test_generate_shortest_path_example():
    g = generate_shorest_path_example(100, 0.01, 10)
    d1 = GraphData.from_networkx(g, feature_key="_features")
    d2 = GraphData.from_networkx(g, feature_key="_target")

    assert tuple(d1.shape) == (4, 0, 0)
    assert tuple(d2.shape) == (2, 2, 0)


class Network(torch.nn.Module):
    def __init__(
        self,
        latent_sizes=(128, 128, 1),
        depths=(1, 1, 1),
        dropout: float = None,
        pass_global_to_edge: bool = True,
        pass_global_to_node: bool = True,
        edge_to_node_aggregators=tuple(["add", "max", "mean", "min"]),
        edge_to_global_aggregators=tuple(["add", "max", "mean", "min"]),
        node_to_global_aggregators=tuple(["add", "max", "mean", "min"]),
        aggregator_activation=defaults.activation,
    ):
        super().__init__()
        self.config = {
            "latent_size": {
                "node": latent_sizes[1],
                "edge": latent_sizes[0],
                "global": latent_sizes[2],
                "core_node_block_depth": depths[0],
                "core_edge_block_depth": depths[1],
                "core_global_block_depth": depths[2],
            },
            "node_block_aggregator": edge_to_node_aggregators,
            "global_block_to_node_aggregator": node_to_global_aggregators,
            "global_block_to_edge_aggregator": edge_to_global_aggregators,
            "aggregator_activation": aggregator_activation,
            "pass_global_to_edge": pass_global_to_edge,
            "pass_global_to_node": pass_global_to_node,
        }
        self.encoder = GraphEncoder(
            EdgeBlock(Flex(MLP)(Flex.d(), latent_sizes[0], dropout=dropout)),
            NodeBlock(Flex(MLP)(Flex.d(), latent_sizes[1], dropout=dropout)),
            GlobalBlock(Flex(MLP)(Flex.d(), latent_sizes[2], dropout=dropout)),
        )

        edge_layers = [self.config["latent_size"]["edge"]] * self.config["latent_size"][
            "core_edge_block_depth"
        ]
        node_layers = [self.config["latent_size"]["node"]] * self.config["latent_size"][
            "core_node_block_depth"
        ]
        global_layers = [self.config["latent_size"]["global"]] * self.config[
            "latent_size"
        ]["core_global_block_depth"]

        self.core = GraphCore(
            AggregatingEdgeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), *edge_layers, dropout=dropout, layer_norm=True),
                    # Flex(torch.nn.Linear)(Flex.d(), edge_layers[-1])
                )
            ),
            AggregatingNodeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), *node_layers, dropout=dropout, layer_norm=True),
                    # Flex(torch.nn.Linear)(Flex.d(), node_layers[-1])
                ),
                Flex(MultiAggregator)(
                    Flex.d(),
                    self.config["node_block_aggregator"],
                    activation=self.config["aggregator_activation"],
                ),
            ),
            AggregatingGlobalBlock(
                torch.nn.Sequential(
                    Flex(MLP)(
                        Flex.d(), *global_layers, dropout=dropout, layer_norm=True
                    ),
                    # Flex(torch.nn.Linear)(Flex.d(), global_layers[-1])
                ),
                edge_aggregator=Flex(MultiAggregator)(
                    Flex.d(),
                    self.config["global_block_to_edge_aggregator"],
                    activation=self.config["aggregator_activation"],
                ),
                node_aggregator=Flex(MultiAggregator)(
                    Flex.d(),
                    self.config["global_block_to_node_aggregator"],
                    activation=self.config["aggregator_activation"],
                ),
            ),
            pass_global_to_edge=self.config["pass_global_to_edge"],
            pass_global_to_node=self.config["pass_global_to_node"],
        )

        self.decoder = GraphEncoder(
            EdgeBlock(
                Flex(MLP)(Flex.d(), latent_sizes[0], latent_sizes[0], dropout=dropout)
            ),
            NodeBlock(
                Flex(MLP)(Flex.d(), latent_sizes[1], latent_sizes[1], dropout=dropout)
            ),
            GlobalBlock(Flex(MLP)(Flex.d(), latent_sizes[2])),
        )

        self.output_transform = GraphEncoder(
            EdgeBlock(Flex(torch.nn.Sigmoid)(Flex.d(), 1)),
            NodeBlock(Flex(torch.nn.Sigmoid)(Flex.d(), 1)),
            GlobalBlock(Flex(torch.nn.Linear)(Flex.d(), 1)),
        )

    def forward(self, data, steps, save_all: bool = False):
        # encoded
        e, x, g = self.encoder(data)
        data = GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

        # graph topography data
        edges = data.edges
        node_idx = data.node_idx
        edge_idx = data.edge_idx
        latent0 = data

        meta = (edges, node_idx, edge_idx)

        outputs = []
        for _ in range(steps):
            # core processing step
            e = torch.cat([latent0.e, e], dim=1)
            x = torch.cat([latent0.x, x], dim=1)
            g = torch.cat([latent0.g, g], dim=1)
            data = GraphBatch(x, e, g, *meta)
            e, x, g = self.core(data)

            # decode
            data = GraphBatch(x, e, g, *meta)

            _e, _x, _g = self.decoder(data)
            decoded = GraphBatch(_x, _e, _g, *meta)

            # transform
            _e, _x, _g = self.output_transform(decoded)
            print()
            gt = GraphBatch(_x, _e, _g, edges, node_idx, edge_idx)
            if save_all:
                outputs.append(gt)
            else:
                outputs = [gt]

        return outputs


def test_train_shortest_path():
    graphs = [generate_shorest_path_example(100, 0.01, 10) for _ in range(10)]
    input_data = [GraphData.from_networkx(g, feature_key="_features") for g in graphs]
    target_data = [GraphData.from_networkx(g, feature_key="_target") for g in graphs]

    loader = GraphDataLoader(input_data, target_data, batch_size=32, shuffle=True)

    agg = lambda: Flex(MultiAggregator)(Flex.d(), ["add", "mean", "max", "min"])

    network = Network()

    for _ in range(2):
        for input_batch, target_batch in loader:
            output = network(input_batch, 10)

    # print(datalist[0])
    # dataset = GraphDataset(datalist)
    # loader = GraphDataLoader(dataset, batch_size=32, shuffle=True)
    # for batch in loader:
    #     print(batch)
