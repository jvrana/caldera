import torch

from caldera.blocks import AggregatingEdgeBlock
from caldera.blocks import AggregatingGlobalBlock
from caldera.blocks import AggregatingNodeBlock
from caldera.blocks import Dense
from caldera.blocks import EdgeBlock
from caldera.blocks import Flex
from caldera.blocks import GlobalBlock
from caldera.blocks import MultiAggregator
from caldera.blocks import NodeBlock
from caldera.data import GraphBatch
from caldera.defaults import CalderaDefaults as defaults
from caldera.models import GraphCore
from caldera.models import GraphEncoder


class Network(torch.nn.Module):
    def __init__(
        self,
        latent_sizes=(16, 16, 1),
        out_sizes=(1, 1, 1),
        latent_depths=(1, 1, 1),
        dropout: float = None,
        pass_global_to_edge: bool = True,
        pass_global_to_node: bool = True,
        activation=defaults.activation,
        out_activation=defaults.activation,
        edge_to_node_aggregators=tuple(["add", "max", "mean", "min"]),
        edge_to_global_aggregators=tuple(["add", "max", "mean", "min"]),
        node_to_global_aggregators=tuple(["add", "max", "mean", "min"]),
        aggregator_activation=defaults.activation,
    ):
        super().__init__()
        self.config = {
            "sizes": {
                "latent": {
                    "edge": latent_sizes[0],
                    "node": latent_sizes[1],
                    "global": latent_sizes[2],
                    "edge_depth": latent_depths[0],
                    "node_depth": latent_depths[1],
                    "global_depth": latent_depths[2],
                },
                "out": {
                    "edge": out_sizes[0],
                    "node": out_sizes[1],
                    "global": out_sizes[2],
                    "activation": out_activation,
                },
            },
            "activation": activation,
            "dropout": dropout,
            "node_block_aggregator": edge_to_node_aggregators,
            "global_block_to_node_aggregator": node_to_global_aggregators,
            "global_block_to_edge_aggregator": edge_to_global_aggregators,
            "aggregator_activation": aggregator_activation,
            "pass_global_to_edge": pass_global_to_edge,
            "pass_global_to_node": pass_global_to_node,
        }

        ###########################
        # encoder
        ###########################

        self.encoder = self._init_encoder()
        self.core = self._init_core()
        self.decoder = self._init_encoder()
        self.output_transform = self._init_out_transform()

        self.output_transform = GraphEncoder(
            EdgeBlock(
                torch.nn.Sequential(
                    Flex(torch.nn.Linear)(Flex.d(), 1), torch.nn.Sigmoid()
                )
            ),
            NodeBlock(
                torch.nn.Sequential(
                    Flex(torch.nn.Linear)(Flex.d(), 1), torch.nn.Sigmoid()
                )
            ),
            GlobalBlock(Flex(torch.nn.Linear)(Flex.d(), 1)),
        )

    def _init_encoder(self):
        return GraphEncoder(
            EdgeBlock(
                Flex(Dense)(
                    Flex.d(),
                    self.config["sizes"]["latent"]["edge"],
                    dropout=self.config["dropout"],
                )
            ),
            NodeBlock(
                Flex(Dense)(
                    Flex.d(),
                    self.config["sizes"]["latent"]["node"],
                    dropout=self.config["dropout"],
                )
            ),
            GlobalBlock(
                Flex(Dense)(
                    Flex.d(),
                    self.config["sizes"]["latent"]["global"],
                    dropout=self.config["dropout"],
                )
            ),
        )

    def _init_core(self):
        edge_layers = [self.config["sizes"]["latent"]["edge"]] * self.config["sizes"][
            "latent"
        ]["edge_depth"]
        node_layers = [self.config["sizes"]["latent"]["node"]] * self.config["sizes"][
            "latent"
        ]["node_depth"]
        global_layers = [self.config["sizes"]["latent"]["global"]] * self.config[
            "sizes"
        ]["latent"]["global_depth"]

        return GraphCore(
            AggregatingEdgeBlock(
                torch.nn.Sequential(
                    Flex(Dense)(
                        Flex.d(),
                        *edge_layers,
                        dropout=self.config["dropout"],
                        layer_norm=True
                    ),
                )
            ),
            AggregatingNodeBlock(
                torch.nn.Sequential(
                    Flex(Dense)(
                        Flex.d(),
                        *node_layers,
                        dropout=self.config["dropout"],
                        layer_norm=True
                    ),
                ),
                Flex(MultiAggregator)(
                    Flex.d(),
                    self.config["node_block_aggregator"],
                    activation=self.config["aggregator_activation"],
                ),
            ),
            AggregatingGlobalBlock(
                torch.nn.Sequential(
                    Flex(Dense)(
                        Flex.d(),
                        *global_layers,
                        dropout=self.config["dropout"],
                        layer_norm=True
                    ),
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

    def _init_out_transform(self):
        return GraphEncoder(
            EdgeBlock(
                torch.nn.Sequential(
                    Flex(torch.nn.Linear)(
                        Flex.d(), self.config["sizes"]["out"]["edge"]
                    ),
                    self.config["sizes"]["out"]["activation"](),
                )
            ),
            NodeBlock(
                torch.nn.Sequential(
                    Flex(torch.nn.Linear)(
                        Flex.d(), self.config["sizes"]["out"]["node"]
                    ),
                    self.config["sizes"]["out"]["activation"](),
                )
            ),
            GlobalBlock(
                torch.nn.Sequential(
                    Flex(torch.nn.Linear)(
                        Flex.d(), self.config["sizes"]["out"]["global"]
                    ),
                    self.config["sizes"]["out"]["activation"](),
                )
            ),
        )

    def _forward_encode(self, data):
        e, x, g = self.encoder(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def _forward_decode(self, data):
        e, x, g = self.decoder(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def _forward_core(self, latent0, data):
        e = torch.cat([latent0.e, data.e], dim=1)
        x = torch.cat([latent0.x, data.x], dim=1)
        g = torch.cat([latent0.g, data.g], dim=1)
        data = GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)
        e, x, g = self.core(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def _forward_out(self, data):
        e, x, g = self.output_transform(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def forward(self, data, steps, save_all: bool = False):
        data = self._forward_encode(data)
        latent0 = data

        outputs = []
        for _ in range(steps):
            data = self._forward_core(latent0, data)
            data = self._forward_decode(data)
            out_data = self._forward_out(data)
            if save_all:
                outputs.append(out_data)
            else:
                outputs = [out_data]
        return outputs
