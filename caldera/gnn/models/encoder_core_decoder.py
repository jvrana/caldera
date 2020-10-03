import torch

from caldera.data import GraphBatch
from caldera.gnn.blocks import AggregatingEdgeBlock
from caldera.gnn.blocks import AggregatingGlobalBlock
from caldera.gnn.blocks import AggregatingNodeBlock
from caldera.gnn.blocks import Aggregator
from caldera.gnn.blocks import Dense
from caldera.gnn.blocks import EdgeBlock
from caldera.gnn.blocks import Flex
from caldera.gnn.blocks import GlobalBlock
from caldera.gnn.blocks import NodeBlock
from caldera.gnn.models.graph_core import GraphCore
from caldera.gnn.models.graph_encoder import GraphEncoder


class EncodeCoreDecode(torch.nn.Module):
    def __init__(
        self,
        latent_sizes=(128, 128, 1),
        output_sizes=(1, 1, 1),
        depths=(1, 1, 1),
        layer_norm: bool = True,
        dropout: float = None,
        pass_global_to_edge: bool = True,
        pass_global_to_node: bool = True,
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
            "output_size": {
                "edge": output_sizes[0],
                "node": output_sizes[1],
                "global": output_sizes[2],
            },
            "node_block_aggregator": "add",
            "global_block_to_node_aggregator": "add",
            "global_block_to_edge_aggregator": "add",
            "pass_global_to_edge": pass_global_to_edge,
            "pass_global_to_node": pass_global_to_node,
        }

        def mlp(*layer_sizes):
            return Flex(Dense)(
                Flex.d(), *layer_sizes, layer_norm=layer_norm, dropout=dropout
            )

        self.encoder = GraphEncoder(
            EdgeBlock(mlp(latent_sizes[0])),
            NodeBlock(mlp(latent_sizes[1])),
            GlobalBlock(mlp(latent_sizes[2])),
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
            AggregatingEdgeBlock(mlp(*edge_layers)),
            AggregatingNodeBlock(
                mlp(*node_layers), Aggregator(self.config["node_block_aggregator"])
            ),
            AggregatingGlobalBlock(
                mlp(*global_layers),
                edge_aggregator=Aggregator(
                    self.config["global_block_to_edge_aggregator"]
                ),
                node_aggregator=Aggregator(
                    self.config["global_block_to_node_aggregator"]
                ),
            ),
            pass_global_to_edge=self.config["pass_global_to_edge"],
            pass_global_to_node=self.config["pass_global_to_node"],
        )

        self.decoder = GraphEncoder(
            EdgeBlock(mlp(latent_sizes[0])),
            NodeBlock(mlp(latent_sizes[1])),
            GlobalBlock(mlp(latent_sizes[2])),
        )

        self.output_transform = GraphEncoder(
            EdgeBlock(Flex(torch.nn.Linear)(Flex.d(), output_sizes[0])),
            NodeBlock(Flex(torch.nn.Linear)(Flex.d(), output_sizes[1])),
            GlobalBlock(Flex(torch.nn.Linear)(Flex.d(), output_sizes[2])),
        )

    def forward(self, data, steps, save_all: bool = True):
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
            gt = GraphBatch(_x, _e, _g, edges, node_idx, edge_idx)
            if save_all:
                outputs.append(gt)
            else:
                outputs[0] = gt
        return outputs
