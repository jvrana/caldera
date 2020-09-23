"""
File containing the network
"""

from typing import List

from caldera.data import GraphBatch
from .configuration import GraphNetConfig, Config
import torch
from pytorch_lightning import LightningModule

from caldera.models import GraphEncoder, GraphCore
from caldera.blocks import MLP, Flex, EdgeBlock, NodeBlock, GlobalBlock, MultiAggregator
from caldera.blocks import (
    AggregatingEdgeBlock,
    AggregatingGlobalBlock,
    AggregatingNodeBlock,
)


class EdgeBlockEncoder(EdgeBlock):
    def __init__(self, size: int, dropout: float):
        super().__init__(Flex(MLP)(Flex.d(), size, dropout))


class NodeBlockEncoder(NodeBlock):
    def __init__(self, size: int, dropout: float):
        super().__init__(Flex(MLP)(Flex.d(), size, dropout))


class GlobalBlockEncoder(GlobalBlock):
    def __init__(self, size: int, dropout: float):
        super().__init__(Flex(MLP)(Flex.d(), size, dropout))


class NodeBlockCore(AggregatingNodeBlock):
    def __init__(self, layers, dropout, layer_norm, aggregator, aggregator_activation):
        super().__init__(
            Flex(MLP)(Flex.d(), *layers, dropout=dropout, layer_norm=layer_norm),
            Flex(MultiAggregator)(
                Flex.d(), aggregator=aggregator, activation=aggregator_activation
            ),
        )


class EdgeBlockCore(AggregatingEdgeBlock):
    def __init__(self, layers, dropout, layer_norm):
        super().__init__(
            Flex(MLP)(Flex.d(), *layers, dropout=dropout, layer_norm=layer_norm)
        )


class GlobalBlockCore(AggregatingGlobalBlock):
    def __init__(
        self,
        layers,
        dropout,
        layer_norm,
        edge_aggregator,
        node_aggregator,
        aggregator_activation,
    ):
        super().__init__(
            mlp=Flex(MLP)(Flex.d(), *layers, dropout=dropout, layer_norm=layer_norm),
            edge_aggregator=Flex(MultiAggregator)(
                Flex.d(), aggregator=edge_aggregator, activation=aggregator_activation
            ),
            node_aggregator=Flex(MultiAggregator)(
                Flex.d(), aggregator=node_aggregator, activation=aggregator_activation
            ),
        )


# TODO: is activator getting passed?


class Network(torch.nn.Module):
    def __init__(self, config: GraphNetConfig):
        """Do stuff with cfg. Use your interpreter/IDE to help construction"""
        super().__init__()
        self.config = config
        self.encoder = self.init_encoder()
        self.core = self.init_core()
        self.decoder = self.init_decoder()
        self.transform = self.init_out_transform()

    # TODO: this size is going to be different, right??
    def init_encoder(self) -> GraphEncoder:
        return GraphEncoder(
            edge_block=EdgeBlockEncoder(self.config.edge_encode.size, self.config.edge_encode.dropout),
            node_block=NodeBlockEncoder(self.config.node_encode.size, self.config.node_encode.dropout),
            global_block=GlobalBlockEncoder(self.config.glob_encode.size, self.config.glob_encode.dropout),
        )

    def init_decoder(self) -> GraphEncoder:
        return GraphEncoder(
            edge_block=EdgeBlockEncoder(self.config.edge_decode.size, self.config.edge_decode.dropout),
            node_block=NodeBlockEncoder(self.config.node_decode.size, self.config.node_decode.dropout),
            global_block=GlobalBlockEncoder(self.config.glob_decode.size, self.config.glob_decode.dropout),
        )


    def init_core(self):
        return GraphCore(
            edge_block=EdgeBlockCore(
                layers=self.config.edge_core.layers,
                dropout=self.config.edge_core.dropout,
                layer_norm=self.config.edge_core.layer_norm
            ),
            node_block=NodeBlockCore(
                layers=self.config.node_core.layers,
                dropout=self.config.node_core.dropout,
                aggregator=self.config.edge_block_to_node_aggregators,
                aggregator_activation=self.config.aggregator_activation,
                layer_norm=self.config.node_core.layer_norm,
            ),
            global_block=GlobalBlockCore(
                layers=self.config.glob_core.layers,
                dropout=self.config.glob_core.dropout,
                edge_aggregator=self.config.global_block_to_edge_aggregators,
                node_aggregator=self.config.global_block_to_node_aggregators,
                layer_norm=self.config.glob_core.layer_norm,
                aggregator_activation=self.config.aggregator_activation
            ),
            pass_global_to_node=self.config.pass_global_to_node,
            pass_global_to_edge=self.config.pass_global_to_edge,
        )

    # TODO: each edge should have its own activation
    def init_out_transform(self):
        return GraphEncoder(
            edge_block=torch.nn.Linear(
                Flex(torch.nn.Linear)(Flex.d(), self.config.edge_out.size),
                self.config.edge_out.activation,
            ),
            node_block=torch.nn.Linear(
                Flex(torch.nn.Linear)(Flex.d(), self.config.node_out.size),
                self.config.node_out.activation,
            ),
            global_block=torch.nn.Linear(
                Flex(torch.nn.Linear)(Flex.d(), self.config.glob.out),
                self.config.glob_out.activation,
            ),
        )

    def encode(self, data: GraphBatch) -> GraphBatch:
        e, x, g = self.encoder(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def decode(self, data: GraphBatch) -> GraphBatch:
        e, x, g = self.decoder(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def core_process(self, latent0: GraphBatch, data: GraphBatch) -> GraphBatch:
        e = torch.cat([latent0.e, data.e], dim=1)
        x = torch.cat([latent0.x, data.x], dim=1)
        g = torch.cat([latent0.g, data.g], dim=1)
        data = GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)
        e, x, g = self.core(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def out_transform(self, data: GraphBatch) -> GraphBatch:
        e, x, g = self.output_transform(data)
        return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

    def forward(
        self, data: GraphBatch, steps: int, save_all: bool = False
    ) -> List[GraphBatch]:
        data = self.encode(data)
        latent0 = data

        outputs = []
        for _ in range(steps):
            data = self.core_process(latent0, data)
            data = self.decode(data)
            out_data = self.out_transform(data)
            if save_all:
                outputs.append(out_data)
            else:
                outputs = [out_data]
        return outputs


class TrainingModule(LightningModule):
    def __init__(self, config: Config):
        self.network = Network(config.network)
        pass  # do additional configuration with config file...

    #     latent_sizes=(32, 32, 32),
    #     out_sizes=(1, 1, 1),
    #     latent_depths=(1, 1, 1),
    #     dropout: float = None,
    #     pass_global_to_edge: bool = True,
    #     pass_global_to_node: bool = True,
    #     activation=defaults.activation,
    #     out_activation=defaults.activation,
    #     edge_to_node_aggregators=tuple(["add", "max", "mean", "min"]),
    #     edge_to_global_aggregators=tuple(["add", "max", "mean", "min"]),
    #     node_to_global_aggregators=tuple(["add", "max", "mean", "min"]),
    #     aggregator_activation=defaults.activation,
    # ):
    #


#
# class Network(torch.nn.Module):
#     def __init__(self, cfg: GraphNetConfig):
#         super().__init__()
#         self.config: GraphNetConfig = cfg
#         # self.encoder = self._init_encoder()
#         # self.core = self._init_core()
#         # self.decoder = self._init_encoder()
#         # self.output_transform = self._init_out_transform()
#
#     def _init_encoder(self):
#         return GraphEncoder(
#             edge_block=EdgeBlock(
#                 Flex(MLP)(
#                     Flex.d(),
#                     self.config.edge.size,
#                     self.config.dropout
#                 )
#             ),
#             node_block=NodeBlock(
#                 Flex(MLP)(
#                     Flex.d(),
#                     self.config.node.size,
#                     self.config.dropout
#                 )
#             ),
#             global_block=GlobalBlock(
#                 Flex(MLP)(
#                     Flex.d(),
#                     self.config.glob.size,
#                     self.config.dropout
#                 )
#             )
#         )
#
#     def _init_core(self):
#         edge_layers = [self.config["sizes"]["latent"]["edge"]] * self.config["sizes"][
#             "latent"
#         ]["edge_depth"]
#         node_layers = [self.config["sizes"]["latent"]["node"]] * self.config["sizes"][
#             "latent"
#         ]["node_depth"]
#         global_layers = [self.config["sizes"]["latent"]["global"]] * self.config[
#             "sizes"
#         ]["latent"]["global_depth"]
#
#         return GraphCore(
#             AggregatingEdgeBlock(
#                 torch.nn.Sequential(
#                     Flex(MLP)(
#                         Flex.d(),
#                         *self.config.edge.layers,
#                         dropout=self.config.dropout,
#                         layer_norm=self.config.edge.layer_norm
#                     ),
#                 )
#             ),
#             AggregatingNodeBlock(
#                 torch.nn.Sequential(
#                     Flex(MLP)(
#                         Flex.d(),
#                         *self.config.node.layers,
#                         dropout=self.config.dropout,
#                         layer_norm=self.config.node.layer_norm
#                     ),
#                 ),
#                 Flex(MultiAggregator)(
#                     Flex.d(),
#                     self.config["node_block_aggregator"],
#                     activation=self.config["aggregator_activation"],
#                 ),
#             ),
#             AggregatingGlobalBlock(
#                 torch.nn.Sequential(
#                     Flex(MLP)(
#                         Flex.d(),
#                         *global_layers,
#                         dropout=self.config["dropout"],
#                         layer_norm=True
#                     ),
#                 ),
#                 edge_aggregator=Flex(MultiAggregator)(
#                     Flex.d(),
#                     self.config["global_block_to_edge_aggregator"],
#                     activation=self.config["aggregator_activation"],
#                 ),
#                 node_aggregator=Flex(MultiAggregator)(
#                     Flex.d(),
#                     self.config["global_block_to_node_aggregator"],
#                     activation=self.config["aggregator_activation"],
#                 ),
#             ),
#             pass_global_to_edge=self.config["pass_global_to_edge"],
#             pass_global_to_node=self.config["pass_global_to_node"],
#         )
#
#     def _init_out_transform(self):
#         return GraphEncoder(
#             EdgeBlock(
#                 torch.nn.Sequential(
#                     Flex(torch.nn.Linear)(
#                         Flex.d(), self.config["sizes"]["out"]["edge"]
#                     ),
#                     self.config["sizes"]["out"]["activation"](),
#                 )
#             ),
#             NodeBlock(
#                 torch.nn.Sequential(
#                     Flex(torch.nn.Linear)(
#                         Flex.d(), self.config["sizes"]["out"]["node"]
#                     ),
#                     self.config["sizes"]["out"]["activation"](),
#                 )
#             ),
#             GlobalBlock(
#                 torch.nn.Sequential(
#                     Flex(torch.nn.Linear)(
#                         Flex.d(), self.config["sizes"]["out"]["global"]
#                     ),
#                     self.config["sizes"]["out"]["activation"](),
#                 )
#             ),
#         )
#
#     def _forward_encode(self, data):
#         e, x, g = self.encoder(data)
#         return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)
#
#     def _forward_decode(self, data):
#         e, x, g = self.decoder(data)
#         return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)
#
#     def _forward_core(self, latent0, data):
#         e = torch.cat([latent0.e, data.e], dim=1)
#         x = torch.cat([latent0.x, data.x], dim=1)
#         g = torch.cat([latent0.g, data.g], dim=1)
#         data = GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)
#         e, x, g = self.core(data)
#         return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)
#
#     def _forward_out(self, data):
#         e, x, g = self.output_transform(data)
#         return GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)
#
#     def forward(self, data, steps, save_all: bool = False):
#         data = self._forward_encode(data)
#         latent0 = data
#
#         outputs = []
#         for _ in range(steps):
#             data = self._forward_core(latent0, data)
#             data = self._forward_decode(data)
#             out_data = self._forward_out(data)
#             if save_all:
#                 outputs.append(out_data)
#             else:
#                 outputs = [out_data]
#         return outputs
