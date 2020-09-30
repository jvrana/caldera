"""File containing the network."""
from functools import wraps
from typing import Callable
from typing import List
from typing import Type

import torch

from .configuration import NetConfig
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
from caldera.models import GraphCore
from caldera.models import GraphEncoder


class EdgeBlockEncoder(EdgeBlock):
    def __init__(self, size: int, dropout: float, activation: Type[torch.nn.Module]):
        super().__init__(
            Flex(MLP)(Flex.d(), size, dropout=dropout, activation=activation)
        )


class NodeBlockEncoder(NodeBlock):
    def __init__(self, size: int, dropout: float, activation: Type[torch.nn.Module]):
        super().__init__(
            Flex(MLP)(Flex.d(), size, dropout=dropout, activation=activation)
        )


class GlobalBlockEncoder(GlobalBlock):
    def __init__(self, size: int, dropout: float, activation: Type[torch.nn.Module]):
        super().__init__(
            Flex(MLP)(Flex.d(), size, dropout=dropout, activation=activation)
        )


class NodeBlockCore(AggregatingNodeBlock):
    def __init__(
        self,
        layers,
        dropout,
        layer_norm,
        aggregator,
        aggregator_activation: Type[torch.nn.Module],
    ):
        super().__init__(
            Flex(MLP)(Flex.d(), *layers, dropout=dropout, layer_norm=layer_norm),
            Flex(MultiAggregator)(
                Flex.d(), aggregators=aggregator, activation=aggregator_activation
            ),
        )


class EdgeBlockCore(AggregatingEdgeBlock):
    def __init__(self, layers, dropout: float, layer_norm: bool):
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
                Flex.d(), aggregators=edge_aggregator, activation=aggregator_activation
            ),
            node_aggregator=Flex(MultiAggregator)(
                Flex.d(), aggregators=node_aggregator, activation=aggregator_activation
            ),
        )


class LinearTransformation(torch.nn.Module):
    def __init__(self, size: int, activation: Type[torch.nn.Module]):
        super().__init__()
        layers = [Flex(torch.nn.Linear)(Flex.d(), size)]
        if activation is not None:
            layers.append(activation())
        self.layer = torch.nn.Sequential(*layers)

    def forward(self, data):
        return self.layer(data)


# TODO: is activator getting passed?


class Network(torch.nn.Module):
    def __init__(self, config: NetConfig):
        """Do stuff with cfg.

        Use your interpreter/IDE to help construction
        """
        super().__init__()
        self.config = config
        self.encoder = self.init_encoder()
        self.core = self.init_core()
        self.decoder = self.init_encoder()
        self.output_transform = self.init_out_transform()

    # TODO: this size is going to be different, right??
    def init_encoder(self) -> GraphEncoder:
        return GraphEncoder(
            edge_block=EdgeBlockEncoder(
                self.config.encode.edge.size,
                self.config.encode.edge.dropout,
                self.config.get_activation(self.config.encode.edge.activation),
            ),
            node_block=NodeBlockEncoder(
                self.config.encode.node.size,
                self.config.encode.node.dropout,
                self.config.get_activation(self.config.encode.node.activation),
            ),
            global_block=GlobalBlockEncoder(
                self.config.encode.glob.size,
                self.config.encode.glob.dropout,
                self.config.get_activation(self.config.encode.glob.activation),
            ),
        )

    # def init_decoder(self) -> GraphEncoder:
    #     return GraphEncoder(
    #         edge_block=EdgeBlockEncoder(
    #             self.config.decode.edge.size,
    #             self.config.decode.edge.dropout,
    #             self.config.get_activation(self.config.decode.edge.activation),
    #         ),
    #         node_block=NodeBlockEncoder(
    #             self.config.decode.node.size,
    #             self.config.decode.node.dropout,
    #             self.config.get_activation(self.config.decode.node.activation),
    #         ),
    #         global_block=GlobalBlockEncoder(
    #             self.config.decode.glob.size,
    #             self.config.decode.glob.dropout,
    #             self.config.get_activation(self.config.decode.glob.activation),
    #         ),
    #     )

    def init_core(self):
        return GraphCore(
            edge_block=EdgeBlockCore(
                layers=self.config.core.edge.layers,
                dropout=self.config.core.edge.dropout,
                layer_norm=self.config.core.edge.layer_norm,
            ),
            node_block=NodeBlockCore(
                layers=self.config.core.node.layers,
                dropout=self.config.core.node.dropout,
                aggregator=self.config.edge_block_to_node_aggregators,
                aggregator_activation=self.config.get_activation(
                    self.config.aggregator_activation
                ),
                layer_norm=self.config.core.node.layer_norm,
            ),
            global_block=GlobalBlockCore(
                layers=self.config.core.glob.layers,
                dropout=self.config.core.glob.dropout,
                edge_aggregator=self.config.global_block_to_edge_aggregators,
                node_aggregator=self.config.global_block_to_node_aggregators,
                layer_norm=self.config.core.glob.layer_norm,
                aggregator_activation=self.config.get_activation(
                    self.config.aggregator_activation
                ),
            ),
            pass_global_to_node=self.config.pass_global_to_node,
            pass_global_to_edge=self.config.pass_global_to_edge,
        )

    # TODO: each edge should have its own activation
    def init_out_transform(self):
        return GraphEncoder(
            edge_block=EdgeBlock(
                LinearTransformation(
                    self.config.out.edge.size,
                    self.config.get_activation(self.config.out.edge.activation),
                ),
            ),
            node_block=NodeBlock(
                LinearTransformation(
                    self.config.out.node.size,
                    self.config.get_activation(self.config.out.node.activation),
                )
            ),
            global_block=GlobalBlock(
                LinearTransformation(
                    self.config.out.glob.size,
                    self.config.get_activation(self.config.out.glob.activation),
                )
            ),
        )

    # TODO: fix the ordering of e, x, g and graph tuple
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

    @wraps(forward)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
