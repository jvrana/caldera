from torch import nn

from caldera import gnn
from caldera._experimental.nngraph import NNGraph
from caldera.data import GraphBatch
from caldera.data import GraphTuple
from examples.traversals.configuration import NetConfig
from examples.traversals.configuration.network import LayerConfig, NetComponentConfig


class Dense(nn.Module):

    def __init__(self, config: LayerConfig):
        super().__init__()
        self.dense = gnn.Flex(gnn.Dense)(..., *[config.size]*config.depth, layer_norm=config.layer_norm,
                                         dropout=config.dropout)

    def forward(self, data):
        return self.dense(data)


class Base(NNGraph):

    def __init__(self):
        super().__init__()

    def init_data_feed(self):
        self.add_edge(lambda data: data.x, "node")
        self.add_edge(lambda data: data.e, "edge")
        self.add_edge(lambda data: data.g, "glob")

    def forward(self, data: GraphBatch):
        assert isinstance(data, GraphBatch)
        with self.run():
            x = self.propogate("node", data)
            e = self.propogate("edge", data)
            g = self.propogate("glob", data)
        return data.new_like(x, e, g)


class Encoder(Base):

    def __init__(self, config: NetComponentConfig):
        super().__init__()
        self.add_node(Dense(config.node), 'node')
        self.add_node(Dense(config.edge), 'edge')
        self.add_node(Dense(config.glob), 'glob')
        self.init_data_feed()


class Core(Base):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.add_node(Dense(config.core.node), 'node')
        self.add_node(Dense(config.core.edge), 'edge')
        self.add_node(Dense(config.core.glob), 'glob')
        self.init_data_feed()

        if config.connectivity.x0_to_edge:
            self.add_edge(lambda data: data.x, "edge", lambda data: data.edges[0])
        if config.connectivity.x1_to_edge:
            self.add_edge(lambda data: data.x, "edge", lambda data: data.edges[1])
        if config.connectivity.g_to_edge:
            self.add_edge(lambda data: data.g, "edge", lambda data: data.edge_idx)
        if config.connectivity.g_to_node:
            self.add_edge(lambda data: data.g, "node", lambda data: data.node_idx)

        if config.connectivity.edge_to_node:
            self.add_edge(
                "edge",
                "node",
                indexer=lambda data: data.edges[1],
                aggregation=gnn.Aggregator("add"),
                size=lambda data: data.x.shape[0],
            )
        if config.connectivity.edge_to_glob:
            self.add_edge(
                "edge",
                "glob",
                indexer=lambda data: data.edge_idx,
                aggregation=gnn.Aggregator("add"),
                size=lambda data: data.g.shape[0],
            )
        if config.connectivity.node_to_glob:
            self.add_edge(
                "node",
                "glob",
                indexer=lambda data: data.node_idx,
                aggregation=gnn.Aggregator("add"),
                size=lambda data: data.g.shape[0],
            )

class OutTransform(Base):
    def __init__(self, config: NetComponentConfig):
        super().__init__()
        self.add_node(
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., config.node.size),
                getattr(nn, config.node.activation)(),
            ),
            "node",
        )
        self.add_node(
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., config.edge.size),
                getattr(nn, config.edge.activation)(),
            ),
            "edge",
        )
        self.add_node(
            nn.Sequential(
                gnn.Flex(nn.Linear)(..., config.glob.size),
                getattr(nn, config.glob.activation)(),
            ),
            "glob",
        )
        self.init_data_feed()


class NNGraphNetwork(NNGraph):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.encoder = Encoder(config.encode)
        self.core = Core(config)
        self.decoder = Encoder(config.encode)
        self.out = OutTransform(config.out)

    def forward(self, data, steps, save_all=True):
        latent0 = self.encoder(data)
        data = latent0.clone()
        out_arr = []
        for _ in range(steps):
            data = GraphBatch.cat(latent0, data)
            data = self.core(data)
            data = self.decoder(data)
            out = self.out(data)
            if save_all:
                out_arr.append(out)
            else:
                out_arr = [out]
        return out_arr
