from torch import nn
from caldera import gnn
import torch


class MessagePassing(nn.Module):

    def __init__(self):
        super().__init__()
        self._connections = []
        self.lin1 = None
        self.lin2 = None
        self.register_connection(
            lambda x: x.x,
            self.lin2,
            mapping=lambda x: x.idx
        )
        self.register_connection(
            lambda x: x.x,
            self.lin2
        )

    def register_connection(self, source, destination, mapping):
        self._connections.append((source, destination, mapping))

    def propogate(self, mod, x):
        out = []
        for c in self._connections:
            if mod is c[1]:
                y = c[0](x)
                z = y[c[2](x)]
                out.append(z)
        return mod(torch.cat(out, 1))

    def forward(self, x):
        self.propogate(x)


MessagePassing()


class GraphCore(MessagePassing):

    def __init__(self):
        self.connection(lambda d: self.net1(d.x), self.net2, lambda x: x.node_idx)
        self.connection(lambda d: self.net2(d.e), self.net2, lambda x: x.edge_idx)

    def forward(self, x):
        self.propogate(self.net2)(x)