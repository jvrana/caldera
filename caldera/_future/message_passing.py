# flake8: noqa

from torch import nn
from caldera import gnn
import torch
from collections import OrderedDict
from caldera.data import GraphBatch
from typing import Mapping, Callable
from dataclasses import dataclass



# @dataclass
# class Connection(object):
#     src: nn.Module
#     dest: nn.Module
#     src_map: Callable
#     dest_map: Callable
#     idx_map: Callable
#
#     def __str__(self):
#         return "Conection"
#
#     def __repr__(self):
#         return "Connection"


class ConnectionMapping(object):

    def __init__(self):
        self.connections = []

    def add(self, connection: Connection):
        self.connections.append(connection)

    def successors(self, src):
        carr = []
        for c in self.connections:
            if src is c.src:
                carr.append(c)
        return carr

    def predecessors(self, dest):
        carr = []
        for c in self.connections:
            if dest is c.dest:
                carr.append(c)
        return carr
    nn.
self.node = Adapter(lin, lambda data: data.x)
self.emit(self.node, 'node')

class GraphCore(nn.Module):

    def __init__(self):
        super().__init__()
        self.node = gnn.Flex(nn.Linear)(..., 1)
        self.edge = gnn.Flex(nn.Linear)(..., 1)

        self.connections = ConnectionMapping()
        self.connections.add(
            Connection(
                src=self.node,
                dest=self.edge,
                src_map=lambda data: data.x,
                dest_map=lambda data: data.e,
                idx_map=lambda data: data.edges[0]
            )
        )
        self.connections.add(
            Connection(
                lambda data: data.g,
                self.edge,
                lambda data: data,
                lambda data: data.e,
                lambda data: data.edge_idx
            )
        )

    def propogate(self, mod, x):
        messages = []
        for c in self.connections.predecessors(mod):
            # emit message
            src_out = c.src(c.src_map(x))

            # receive message
            # pool
            src_to_dest_map = c.idx_map(x)
            msg = src_out[src_to_dest_map]
            messages.append(msg)

        # collect and apply
        cat = torch.cat([c.dest_map(x)] + messages, 1)
        return c.dest(cat)

    def forward(self, data):
        return self.propogate(self.edge, data)


b = GraphBatch.random_batch(10, 5, 10, 5)
core = GraphCore()
out = core(b)
print(out)

return self.mod(self.func(x))


class Foo(nn.Module):

    def __init__(self):
        super().__init__()
        self.node = gnn.Flex(nn.Linear)(..., 1)
        self.edge = gnn.Flex(nn.Linear)(..., 1)
        self.connections = [
            (lambda data: data.x, self.node),
            (lambda data: data.e, self.edge),
            (self.node, self.edge, lambda data: data.edges[0])
        ]

    def propogate(self, mod, data):
        print("propogating " + str(mod.__class__.__name__))
        connections = [c for c in self.connections if c[1] is mod]
        if not connections:
            out = mod(data)

            return out
        else:
            results = []
            for c in connections:
                a = self.propogate(c[0], data)
                if len(c) == 3:
                    a = a[c[2](data)]
                results.append(a)
            b = c[1](torch.cat(results, 1))
            return b

    #                 if len(c) == 2:
    #                     out = c[1](c[0](data))
    #                     resutls.append(out)
    #                 out = c[0](data)
    #                 print(out)

    def forward(self, data):
        return self.propogate(self.edge, data)
