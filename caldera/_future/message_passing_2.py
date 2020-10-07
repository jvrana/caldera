# class ConnectionMod(nn.Module):
#
#     def __init__(self, src, dest):
#         super().__init__()
#         self.src = src
#         self.dest = dest
#
#
# class Adapter(nn.Module):
#
#     def __init__(self, mod: nn.Module, func: Callable):
#         super().__init__()
#         self.mod = mod
#         self.func = func
#
#     def forward(self, x):
#         return self.mod(self.func(x))
from typing import Union, Callable, Optional, List

from torch import nn
from caldera import gnn
import torch
from caldera.data import GraphBatch
# class Connection(nn.Module):
#
#     def __init__(self, src, dest, mapping: Callable):
#         self.src = src
#         self.dest

class MessagePassing(nn.Module):

    def __init__(self):
        super().__init__()
        self._connections = []

    def register_connection(self, src: Union[Callable, nn.Module],
                            dest: Union[Callable, nn.Module],
                            mapping: Optional[Union[nn.Module, Callable]] = None,
                            aggregation = None):
        # TODO: check modules are contained in this module
        self._connections.append((src, dest, mapping, aggregation))

    def _predecessor_connections(self, dest: Union[Callable, nn.Module]) -> List[Union[Callable, nn.Module]]:
        return [c for c in self._connections if c[1] is dest]

    def propogate(self, dest, data):
        connections = self._predecessor_connections(dest)
        if not connections:
            out = dest(data)
        else:
            results = []
            for src, _, mapping, aggregation in connections:
                result = self.propogate(src, data)
                if mapping:
                    result = result[mapping(data)]
                results.append(result)
            if aggregation:
                cat = torch.cat(results, 1)
                out = dest(cat, aggregation[0](data), dim=0, dim_size=aggregation[1](data))
            else:
                out = dest(torch.cat(results, 1))
        return out


class Foo(MessagePassing):

    def __init__(self):
        super().__init__()
        self.node = gnn.Flex(nn.Linear)(..., 1)
        self.edge = gnn.Flex(nn.Linear)(..., 1)
        self.register_connection(lambda data: data.x, self.node)
        self.register_connection(lambda data: data.e, self.edge)
        self.register_connection(self.node, self.edge, lambda data: data.edges[0])
        self.register_connection(lambda data: data.g, self.edge, lambda data: data.edge_idx)

    def forward(self, data):
        return self.propogate(self.edge, data)

foo = Foo()
out = foo(GraphBatch.random_batch(10, 5, 4, 3))
print(out)


class Foo2(MessagePassing):

    def __init__(self):
        super().__init__()
        self.node = gnn.Flex(nn.Linear)(..., 1)
        self.edge = gnn.Flex(nn.Linear)(..., 1)
        self.glob = gnn.Flex(nn.Linear)(..., 1)
        self.edge_to_node_agg = gnn.Aggregator('add')
        self.node_to_glob_agg = gnn.Aggregator('add')
        self.edge_to_glob_agg = gnn.Aggregator('add')

        self.register_connection(lambda data: data.x, self.node)
        self.register_connection(lambda data: data.e, self.edge)
        self.register_connection(lambda data: data.g, self.glob)

        self.register_connection(lambda data: data.x, self.edge, lambda data: data.edges[0])
        self.register_connection(lambda data: data.x, self.edge, lambda data: data.edges[1])
        self.register_connection(lambda data: data.g, self.edge, lambda data: data.edge_idx)

        self.register_connection(self.edge, self.edge_to_node_agg, None, (lambda data: data.edges[1], lambda data: data.x.size(0))),
        self.register_connection(self.edge_to_node_agg, self.node)

        self.register_connection(self.node, self.node_to_glob_agg, None, (lambda data: data.node_idx, lambda data: data.g.size(0)))
        self.register_connection(self.edge, self.edge_to_glob_agg, None, (lambda data: data.edge_idx, lambda data: data.g.size(0)))

    def forward(self, data):
        return self.propogate(self.node, data)

foo = Foo2()
out = foo(GraphBatch.random_batch(10, 5, 4, 3))
print(out)
