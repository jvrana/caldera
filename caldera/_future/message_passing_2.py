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
import uuid
# class Connection(nn.Module):
#
#     def __init__(self, src, dest, mapping: Callable):
#         self.src = src
#         self.dest

# TODO: draw connections
class Flow(nn.Module):

    def __init__(self):
        super().__init__()
        self._connections = {}
        self._cached = {}

    def register_connection(self, src: Union[Callable, nn.Module],
                            dest: Union[Callable, nn.Module],
                            mapping: Optional[Union[nn.Module, Callable]] = None,
                            aggregation = None,
                            name = None):
        # TODO: check modules are contained in this module
        if name is None:
            name = str(uuid.uuid4())[-5:]
        self._connections[name] = (src, dest, mapping, aggregation)

    def _predecessor_connections(self, dest: Union[Callable, nn.Module]) -> List[Union[Callable, nn.Module]]:
        return {n: c for n, c in self._connections.items() if c[1] is dest}

    # TODO: Here, we want to only pass data through the layers *a single time*
    #       The way it is currently implemented, this may happen multiple times.
    # TODO: in what way to 'cache' the results
    def apply(self, mod, data, *args, **kwargs):
        if mod not in self._cached:
            self._cached[mod] = mod(data, *args, **kwargs)
        else:
            print("using cached")
        return self._cached[mod]

    def propogate(self, dest, data):
        connections = self._predecessor_connections(dest)
        if not connections:
            out = self.apply(dest, data)
        else:
            results = []
            for name, (src, _, mapping, aggregation) in connections.items():
                print("connection: {}".format(name))
                result = self.propogate(src, data)
                if mapping:
                    result = result[mapping(data)]
                results.append(result)
            cat = torch.cat(results, 1)
            if aggregation:
                cat = torch.cat(results, 1)
                out = self.apply(dest, cat, aggregation[0](data), dim=0, dim_size=aggregation[1](data))
            else:
                out = self.apply(dest, cat)
        return out

    def forward(self):
        self._cached = {}


class Foo(Flow):

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

# foo = Foo()
# out = foo(GraphBatch.random_batch(10, 5, 4, 3))
# # print(out)

# TODO: raise error if there are connections that have not been touched in the forward propogation
# TODO: create a simple `propogate` function that detects leaves and automatically applies data
# TODO: check gradient propogation
# TODO: allow module dict and keys to be used...
# TODO: draw connections using daft
class Foo2(Flow):

    def __init__(self):
        super().__init__()
        self.node = gnn.Flex(nn.Linear)(..., 1)
        self.edge = gnn.Flex(nn.Linear)(..., 1)
        self.glob = gnn.Flex(nn.Linear)(..., 1)
        self.edge_to_node_agg = gnn.Aggregator('add')
        self.node_to_glob_agg = gnn.Aggregator('add')
        self.edge_to_glob_agg = gnn.Aggregator('add')

        # the following connections determines how modules interact
        # if we did not add *any* connections, this would be a simple graph encoder
        # we can pick and choose which connections we add as well as add as many arbitrary
        # connections (and meanings behind the data) as we want, provided
        # we maintain appropriate indices among the data.
        # For example, the `node_idx` maps node indices to global indices, and so
        # we can use that as a connection from data.g to data.x
        # --OR-- in the reverse direction, from data.x to data.g using an aggregation function

        # these connections extract relevant tensors from the data
        self.register_connection(lambda data: data.x, self.node)
        self.register_connection(lambda data: data.e, self.edge)
        self.register_connection(lambda data: data.g, self.glob)

        # these connections use indices
        self.register_connection(lambda data: data.g, self.node, lambda data: data.node_idx)
        self.register_connection(lambda data: data.x, self.edge, lambda data: data.edges[0])
        self.register_connection(lambda data: data.x, self.edge, lambda data: data.edges[1])
        self.register_connection(lambda data: data.g, self.edge, lambda data: data.edge_idx)

        # TODO: better connection API
        # these connections use an index and aggregation function (e.g. scatter_add) to make the connection
        self.register_connection(self.edge, self.edge_to_node_agg, None, (lambda data: data.edges[1], lambda data: data.x.size(0))),
        self.register_connection(self.edge_to_node_agg, self.node)

        self.register_connection(self.node, self.node_to_glob_agg, None, (lambda data: data.node_idx, lambda data: data.g.size(0)))
        self.register_connection(self.node_to_glob_agg, self.glob)

        self.register_connection(self.edge, self.edge_to_glob_agg, None, (lambda data: data.edge_idx, lambda data: data.g.size(0)))
        self.register_connection(self.edge_to_glob_agg, self.glob)

    def forward(self, data):
        super().forward()  # TODO: enforce call to super using metaclass...
        x = self.propogate(self.edge, data)
        e = self.propogate(self.node, data)
        g = self.propogate(self.glob, data)
        return x, e, g

foo = Foo2()
out = foo(GraphBatch.random_batch(1000, 5, 4, 3))
# print(out)
