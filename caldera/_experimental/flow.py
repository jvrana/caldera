"""Modules for create Caldera flow neural networks.

Flow neural networks are arbitrarily connected sub neural networks.
"""
import ast
import inspect
import uuid
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn

from caldera import gnn
from caldera.data import GraphBatch


# TODO: raise error if there are connections that have not been touched in the forward propogation
# TODO: create a simple `propagate` function that detects leaves and automatically applies data
# TODO: check gradient propagation
# TODO: allow module dict and keys to be used...
# TODO: draw connections using daft
# TODO: make connection first class object
# TODO: improve __str__ and __repr__ of modules
def func_repr(func, len_limit=30):
    lines = inspect.getsource(func).splitlines()
    if len(lines) > 1:
        return func.__name__
    else:
        line = lines[0].strip().split(",")[0].strip()
        if len(line) > len_limit:
            return line[:len_limit] + "..."
        else:
            return line


class Connection:
    def __init__(
        self, src, dest, mapping=None, aggregation=None, name=None, parent_modules=None
    ):
        super().__init__()
        self.src = src
        self.dest = dest
        self.mapping = mapping
        self.aggregation = aggregation
        self.name = name
        self._parent_modules = parent_modules

    def __repr__(self):
        if self.mapping:
            on = "map(" + str(self.mapping.__class__.__name__) + ")"
        elif self.aggregation:
            on = str(self.dest)
        else:
            on = ""

        src, dest = "", ""
        if self._parent_modules:
            for k, v in self._parent_modules.items():
                if v is self.src:
                    src = "(" + k + ")" + " " + src
                if v is self.dest:
                    dest = "(" + k + ")" + " " + dest

        if not src and inspect.isfunction(self.src):
            src = func_repr(self.src)

        if not dest and inspect.isfunction(self.dest):
            dest = func_repr(self.dest)

        if not src:
            src = self.src.__class__.__name__
        if not dest:
            dest = self.dest.__class__.__name__
        return "{c} ( {src} -[{on}]-> {dest} )".format(
            c=self.__class__.__name__, src=src, dest=dest, on=on
        )


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self._connections = OrderedDict()
        self._cached: Dict[str, torch.Tensor] = {}

    def register_connection(
        self,
        src: Union[Callable, nn.Module],
        dest: Union[Callable, nn.Module],
        mapping: Optional[Union[nn.Module, Callable]] = None,
        aggregation: Optional[Tuple[Callable, Callable]] = None,
        name: Optional[str] = None,
    ):
        if name is None:
            name = str(uuid.uuid4())[-5:]
        self._connections[name] = Connection(
            src, dest, mapping, aggregation, parent_modules=dict(self._modules)
        )

    def register_feed(
        self, src: Union[Callable, nn.Module], dest: Union[Callable, nn.Module]
    ):
        self.register_connection(src, dest)

    def register_map(
        self,
        src: Union[Callable, nn.Module],
        dest: Union[Callable, nn.Module],
        mapping: Optional[Union[nn.Module, Callable]] = None,
    ):
        self.register_connection(src, dest, mapping)

    def register_aggregation(
        self,
        src: Union[Callable, nn.Module],
        agg: gnn.Aggregator,
        dest: Union[Callable, nn.Module],
        index: Union[Callable, nn.Module],
        out_size: Union[Callable, nn.Module],
    ):
        if not issubclass(agg.__class__, gnn.Aggregator):
            raise TypeError(
                "Aggregator must be an instance or subclass of {}, but found {}".format(
                    gnn.Aggregator, agg.__class__
                )
            )
        self.register_connection(src, agg, None, (index, out_size))
        self.register_connection(agg, dest)

    def _predecessor_connections(
        self, dest: Union[Callable, nn.Module]
    ) -> List[Union[Callable, nn.Module]]:
        # print(self._connections)
        return {n: c for n, c in self._connections.items() if c.dest is dest}

    # TODO: Here, we want to only pass data through the layers *a single time*
    #       The way it is currently implemented, this may happen multiple times.
    # TODO: in what way to 'cache' the results
    def apply(self, mod, data, *args, **kwargs):
        if mod not in self._cached:
            self._cached[mod] = mod(data, *args, **kwargs)
        return self._cached[mod]

    def _propogate(self, dest: nn.Module, data: Any) -> torch.Tensor:
        connections = self._predecessor_connections(dest)
        if not connections:
            out = self.apply(dest, data)
        else:
            results = []
            for name, conn in connections.items():
                src = conn.src
                mapping = conn.mapping
                aggregation = conn.aggregation
                result = self._propogate(src, data)
                if mapping:
                    result = result[mapping(data)]
                results.append(result)
            cat = torch.cat(results, 1)
            if aggregation:
                cat = torch.cat(results, 1)
                out = self.apply(
                    dest,
                    cat,
                    aggregation[0](data),
                    dim=0,
                    dim_size=aggregation[1](data),
                )
            else:
                out = self.apply(dest, cat)
        return out

    def propogate(self, dest: nn.Module, data: Any) -> torch.Tensor:
        return self._propogate(dest, data)

    def forward(self):
        self._cached = {}


class FlowExample(Flow):
    def __init__(self):
        super().__init__()
        self.node = gnn.Flex(gnn.Dense)(..., 10, 1, layer_norm=True)
        self.edge = gnn.Flex(gnn.Dense)(..., 10, 1)
        self.glob = gnn.Flex(gnn.Dense)(..., 10, 10, 1)
        self.edge_to_node_agg = gnn.Flex(gnn.MultiAggregator)(..., ["add"])
        self.node_to_glob_agg = gnn.Flex(gnn.MultiAggregator)(..., ["add"])
        self.edge_to_glob_agg = gnn.Flex(gnn.MultiAggregator)(..., ["add"])

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
        self.register_connection(
            lambda data: data.g, self.node, lambda data: data.node_idx
        )
        self.register_connection(
            lambda data: data.x, self.edge, lambda data: data.edges[0]
        )
        self.register_connection(
            lambda data: data.x, self.edge, lambda data: data.edges[1]
        )
        self.register_connection(
            lambda data: data.g, self.edge, lambda data: data.edge_idx
        )

        # TODO: better connection API
        # these connections use an index and aggregation function (e.g. scatter_add) to make the connection
        self.register_connection(
            self.edge,
            self.edge_to_node_agg,
            None,
            (lambda data: data.edges[1], lambda data: data.x.size(0)),
        ),
        self.register_connection(self.edge_to_node_agg, self.node)

        self.register_connection(
            self.node,
            self.node_to_glob_agg,
            None,
            (lambda data: data.node_idx, lambda data: data.g.size(0)),
        )
        self.register_connection(self.node_to_glob_agg, self.glob)

        self.register_connection(
            self.edge,
            self.edge_to_glob_agg,
            None,
            (lambda data: data.edge_idx, lambda data: data.g.size(0)),
        )
        self.register_connection(self.edge_to_glob_agg, self.glob)

    def forward(self, data):
        super().forward()  # TODO: enforce call to super using metaclass...
        x = self._propogate(self.edge, data)
        e = self._propogate(self.node, data)
        g = self._propogate(self.glob, data)
        return x, e, g


foo = FlowExample()
out = foo(GraphBatch.random_batch(1000, 5, 4, 3))

print(foo)

print(sum([param.nelement() for param in foo.parameters()]))
