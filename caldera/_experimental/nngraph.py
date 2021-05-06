"""nngraph.py.

Future API:

.. code-block::

    from torch import nn
    from caldera import gnn

    class Network(NNGraph):

        def __init__(self):
            super().__init__()
            self.x_layer = nn.Linear(8, 1)
            self.y_layer = nn.Linear(10, 1)
            self.z_layer = nn.Linear(5, 2)

            # this expect a TensorComposition instance or a dataclass
            self.data_bind(self.x_layer, DataCls, 'x')
            self.data_bind(self.y_layer, DataCls, 'y')

            # y_to_x is a index mapping where each entry in y_to_x indicates an x idx
            self.register_data_mapping('y', 'x', 'y_to_x')

            # optionally feed 'x' data into self.y_layer
            # this requires a mapping from y_to_x since 'y' is bound to y_layer
            self.feed('x', self.y_layer)

            # connect two layers, automatically using mappings and bound data
            # to perform the aggregation function
            self.connect(self.y_layer, self.x_layer, aggregation='add')
            self.connect(self.y_layer, self.z_layer, aggregation='add')
"""
from contextlib import contextmanager
from functools import wraps
from typing import Any
from typing import Callable
from typing import Generator
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import torch
from torch import nn

from caldera import gnn
from caldera.gnn.blocks.aggregator import AggregatorBase

_T = TypeVar("T")
_F = TypeVar("F", bound=Callable)

Module = Union[nn.Module, Callable]
Key = Union[str, Module]


class NNEdge(nn.Module):
    def __init__(
        self,
        src: Module,
        dest: Module,
        indexer: Optional[Module] = None,
        aggregation: Optional[gnn.Aggregator] = None,
        size: Optional[Module] = None,
    ):
        super().__init__()
        if aggregation or size:
            if not all(
                [aggregation is not None, size is not None, indexer is not None]
            ):
                raise ValueError(
                    "For aggregating edges, 'aggregation', 'size', and 'indexer' args must all be provided."
                )
            if aggregation:
                pass
                # if not issubclass(aggregation.__class__, AggregatorBase):
                #     raise TypeError(
                #         "aggregator must be a subclass of {} but found instance of {}".format(
                #             AggregatorBase.__name__, aggregation.__class__.__name__
                #         )
                #     )

        self.src = src
        self.dest = dest
        self.indexer = indexer
        self.aggregation: gnn.Aggregator = aggregation
        self.size = size

    # def __repr__(self):
    #     return "{}( {} -> {})".format(self.__class__.__name__, self.src, self.dest)
    # def forward(self, data: _T) -> _T:
    #     if self.aggregation:
    #         data = self.aggregation(data,
    #                                 self.indexer(data),
    #                                 dim=0,
    #                                 dim_size=self.size(data))
    #     elif self.indexer:
    #         idx = self.indexer(data)
    #         data = data[idx]
    #     return data


class FunctionModule(nn.Module):
    def __init__(self, f):
        if not callable(f):
            raise ValueError(
                "Function f must be callable. Found {}".format(f.__class__)
            )
        super().__init__()
        self.f = f

    def forward(self, data):
        return self.f(data)

    def __repr__(self):
        return "Callable({})".format(self.f)


# TODO: cache results
# TODO: nested NNGraph (NNHyperGraph)
# TODO: cached


def nngraph_forward(func: _F) -> _F:
    if hasattr(func, "__nngraph__"):

        @wraps(func)
        def forward(self, *args, **kwargs):
            return func(self, *args, **kwargs)

    else:

        @wraps(func)
        def forward(self, *args, **kwargs):
            with self.use_cache():
                result = func(self, *args, **kwargs)
                self.validate_visited()
            return result

    forward.__nngraph__ = True
    return forward


class NNGraphMeta(nn.Module):
    def __new__(typ, name, bases, namespace):
        newcls = super().__new__(typ, name, bases, namespace)
        # if 'forward' in namespace:
        #     @wraps(namespace['forward'])
        #     def forward(self, *args, **kwargs):
        #         self._cache = {}
        #         self._use_cache = True
        #
        #         result = namespace['forward'](self, *args, **kwargs)
        #
        #         self._cache = {}
        #         self._use_cache = False
        #
        #         missing_nodes = []
        #         for name, mod in self.nodes.items():
        #             if name not in self._cache:
        #                 missing_nodes.append(name)
        #         if missing_nodes:
        #             raise RuntimeError("The following nodes were not touched during forward propogation {}.".format(
        #                 missing_nodes
        #             ))
        #         return result

        # newcls.forward = forward
        return newcls


# TODO: how to handle cache??? properly
class NNGraph(nn.Module):
    def __init__(self, reducer: Optional[Module] = None):
        super().__init__()
        self.add_graph_forward_hook()
        self.nodes = nn.ModuleDict()
        self._cache = {}
        self._use_cache = False
        self.edges = nn.ModuleList()
        self.reducer = reducer or self.__class__.reducer

    def add_graph_forward_hook(self):
        nngraph_hook = "__nngraph_hook__"
        func = self.forward
        if hasattr(func, nngraph_hook):

            @wraps(func)
            def forward(_self, *args, **kwargs):
                return func(_self, *args, **kwargs)

        else:

            @wraps(func)
            def forward(_self, *args, **kwargs):
                with _self.use_cache():
                    result = func(_self, *args, **kwargs)
                    _self.validate_visited()
                return result

        setattr(forward, nngraph_hook, True)
        return forward

    def validate_visited(self):
        missing = []
        for n in self.nodes:
            if n not in self._cache:
                missing.append(n)
        if missing:
            raise RuntimeError(
                "Some nodes were not visited during forward propogation:\n{}".format(
                    missing
                )
            )

    @classmethod
    def reducer(cls, data_list):
        """Default reducer function for multiple incoming edges."""
        return torch.cat(data_list, dim=1)

    def set_default_reducer(self, func):
        self.reducer = func

    @staticmethod
    def _node_hash(mod):
        return "{cls}_{hsh}".format(cls=mod.__class__.__name__, hsh=hash(mod))

    def _get_node_by_module(self, mod: Module) -> Union[Tuple[str, Module], None]:
        for k, v in self.nodes.items():
            if v is mod:
                return (k, v)

    def get_node(self, item: Key) -> Union[Tuple[str, Module], None]:
        if isinstance(item, str):
            if item in self.nodes:
                return (item, self.nodes[item])
            else:
                return None
        return self._get_node_by_module(item)

    # @staticmethod
    # def is_callable(mod):
    #     if isinstance(mod, nn.Module):
    #         return True
    #     elif callable(mod):
    #         return True
    #     return False

    def add_node(self, mod, name: Optional[Union[str, Callable[[Key], str]]] = None):
        if name is None:
            name = self._node_hash(mod)
        if callable(name):
            name = name(mod)
        if isinstance(mod, nn.Module):
            x = self.get_node(mod)
            if x is not None:
                return x
            else:
                if name in self.nodes and self.nodes[name] is not mod:
                    raise ValueError("Node '{}' already exists".format(name))
                self.nodes[name] = mod
                return name, mod
        else:
            mod = FunctionModule(mod)
            self.nodes[name] = mod
            return name, mod

    def add_edge(
        self,
        node1: Key,
        node2: Key,
        indexer: Optional[Module] = None,
        size: Optional[Module] = None,
        aggregation: Optional[AggregatorBase] = None,
    ) -> None:
        # TODO: edges should either be a simple callable, apply selection or apply aggregation
        if self._potential_cycle(node1, node2):
            raise ValueError("Cannot add edge because this would create a cycle.")
        if isinstance(node1, str):
            node1, mod1 = self.get_node(node1)
        else:
            node1, mod1 = self.add_node(node1)
        if isinstance(node2, str):
            node2, mod2 = self.get_node(node2)
        else:
            node2, mod2 = self.add_node(node2)
        if indexer and not isinstance(indexer, nn.Module):
            indexer = FunctionModule(indexer)
        if size and not isinstance(size, nn.Module):
            size = FunctionModule(size)
        self.edges.append(
            NNEdge(node1, node2, indexer=indexer, size=size, aggregation=aggregation)
        )

    def _potential_cycle(self, node1: Key, node2: Key) -> bool:
        """Check for a potential cycle if edge between node1 and node2 were
        added.

        :param node1:
        :param node2:
        :return:
        """
        if self.get_node(node2):
            visited = self.bfs(node2)
            if self._node_hash(node1) in visited:
                return True
        return False

    def bfs(self, source: Key, visited: Optional[List[Module]] = None) -> List[Module]:
        """Breadth-first search."""
        visited = visited or list()
        if source in visited:
            return []
        else:
            _, source = self.get_node(source)
            visited.append(source)
        for edge in self.successor_edges(source):
            self.bfs(edge.dest, visited)
        return visited

    def call_node(self, item: Key, data: _T, trace=None) -> Any:
        node, mod = self.get_node(item)
        try:
            if self._use_cache and node in self._cache:
                out = self._cache[node]
            else:
                out = mod(data)
                self._cache[node] = out
            return out
        except RuntimeError as e:
            msg = "An error occurred during `{}.propogate`\n".format(
                self.__class__.__name__
            )
            msg += "Node: '{}'\n".format(node)
            msg += "Module: {}\n".format("\n".join(str(mod).splitlines()[:4]))
            if trace:
                msg += "Trace: {}\n".format(trace)
            msg += "Original exception: "
            msg += str(e)
            raise RuntimeError(msg) from e

    def propogate(self, item, data):
        self._require_ctx()
        name, dest_mod = self.get_node(item)
        incoming_edges = list(self.predecessor_edges(dest_mod))
        if not incoming_edges:
            return self.call_node(dest_mod, data)
        else:
            collected = []
            for edge in incoming_edges:
                _, src = self.get_node(edge.src)
                propogated = self.propogate(src, data)
                if edge.aggregation:
                    indices = edge.indexer(data)
                    size = edge.size(data)
                    propogated = edge.aggregation(
                        propogated, indices=indices, dim=0, dim_size=size
                    )
                elif edge.indexer:
                    idx = edge.indexer(data)
                    propogated = propogated[idx]
                collected.append(propogated)
            cat = self.reducer(tuple(collected))
            trace = [edge.src for edge in incoming_edges]
            return self.call_node(dest_mod, cat, trace=trace)

    def predecessor_edges(self, item: Key) -> Generator[NNEdge, None, None]:
        node, _ = self.get_node(item)
        for edge in self.edges:
            if edge.dest == node:
                yield edge

    def successor_edges(self, item: Key) -> Generator[NNEdge, None, None]:
        node, _ = self.get_node(item)
        for edge in self.edges:
            if edge.src == node:
                yield edge

    def _require_ctx(self):
        if not self._use_cache:
            msg = (
                "Cannot run `{cls}.forward` outside of `{cls}.{ctx}` context. Instead, please use:\n"
                "```\n"
                "   with graph.{ctx}():\n"
                "       graph.forward(mod, data)\n"
                "```"
            ).format(cls=self.__class__.__name__, ctx=self.run.__name__)
            raise RuntimeError(msg)

    def begin_run(self):
        self._cache = {}
        self._use_cache = True

    def end_run(self):
        self._cache = {}
        self._use_cache = False

    @contextmanager
    def run(self):
        self.begin_run()
        yield
        self.end_run()

    def forward(self, mod_or_name: Key, data: _T) -> Any:
        self._require_ctx()
        return self.propogate(mod_or_name, data)

    # TODO: somehow enforce resetting of cache


# class NNGraphAdv(NNGraph):
#
#     def __init__(self, reducer=None):
#         super().__init__(reducer=reducer)
#         self._data_bindings = {}
#         self._data_mappings = {}
#
#     def data_bind(self, item: Key, data_class: Type, attribute: str):
#
#         def create_bind(attribute, data_class):
#             def bind(data):
#                 if not isinstance(data, data_class):
#                     raise TypeError("Expected a {}".format(data_class))
#                 return getattr(bind, attribute)
#             bind.__name__ = 'get_{}_{}'.format(data_class, attribute)
#         bind = create_bind(attribute, data_class)
#
#         node, _ = self.get_node(item)
#
#         self._data_bindings[node] = (data_class, attribute)
#         self.add_edge(bind, item)
#
#     def register_data_mapping(self, data_class, x, y, xymap):
#         self._data_mappings.setdefault(data_class, dict())
#         self._data_mapping[data_class].setdefault(x, dict())
#         self._data_mapping[data_class][x][y] = xymap
#
#     def feed(self, key, item):
#         node, _ = self.get_node(item)
#         bindings = self._data_bindings[node]
#         maps = []
#         for bind in bindings:
#             data_class, attribute = bind
#             maps.append(self._data_mappings[data_class][attribute][key])
#         self.add_edge(lambda data: getattr(data, key), node, indexer=lambda data: )
#
#
