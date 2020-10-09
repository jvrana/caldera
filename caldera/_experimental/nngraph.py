from torch import nn
from caldera import gnn
from typing import Union, Callable, Optional, Hashable, Tuple
from typing import TypeVar
from caldera.data import GraphBatch
import torch
from functools import wraps
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager

_T = TypeVar('T')
_F = TypeVar('F', bound=Callable)

Module = Union[nn.Module, Callable]


class NNEdge(nn.Module):

    def __init__(self, src: Module,
                 dest: Module,
                 indexer: Optional[Module]=None,
                 aggregation: Optional[gnn.Aggregator] = None,
                 size: Optional[Module] = None,):
        super().__init__()
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
            raise ValueError("Function f must be callable. Found {}".format(
                f.__class__
            ))
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
    if hasattr(func, '__nngraph__'):
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

class NNGraphMeta(ABCMeta):

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
class NNGraphABC(nn.Module, metaclass=NNGraphMeta):

    def __init__(self, reducer: Optional[Union[Callable, nn.Module]] = None):
        super().__init__()
        self.add_graph_forward_hook()
        self.nodes = nn.ModuleDict()
        self._cache = {}
        self._use_cache = False
        self.edges = nn.ModuleList()
        self.reducer = reducer or self.__class__.reducer

    def add_graph_forward_hook(self):
        nngraph_hook = '__nngraph_hook__'
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

    def use_cache(self):
        self._cache = {}
        self._use_cache = True
        yield
        self._cache = {}
        self._use_cache = False

    def validate_visited(self):
        missing = []
        for n in self.nodes:
            if n not in self._cache:
                missing.append(n)
        if missing:
            raise RuntimeError("Some nodes were not visited during forward propogation:\n{}".format(
                missing
            ))

    @classmethod
    def reducer(cls, data_list):
        """Default reducer function for multiple incoming edges"""
        return torch.cat(data_list, dim=1)

    def set_default_reducer(self, func):
        self.reducer = func

    @staticmethod
    def _node_hash(mod):
        return '{cls}_{hsh}'.format(cls=mod.__class__.__name__, hsh=hash(mod))

    def _get_node_by_module(self, mod):
        for k, v in self.nodes.items():
            if v is mod:
                return (k, v)

    def get_node(self, item: Union[str, Callable, nn.Module, Hashable]) -> \
            Union[Tuple[str, Union[Callable, nn.Module]], None]:
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

    def add_node(self, mod, name: Optional[Union[Callable[[Union[Callable, nn.Module], str], str]]]=None):
        if name is None:
            name = self._node_hash(mod)
        if callable(name):
            name = name(mod)
        if isinstance(mod, nn.Module):
            if name in self.nodes and self.nodes[name] is not mod:
                raise ValueError("Node '{}' already exists".format(name))
            self.nodes[name] = mod
            return name, mod
        else:
            mod = FunctionModule(mod)
            self.nodes[name] = mod
            return name, mod

    def add_edge(self, node1, node2, indexer=None, size=None, aggregation=None):
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
            NNEdge(
                node1, node2,
                indexer=indexer,
                size=size,
                aggregation=aggregation
            )
        )

    def _potential_cycle(self, node1, node2):
        if self.get_node(node2):
            visited = self.bfs(node2)
            if self._node_hash(node1) in visited:
                return True
        return False

    def bfs(self, source, visited=None):
        visited = visited or list()
        if source in visited:
            return []
        else:
            visited.append(source)
        for edge in self.successors(source):
            self.bfs(edge.dest, visited)
        return visited

    def mod_fwd(self, mod, data):
        node, mod = self.get_node(mod)
        try:
            if self._use_cache and node in self._cache:
                out = self._cache[node]
            else:
                out = mod(data)
                self._cache[node] = out
            return out
        except RuntimeError as e:
            msg = "An error occurred during `{}.propogate`\n".format(self.__class__.__name__)
            msg += "Node: '{}'\n".format(node)
            msg += "Module: {}\n".format('\n'.join(str(mod).splitlines()[:4]))
            msg += "Original exception: "
            msg += str(e)
            raise RuntimeError(msg) from e

    def propogate(self, node2, data):
        node2, dest = self.get_node(node2)
        incoming_edges = list(self.predecessors(dest))
        if not incoming_edges:
            return self.mod_fwd(dest, data)
        else:
            collected = []
            for edge in incoming_edges:
                _, src = self.get_node(edge.src)
                propogated = self.propogate(src, data)
                if edge.aggregation:
                    indices = edge.indexer(data)
                    size = edge.size(data)
                    propogated = edge.aggregation(propogated,
                                                  indices=indices,
                                                  dim=0,
                                                  dim_size=size)
                elif edge.indexer:
                    idx = edge.indexer(data)
                    propogated = propogated[idx]
                collected.append(propogated)
            cat = self.reducer(tuple(collected))
            return self.mod_fwd(dest, cat)

    def predecessors(self, item):
        node, _ = self.get_node(item)
        for edge in self.edges:
            if edge.dest == node:
                yield edge

    def successors(self, item):
        node, _ = self.get_node(item)
        for edge in self.edges:
            if edge.src == node:
                yield edge

    @abstractmethod
    def forward(self):
        pass

    # TODO: somehow enforce resetting of cache


# graph = NNGraph()
# node = nn.Linear(8, 1)
# edge = nn.Linear(8, 1)
# glob = nn.Linear(8, 1)
# graph.add_edge(lambda data: data.x, node)
# graph.add_edge(lambda data: data.e, edge)
# graph.add_edge(node, edge, indexer=lambda data: data.edges[0])
# graph.add_edge(node, edge, indexer=lambda data: data.edges[1])
# graph.add_edge(edge, glob, aggregation=gnn.Aggregator('add'), indexer=lambda data: data.edge_idx, size=lambda data: data.g.shape[0])
#
#
# graph.propogate(nod)