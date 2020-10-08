from torch import nn
from caldera import gnn
from typing import Union, Callable, Optional, Hashable, Tuple
from typing import TypeVar
from caldera.data import GraphBatch
import torch
from functools import wraps


_T = TypeVar('T')
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

class NNGraphMeta(type):

    def __new__(typ, clsname, superclasses, attributedict):

        newcls =  super().__new__(typ, clsname, superclasses, attributedict)

        if 'forward' in attributedict:
            @wraps(attributedict['forward'])
            def forward(self, *args, **kwargs):
                self._cache = {}
                self._use_cache = True
                result = attributedict['forward'](self, *args, **kwargs)
                self._use_cache = False
                self._cache = {}
                return result

            newcls.forward = forward
        return newcls

# TODO: how to handle cache??? properly
class NNGraph(nn.Module, metaclass=NNGraphMeta):

    def __init__(self, reducer: Optional[Union[Callable, nn.Module]] = None):
        super().__init__()
        self.nodes = nn.ModuleDict()
        self._cache = {}
        self._use_cache = False
        self.edges = nn.ModuleList()
        self.reducer = reducer or self.__class__.reducer

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

    # TODO: somehow enforce resetting of cache


#
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