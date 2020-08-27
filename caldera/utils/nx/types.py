from typing import Union

import networkx as nx

Graph = Union[
    nx.DiGraph,
    nx.Graph,
    nx.MultiGraph,
    nx.MultiDiGraph,
    nx.OrderedDiGraph,
    nx.OrderedGraph,
]
DirectedGraph = Union[
    nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph, nx.OrderedMultiDiGraph
]
UndirectedGraph = Union[nx.Graph, nx.MultiGraph, nx.OrderedGraph, nx.OrderedMultiGraph]
