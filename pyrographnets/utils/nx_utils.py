import networkx as nx
from typing import Generator, Hashable


def iter_roots(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.predecessors(n)):
            yield n


def iter_leaves(g: nx.DiGraph) -> Generator[Hashable, None, None]:
    for n in g.nodes():
        if not list(g.successors(n)):
            yield n
