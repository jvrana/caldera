"""Adds the following to nx.Graph.

`get_global()`
"""
import functools

import networkx as nx

from caldera.utils.nx._global_accessor import GlobalAccess
from caldera.utils.nx._global_accessor import GraphWithGlobal


def add_global_access_to_nx(key):
    nx.Graph.get_global_key = functools.partialmethod(
        GraphWithGlobal.get_global_key, default_key=key
    )
    nx.Graph.get_global = GraphWithGlobal.get_global
    nx.Graph.set_global = GraphWithGlobal.set_global
    nx.Graph.globals = GraphWithGlobal.globals
    setattr(nx.Graph, key, GlobalAccess())
