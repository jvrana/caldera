"""Adds the following to nx.Graph.

`get_global()`
"""
import functools

import networkx as nx

from caldera.defaults import CalderaDefaults
from caldera.utils.nx._global_accessor import GlobalAccess
from caldera.utils.nx._global_accessor import GraphWithGlobal


def add_global_access_to_nx(key):
    nx.Graph.get_global_key = functools.partialmethod(
        GraphWithGlobal.get_global_key, default_key=key
    )
    nx.Graph.get_global = GraphWithGlobal.get_global
    nx.Graph.set_global = GraphWithGlobal.set_global
    setattr(nx.Graph, key, GlobalAccess())


add_global_access_to_nx(CalderaDefaults.nx_global_key)
