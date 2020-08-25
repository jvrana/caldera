import networkx as nx
import torch

from .contexts import _context_manager_test_cases
from .contexts import pytest_contexts


def nx_random_features(g: nx.DiGraph, n_feat: int, e_feat: int, g_feat: int):
    for _, ndata in g.nodes(data=True):
        ndata["features"] = torch.randn(n_feat)
    for _, _, edata in g.edges(data=True):
        edata["features"] = torch.randn(e_feat)
    g.data = {"features": torch.randn(g_feat)}
    return g
