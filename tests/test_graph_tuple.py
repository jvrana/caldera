from typing import *
import numpy as np
from pyro_graph_nets.utils.data import random_input_output_graphs
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple, validate_gt
import torch
import networkx as nx
import pytest

def graph_generator(
        n_nodes: Tuple[int, int],
        n_features: Tuple[int, int],
        e_features: Tuple[int, int],
        g_features: Tuple[int, int]
):
    gen = random_input_output_graphs(
        lambda: np.random.randint(*n_nodes),
        20,
        lambda: np.random.uniform(1, 10, n_features[0]),
        lambda: np.random.uniform(1, 10, e_features[0]),
        lambda: np.random.uniform(1, 10, g_features[0]),
        lambda: np.random.uniform(1, 10, n_features[1]),
        lambda: np.random.uniform(1, 10, e_features[1]),
        lambda: np.random.uniform(1, 10, g_features[1]),
        input_attr_name='features',
        target_attr_name='target',
        do_copy=False
    )
    return gen

class TestToGraphTuple(object):

    @pytest.mark.parametrize('node_name', [1, 10, 'node10'])
    def test_valid_single_graph_tuple(self, node_name):
        """Simple test for to_graph_tuple"""
        g = nx.DiGraph()
        g.add_edge(0, node_name)
        g.add_edge(node_name, 2)
        for _, ndata in g.nodes(data=True):
            ndata['features'] = torch.zeros(3)

        for _, _, edata in g.edges(data=True):
            edata['features'] = torch.ones(4)

        g.data = {
            'features': torch.tensor([0., 10.])
        }

        gt = to_graph_tuple([g])

        assert torch.all(torch.eq(gt.node_attr, torch.tensor([0, 0, 0])))
        assert torch.all(torch.eq(gt.edge_attr, torch.tensor([1, 1, 1, 1])))
        assert torch.all(torch.eq(gt.global_attr, torch.tensor([0, 10])))

        assert torch.all(torch.eq(gt.edges, torch.tensor([
            [0, 1],
            [1, 2]
        ])))

        assert torch.all(torch.eq(gt.node_indices, torch.tensor([0, 0, 0])))
        assert torch.all(torch.eq(gt.edge_indices, torch.tensor([0, 0])))

    def test_two_graphs(self):
        graphs = []

        for i in range(2):
            g = nx.DiGraph()
            g.add_edge(0, 1)
            g.add_edge(1, 2)
            for _, ndata in g.nodes(data=True):
                ndata['features'] = torch.ones(3) * i

            for _, _, edata in g.edges(data=True):
                edata['features'] = torch.ones(4) * i + 10

            g.data = {
                'features': torch.ones(2) * i + 100
            }
            graphs.append(g)

        gt = to_graph_tuple(graphs)

        # check attribute shapes
        assert gt.node_attr.shape == (6, 3)
        assert gt.edge_attr.shape == (4, 4)
        assert gt.global_attr.shape == (2, 2)

        assert torch.all(torch.eq(gt.node_attr, torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])))
        assert torch.all(torch.eq(gt.edge_attr, torch.tensor([
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [11, 11, 11, 11],
            [11, 11, 11, 11]
        ])))
        assert torch.all(torch.eq(gt.global_attr, torch.tensor([
            [100, 100],
            [101, 101]
        ])))

        # check edges do not conflict
        assert torch.all(torch.eq(gt.edges, torch.tensor([
            [0, 1],
            [1, 2],
            [3, 4],
            [4, 5]
        ])))

        # check indices
        assert torch.all(torch.eq(gt.node_indices, torch.tensor([0, 0, 0, 1, 1, 1])))
        assert torch.all(torch.eq(gt.edge_indices, torch.tensor([0, 0, 1, 1])))


def test_validate_graph_tuples():
    gen = graph_generator((2, 10), (2, 2), (2, 2), (2, 2))

    for _ in range(100):
        graph = next(gen)
        input_gt = to_graph_tuple([graph])
        print(input_gt)
        validate_gt(input_gt)