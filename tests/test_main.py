from pyro_graph_nets.utils import generate_networkx_graphs
from pyro_graph_nets.graph_tuple import to_graph_tuple, print_graph_tuple_shape
from typing import Tuple
import numpy as np
from pyro_graph_nets.models import EncoderProcessDecoder

seed = 2
rand = np.random.RandomState(seed=seed)


def generate_graph_batch(num_train, n_nodes: Tuple[int, int], theta: int = 20):
    input_graphs, target_graphs, graphs = generate_networkx_graphs(rand, num_train, n_nodes, theta)
    return to_graph_tuple(input_graphs), to_graph_tuple(target_graphs)


def test_generate_graphs():

    input_gt, target_gt = generate_graph_batch(100, (2, 20))
    assert input_gt
    assert target_gt
    print_graph_tuple_shape(input_gt)
    print("target")
    print_graph_tuple_shape(target_gt)


# def test_encoder():
#     model = EncoderProcessDecoder(
#         5,
#     )
#     input_gt, target_gt = generate_graph_batch(100, (2, 20))
#

