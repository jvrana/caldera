from pyro_graph_nets.utils import generate_networkx_graphs
from pyro_graph_nets.utils.graph_tuple import to_graph_tuple, print_graph_tuple_shape
from typing import Tuple
import numpy as np
from pyro_graph_nets.models import EncoderProcessDecoder
from pyro_graph_nets.models import gt_wrap_replace
from pyro_graph_nets.blocks import MLP, EdgeBlock, NodeBlock, GlobalBlock, Aggregator
from pyro_graph_nets.models import GraphNetwork, GraphEncoder, cat_gt
import pytest


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


@pytest.fixture(scope='function')
def input_target():
    input_gt, target_gt = generate_graph_batch(100, (2, 20))
    return input_gt, target_gt


@pytest.fixture(scope='function')
def encoder():
    encoder_model = GraphEncoder(
        EdgeBlock(MLP(1, 1), independent=True),
        NodeBlock(MLP(5, 16, 5), independent=True),
        None
    )
    return encoder_model


def test_encoder(encoder, input_target):
    input_gt, target_gt = input_target
    out = encoder(input_gt)
    print()
    print_graph_tuple_shape(out)


@pytest.mark.parametrize('steps', [1, 5])
def test_core(input_target, steps):
    input_gt, target_gt = input_target

    #######
    # Model
    #######

    enc_v = (5, 16, 5)
    enc_e = (1, 1)
    core_e = (enc_e[-1] * 2 + enc_v[-1] * 4, 16, enc_e[-1])
    core_v = (core_e[-1] + enc_v[-1] * 2, 16, enc_v[-1])

    print(enc_v)
    print(enc_e)
    print(core_e)
    print(core_v)

    encoder = GraphEncoder(
        EdgeBlock(MLP(*enc_e), independent=True),
        NodeBlock(MLP(*enc_v), independent=True),
        None
    )

    core = GraphNetwork(
        EdgeBlock(MLP(*core_e), independent=False),
        NodeBlock(MLP(*core_v), independent=False, edge_aggregator=Aggregator('mean')),
        None
    )

    decoder = GraphEncoder(
        EdgeBlock(MLP(*enc_e), independent=True),
        NodeBlock(MLP(*enc_v), independent=True),
        None
    )

    #######
    # Wrap
    #######
    #######
    # Train
    #######

    latent = encoder(input_gt)
    latent0 = latent

    output = []
    for _ in range(steps):
        print(_)
        core_input = cat_gt(latent0, latent)
        print_graph_tuple_shape(core_input)
        latent = core(core_input)
        decoded = decoder(latent)
        output.append(decoded)


