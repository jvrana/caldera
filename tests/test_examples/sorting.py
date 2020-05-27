import itertools
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

# from pgn.models import EncoderCoreDecoder
# from pgn.utils import batch_data

torch.set_num_threads(4)


def graph_data_from_list(input_list):
    """Takes a list with the data, generates a fully connected graph with
    values of the list as nodes.

    Parameters
    ----------
    input_list: list
        list of the numbers to sort

    Returns
    -------
    graph_data: dict
        Dict of entities for the list provided.
    """
    connectivity = torch.tensor(
        [el for el in itertools.product(range(len(input_list)), repeat=2)],
        dtype=torch.long,
    ).t()
    vdata = torch.tensor([[v] for v in input_list], dtype=torch.float)
    edata = torch.zeros(connectivity.shape[1], 1, dtype=torch.float)
    return (vdata, edata, connectivity)


def edge_id_by_sender_and_receiver(connectivity, sid, rid):
    """Get edge id from the information about its sender and its receiver.

    Parameters
    ----------
    metadata: list
        list of pgn.graph.Edge objects
    sid: int
        sender id
    rid: int
        receiver id

    Returns
    -------
    """
    return (connectivity[0, :] == sid).mul(connectivity[1, :] == rid).nonzero().item()


def create_target_data(vdata, edata, connectivity):
    """Generate target data for training.

    Parameters
    ----------
    input_data: list
        list of data to sort

    Returns
    -------
    res: dict
        dict of target graph entities
    """
    # two nodes might have true since they might have similar values
    min_val = vdata.min()

    # [prob_true, prob_false]
    target_vertex_data = torch.tensor(
        [[1.0, 0.0] if v == min_val else [0.0, 1.0] for v in vdata], dtype=torch.double
    )

    sorted_ids = vdata.argsort(dim=0).flatten()
    target_edge_data = torch.zeros(edata.shape[0], 2, dtype=torch.double)
    for sidx, sid in enumerate(sorted_ids):
        for ridx, rid in enumerate(sorted_ids):
            eid = edge_id_by_sender_and_receiver(connectivity, sid, rid)
            # we look for exact comparison here since we sort
            if sidx < len(sorted_ids) - 1 and ridx == sidx + 1:
                target_edge_data[eid][0] = 1.0
            else:
                target_edge_data[eid][1] = 1.0

    return target_vertex_data, target_edge_data


def generate_graph_batch(n_examples, sample_length):
    """generate all of the training data.

    Parameters
    ----------
    n_examples: int
        Num of the samples
    sample_length: int
        Length of the samples.
        # TODO we should implement samples of different lens as in the DeepMind example.
    Returns
    -------
    res: tuple
        (input_data, target_data), each of the elements is a list of entities dicts
    """

    input_data = [
        graph_data_from_list(np.random.uniform(size=sample_length))
        for _ in range(n_examples)
    ]
    target_data = [create_target_data(v, e, conn) for v, e, conn in input_data]

    return input_data, target_data


def test_generate_data():
    input_data, target_data = generate_graph_batch(100, 30)
    print(input_data)
    print(target_data)
    print()


#
# def batch_loss(outs, targets, criterion, batch_size, core_steps):
#     """get the loss for the network outputs
#
#     Parameters
#     ----------
#     outs: list
#         list of lists of the graph network output, time is 0-th dimension, batch is 1-th dimension
#     targets: list
#         list of the graph entities for the expected output
#     criterion: torch._Loss object
#         loss to use
#     Returns
#     -------
#     loss: float
#         Shows how good your mode is.
#     """
#     loss = 0
#     vsize = targets[0].shape[0] // batch_size
#     esize = targets[1].shape[0] // batch_size
#     for out in outs:
#         for i in range(batch_size):
#             loss += criterion(
#                 out[0][i * vsize : (i + 1) * vsize],
#                 targets[0][i * vsize : (i + 1) * vsize],
#             )
#         for i in range(batch_size):
#             loss += criterion(
#                 out[1]["default"][i * esize : (i + 1) * esize],
#                 targets[1][i * esize : (i + 1) * esize],
#             )
#
#     return loss / core_steps / batch_size
