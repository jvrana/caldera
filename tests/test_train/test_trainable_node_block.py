import functools

import pytest
import torch
import torch.optim as optim

from pyrographnets.blocks import Flex, MLP, NodeBlock, EdgeBlock
from pyrographnets.data import GraphData, GraphDataLoader
from typing import Dict, Any, Callable, TypeVar, Hashable, Tuple


def train(net: torch.nn.Module,
          loader: GraphDataLoader,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          criterion,
          batch_to_input: Callable,
          batch_to_target: Callable,
          device: str = None):
    """Run training loop"""
    loss_arr = torch.zeros(epochs)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.
        for batch in loader:
            if device:
                batch = batch.to(device)
            input = batch_to_input(batch)
            target = batch_to_target(batch)

            optimizer.zero_grad()  # zero the gradient buffers
            output = net(input)
            loss = criterion(output, target)
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()
        loss_arr[epoch] = running_loss
    return loss_arr


class NetworkTestCase(object):

    def __init__(self, network, loader, device, to_input, to_output):
        self.to_input = to_input
        self.to_output = to_output
        network.to(device)
        network(to_input(loader.first()).to(device))
        self.network = network

    def train(self):
        optimizer = optim.AdamW(self.network.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()

        losses = train(self.network,
                       self.loader,
                       20,
                       optimizer,
                       criterion,
                       self.to_input,
                       self.to_output)
        return losses


@pytest.fixture(params=[
    ()
])
def cases():
    pass


def test_(device):
    epochs = 20
    datalist = [GraphData.random(5, 5, 5, requires_grad=True) for _ in range(1000)]

    def to_input(batch):
        return batch.view(slice(None, -1), None, None).x

    def to_output(batch):
        return batch.view(slice(-1, None), None, None).x

    # modify datalist
    for data in datalist:
        data.x = torch.cat([
            data.x,
            data.x.sum(axis=1, keepdim=True)
        ], axis=1)

    # create loader
    loader = GraphDataLoader(datalist, batch_size=100)

    # create network
    network = torch.nn.Sequential(
        torch.nn.Sequential(
            NodeBlock(
                Flex(MLP)(Flex.d(), 25, 25, layer_norm=False)
            ),
            torch.nn.Linear(25, 1)
        )
    )

    network.to(device)
    network(to_input(loader.first()).to(device))

    optimizer = optim.AdamW(network.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    losses = train(network, loader, epochs, optimizer, criterion,
                   to_input,
                   to_output)
    print(losses)
    assert losses[-1].item() < losses[0].item()