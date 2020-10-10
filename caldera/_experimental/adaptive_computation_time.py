from typing import List

from torch import nn

from caldera import gnn
from caldera.data import GraphBatch
import torch

class GraphSigmoid(nn.Module):

    def __init__(self):
        pass

    def forward(self, data):
        return nn.Sigmoid(nn.Linear(3, 1), nn.Sigmoid())

class ACTGraphNet(object):

    def __init__(self):
        self.halt =

    def forward(self, data):
        latent0 = self.encode(data)
        data = latent0.clone()
        epsilon = 1.
        r = torch.tensor(1.)

        while r > epsilon:
            # normal core stuff

            h = self.halt(out)
            _r = r - h