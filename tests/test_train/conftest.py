"""test_train/conftest.py

Tests blocks and networks are differentiable and trainable"""

import pytest
import torch


def get_cuda_device():
    if torch.cuda.is_available():
        return 'cuda:' + str(torch.cuda.current_device())

devices = ['cpu']
if get_cuda_device():
    devices.append(get_cuda_device())


@pytest.fixture(params=devices)
def device(request):
    return request.param