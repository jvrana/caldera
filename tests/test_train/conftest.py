"""test_train/conftest.py

Tests blocks and networks are differentiable and trainable"""

import pytest
import torch


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return 'cuda:' + str(torch.cuda.current_device())
    return None