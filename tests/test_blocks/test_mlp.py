from pyrographnets.blocks import MLP
import torch
import pytest

@pytest.mark.parametrize('layers', [
    (8, 32),
    (16, 16, 32),
    (64, 16, 8)
])
def test_mlp(layers):
    block = MLP(*layers)
    out = block(torch.randn(10, layers[0]))
    assert out.shape[1] == layers[-1]
    print(list(block.modules()))
    # assert len(list(block.modules())) == 4 * len(layers)