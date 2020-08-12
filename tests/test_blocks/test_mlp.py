import pytest
import torch

from caldera.blocks import MLP


@pytest.mark.parametrize("layers", [(8, 32), (16, 16, 32), (64, 16, 8)])
@pytest.mark.parametrize("dropout", [None, 0.0, 0.2, 0.5])
@pytest.mark.parametrize("layer_norm", [False, True])
def test_mlp(layers, dropout, layer_norm):
    block = MLP(*layers, dropout=dropout, layer_norm=layer_norm)
    out = block(torch.randn(10, layers[0]))
    assert out.shape[1] == layers[-1]
    print(list(block.modules()))
    # assert len(list(block.modules())) == 4 * len(layers)
