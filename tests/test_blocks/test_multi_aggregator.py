import pytest
import torch

from caldera.blocks import Aggregator
from caldera.blocks import MultiAggregator


@pytest.mark.parametrize("methods", [["min", "max"], ["min", "max", "mean"]])
@pytest.mark.parametrize("hard_select", [True, False])
@pytest.mark.parametrize("attn", ["local", "global"])
def test_aggregators(methods, hard_select, attn):
    shape = (10, 5)
    block = MultiAggregator(shape[1], methods, attention=attn, hard_select=hard_select)
    print(block)
    idx = torch.randint(0, 20, (shape[0],))
    x = torch.randn(shape)
    out = block(x, idx, dim=0, dim_size=20)
    assert out.shape == (20, 5)
