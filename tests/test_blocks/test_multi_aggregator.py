import pytest
import torch

from caldera.blocks import Aggregator
from caldera.blocks import MultiAggregator


@pytest.mark.parametrize("methods", [["min", "max"]])
def test_aggregators(methods):
    shape = (10, 5)
    block = MultiAggregator(shape[1], methods)
    idx = torch.randint(0, 20, (shape[0],))
    x = torch.randn(shape)
    out = block(x, idx, dim=0, dim_size=20)
    print(out)
