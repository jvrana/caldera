from pyrographnets.blocks import Aggregator
import torch
import pytest


@pytest.mark.parametrize("method", ["mean", "max", "min", "add"])
def test_aggregators(method):
    block = Aggregator(method)
    idx = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    x = torch.tensor([0, 1, 10, 3, 4, 55, 6])
    out = block(x, idx, dim=0)
    print(out)


def test_invalid_method():
    with pytest.raises(ValueError):
        block = Aggregator("not a method")
