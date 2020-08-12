import pytest
import torch

from caldera.blocks import Aggregator
from caldera.blocks import MultiAggregator


@pytest.mark.parametrize("method", ["mean", "max", "min", "add"])
def test_aggregators(method):
    block = Aggregator(method)
    idx = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    x = torch.tensor([0, 1, 10, 3, 4, 55, 6])
    out = block(x, idx, dim=0)
    print(out)


@pytest.mark.parametrize("method", ["mean", "max", "min", "add"])
def test_aggregators_2d(method):
    block = Aggregator(method)
    idx = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    x = torch.randn((7, 3))
    out = block(x, idx, dim=0)
    print(out)


def test_invalid_method():
    with pytest.raises(ValueError):
        block = Aggregator("not a method")


@pytest.mark.parametrize("methods", [["min", "max"]])
def test_multi_aggregators(methods):
    shape = (10, 5)
    block = MultiAggregator(shape[1], methods)
    idx = torch.randint(0, 20, (shape[0],))
    x = torch.randn(shape)
    out = block(x, idx, dim=0, dim_size=20)
    print(out)
