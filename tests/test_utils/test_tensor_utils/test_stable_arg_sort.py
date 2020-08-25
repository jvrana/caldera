import pytest
import torch

from caldera.utils import stable_arg_sort_long


@pytest.mark.parametrize(
    "a",
    [
        torch.randint(10, (10,)),
        torch.randint(10, (100,)),
        torch.randint(2, (100,)),
        torch.randint(100, (100,)),
        pytest.param(
            torch.randn((10,)),
            marks=pytest.mark.xfail(strict=True, reason="float not supported"),
        ),
    ],
)
def test_stable_arg_sort_long(a):
    idx = stable_arg_sort_long(a)
    assert torch.all(a[idx] == a.sort().values)


@pytest.mark.parametrize(
    "a",
    [
        torch.randint(10, (2, 10)),
        torch.randint(10, (2, 100)),
        torch.randint(100, (10, 100)),
    ],
)
def test_stable_arg_sort_long_broadcast(a):
    idx = stable_arg_sort_long(a)
    assert torch.all(a.sort(dim=-1).values == torch.gather(a, -1, idx))
