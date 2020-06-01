from pyrographnets.utils import scatter_group
import torch
from flaky import flaky

def test_scatter_group_0():

    idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

    out = scatter_group(x, idx)
    assert torch.all(torch.eq(out[0], torch.tensor([0, 1, 2])))
    assert len(out[1]) == 3
    assert torch.all(torch.eq(out[1][0], torch.tensor([0, 1, 2])))
    assert torch.all(torch.eq(out[1][1], torch.tensor([3, 4, 5])))
    assert torch.all(torch.eq(out[1][2], torch.tensor([6, 7, 8])))

def test_scatter_group_1():

    idx = torch.tensor([0, 0, 0, 1, 1, 1, 3, 3, 3])
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

    out = scatter_group(x, idx)
    assert torch.all(torch.eq(out[0], torch.tensor([0, 1, 3])))
    assert len(out[1]) == 3
    assert torch.all(torch.eq(out[1][0], torch.tensor([0, 1, 2])))
    assert torch.all(torch.eq(out[1][1], torch.tensor([3, 4, 5])))
    assert torch.all(torch.eq(out[1][2], torch.tensor([6, 7, 8])))


def test_scatter_group_2():

    idx = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2, 2, 2])
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    out = scatter_group(x, idx)
    assert torch.all(torch.eq(out[0], torch.tensor([0, 1, 2])))
    assert len(out[1]) == 3
    assert torch.all(torch.eq(out[1][0], torch.tensor([0, 1, 2, 3, 4, 5])))
    assert torch.all(torch.eq(out[1][1], torch.tensor([6])))
    assert torch.all(torch.eq(out[1][2], torch.tensor([7, 8, 9])))


def test_scatter_group_3():

    idx = torch.tensor([2, 2, 0, 1, 1, 1, 2])
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6])

    out = scatter_group(x, idx)
    assert torch.all(torch.eq(out[0], torch.tensor([0, 1, 2])))

    assert torch.all(torch.eq(out[1][0], torch.tensor([2])))
    assert torch.all(torch.eq(out[1][1], torch.tensor([3, 4, 5])))
    assert torch.all(torch.eq(out[1][2], torch.tensor([0, 1, 6])))


@flaky(min_passes=10, max_runs=10)
def test_scatter_group4():

    idx = torch.tensor([0, 0, 0, 0, 0, 1, 1])
    x = torch.randn(7, 3)
    out = scatter_group(x, idx)
    assert torch.all(torch.eq(out[1][0], x[:5]))
    assert torch.all(torch.eq(out[1][1], x[5:]))