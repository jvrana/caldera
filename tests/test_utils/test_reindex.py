from caldera.utils import reindex_tensor
import torch


def test_reindex():
    a = torch.tensor([1, 1, 1, 1, 0, 2, 0, 5, 6])
    expected = torch.tensor([0, 0, 0, 0, 1, 2, 1, 3, 4])
    b = reindex_tensor(a)
    assert torch.all(b == expected)


def test_reindex_tuple():
    a = torch.tensor([1, 1, 1, 1, 0, 2, 0, 5, 6])
    b = torch.tensor([6, 5, 1, 70])
    c = torch.tensor([0, 80, 5])
    expected1 = torch.tensor([0, 0, 0, 0, 1, 2, 1, 3, 4])
    expected2 = torch.tensor([4, 3, 0, 5])
    expected3 = torch.tensor([1, 6, 3])
    d, e, f = reindex_tensor(a, b, c)
    assert torch.all(d == expected1)
    assert torch.all(e == expected2)
    assert torch.all(f == expected3)


def test_reindex_tuple_ndim():
    a = torch.tensor([1, 1, 1, 1, 0, 2, 0, 5, 6])
    b = torch.tensor([[6, 5, 1, 70], [0, 80, 5, 6]])
    expected1 = torch.tensor([0, 0, 0, 0, 1, 2, 1, 3, 4])
    expected2 = torch.tensor([[4, 3, 0, 5], [1, 6, 3, 4]])
    c, d = reindex_tensor(a, b)
    assert torch.all(c == expected1)
    assert torch.all(d == expected2)
