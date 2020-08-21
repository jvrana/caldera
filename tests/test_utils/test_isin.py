from caldera.utils import long_isin
import torch
import pytest
from caldera.utils import deterministic_seed


def test_long_isin_explicit():
    a = torch.LongTensor([1, 2, 3, 4])
    b = torch.LongTensor([2, 3, 4, 5])
    c = long_isin(a, b)
    assert torch.all(c == torch.BoolTensor([False, True, True, True]))
    d = long_isin(b, a)
    assert torch.all(d == torch.BoolTensor([True, True, True, False]))


def test_long_isin_explicit2():
    a = torch.LongTensor([1, 2, 3, 4])
    b = torch.LongTensor([])
    c = long_isin(a, b)
    assert torch.all(c == torch.BoolTensor([False, False, False, False]))
    d = long_isin(b, a)
    assert torch.all(d == torch.BoolTensor([]))


def test_long_isin_explicit3():
    a = torch.LongTensor([1, 2, 3, 4])
    b = torch.LongTensor([5, 6])
    c = long_isin(a, b)
    assert torch.all(c == torch.BoolTensor([False, False, False, False]))
    d = long_isin(b, a)
    assert torch.all(d == torch.BoolTensor([False, False]))


def test_long_isin_explicit4():
    a = torch.LongTensor([1, 2, 3, 4])
    b = a
    c = long_isin(a, b)
    assert torch.all(c == torch.BoolTensor([True, True, True, True]))
    d = long_isin(b, a)
    assert torch.all(d == torch.BoolTensor([True, True, True, True]))


def isin(a, b):
    c = []
    for _a in a:
        _c = False
        for _b in b:
            if _a == _b:
                _c = True
                break
        c.append(_c)
    return c


@pytest.mark.parametrize(
    ("a", "b"), [([1, 2, 3, 4], [4, 5]), ([4, 3, 2, 1], [5, 100, 2])]
)
def test_long_isin_compare(a, b):
    a = torch.LongTensor(a)
    b = torch.LongTensor(b)
    expected_c = torch.LongTensor(isin(a, b))
    expected_d = torch.LongTensor(isin(b, a))
    c = long_isin(a, b)
    d = long_isin(b, a)

    assert torch.all(c == expected_c)
    assert torch.all(d == expected_d)


@pytest.mark.parametrize(
    ("a_range", "b_range"),
    [
        ((0, 5, 5), (2, 10, 10)),
        ((0, 2, 7), (2, 6, 11)),
        ((2, 10, 10), (0, 5, 5)),
        ((0, 50, 50), (2, 10, 100)),
        ((2, 10, 100), (0, 50, 50)),
        ((0, 1, 10), (0, 1, 10)),
        ((0, 1, 10), (3, 4, 10)),
        ((0, 1, 10), (3, 4, 0)),
        ((0, 1, 0), (3, 4, 10)),
    ],
)
def test_long_isin_random(a_range, b_range):
    deterministic_seed(0)
    a = torch.randint(a_range[0], a_range[1], (a_range[2],))
    b = torch.randint(b_range[0], b_range[1], (b_range[2],))
    expected_c = torch.BoolTensor(isin(a, b))
    c = long_isin(a, b)
    assert torch.all(c == expected_c)


@pytest.mark.parametrize(
    ("x", "y", "does_raise"),
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_double_isin(x, y, does_raise):
    a = torch.LongTensor([1, 2, 3, 4])
    b = torch.LongTensor([2, 3, 4, 5])
    if x:
        a = a.to(torch.float)
    if y:
        b = b.to(torch.float)
    if does_raise:
        with pytest.raises(ValueError):
            long_isin(a, b)
    else:
        long_isin(a, b)
