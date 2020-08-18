from caldera.utils import long_isin
import torch
import pytest


def test_long_isin():
    a = torch.LongTensor([1, 2, 3, 4])
    b = torch.LongTensor([2, 3, 4, 5])
    c = long_isin(a, b)
    assert torch.all(c == torch.BoolTensor([False, True, True, True]))
    d = long_isin(b, a)
    assert torch.all(d == torch.BoolTensor([True, True, True, False]))


@pytest.mark.parametrize(('x', 'y', 'does_raise'), [
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (True, True, True)
]
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
    # a = torch.LongTensor([1, 2, 3, 4])
    # b = torch.LongTensor([2, 3, 4, 5])
    # c = long_isin(a, b)