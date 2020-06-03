import torch
from pyrographnets.blocks import Flex
import pytest


def test_flex_block():
    flex_linear = Flex(torch.nn.Linear)
    model = flex_linear(Flex.d(), 11)
    print(model)
    x = torch.randn((30, 55))
    model(x)
    print(model)

@pytest.mark.parametrize('x', [16, 32, 44])
def test_flex_block_chain(x):

    model = torch.nn.Sequential(
        Flex(torch.nn.Linear)(Flex.d(), 16),
        Flex(torch.nn.Linear)(Flex.d(), 32),
        Flex(torch.nn.Linear)(Flex.d(), 64)
    )

    data = torch.randn((10, x))
    out = model(data)
    assert out.shape[1] == 64


@pytest.mark.parametrize('x', [16, 32, 44])
def test_flex_block_custom_position(x):
    class FooBlock(torch.nn.Module):

        def __init__(self, a, b):
            super().__init__()
            self.block = torch.nn.Linear(a, b)

        def forward(self, steps, data):
            return self.block(data)

    model = Flex(FooBlock)(Flex.d(1), 16)
    data = torch.randn((10, x))
    model('arg0', data)