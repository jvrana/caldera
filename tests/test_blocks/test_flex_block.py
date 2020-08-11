import torch
from pyrographnets.blocks import Flex
import pytest


def test_flex_block():
    flex_linear = Flex(torch.nn.Linear)
    model = flex_linear(Flex.d(), 11)
    print(model.__str__())
    print(model.__repr__())
    x = torch.randn((30, 55))
    model(x)
    print(model.__str__())
    print(model.__repr__())

@pytest.mark.parametrize("x", [16, 32, 44])
def test_flex_block_chain(x):

    model = torch.nn.Sequential(
        Flex(torch.nn.Linear)(Flex.d(), 16),
        Flex(torch.nn.Linear)(Flex.d(), 32),
        Flex(torch.nn.Linear)(Flex.d(), 64),
    )

    data = torch.randn((10, x))
    out = model(data)
    assert out.shape[1] == 64


@pytest.mark.parametrize("x", [16, 32, 44])
def test_flex_block_custom_position(x):
    class FooBlock(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.block = torch.nn.Linear(a, b)

        def forward(self, steps, data):
            return self.block(data)

    model = Flex(FooBlock)(Flex.d(1), 16)
    data = torch.randn((10, x))
    model("arg0", data)

def test_flex_block_children():
    model = Flex(torch.nn.Linear)(Flex.d(), 1)
    model.reset_parameters()
    # for child in model.children():
    #     print(child)
    #
    # model(torch.randn((10, 3)))
    #
    # for child in model.children():
    #     print(hasattr(child, 'reset_parameters'))


@pytest.mark.parametrize('dtype', [None, torch.float64])
@pytest.mark.parametrize('dtype2', [None, torch.float32])
def test_flex_block_to_playback(dtype, dtype2):
    class Network(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = Flex(torch.nn.Linear)(Flex.d(), 5)

        def forward(self, x):
            return self.layers(x)

    net = Network()
    if dtype:
        net.to(torch.float64)

    print("Before")
    print(list(net.state_dict()))

    if not dtype:
        data = torch.randn(5, 8)
    else:
        data = torch.randn(5, 8, dtype=dtype)
    net(data)

    print("After")
    print(list(net.state_dict()))
    for p in net.parameters():
        if dtype:
            assert p.dtype is dtype

    if dtype2:
        net.to(dtype2)
    for p in net.parameters():
        if dtype2:
            assert p.dtype is dtype2