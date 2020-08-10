import pytest


class NetworkTestCase(object):

    def __init__(self, network, loader, device, to_input, to_output):
        pass


    def train(self):
        pass


@pytest.fixture(params=[
    ()
])
def cases(request):
    print(request.param)


def test(cases):
    pass