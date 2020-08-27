from caldera.data import GraphBatch
from caldera.transforms import Compose
from caldera.transforms import RandomHop
from caldera.transforms import RandomNodeMask
from caldera.transforms import Shuffle


def test_compose(seeds):
    compose = Compose([RandomNodeMask(0.1), RandomHop(1, 3), Shuffle()])

    data = GraphBatch.random_batch(1000, 5, 4, 3)
    data = compose(data)


def test_compose(seeds):
    compose = Compose([RandomNodeMask(0.1), RandomHop(1, 3), Shuffle()])

    data = GraphBatch.random_batch(1000, 5, 4, 3)
    data = compose(data)
