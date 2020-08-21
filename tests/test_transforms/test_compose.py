from caldera.transforms import Compose, RandomNodeMask, RandomHop, Shuffle
from caldera.data import GraphBatch


def test_compose(seeds):
    compose = Compose([RandomNodeMask(0.1), RandomHop(1, 3), Shuffle()])

    data = GraphBatch.random_batch(1000, 5, 4, 3)
    data = compose(data)
