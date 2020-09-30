import pickle

from caldera.data import GraphBatch
from caldera.data import GraphData


def test_serialize_graph_data():
    data = GraphData.random(5, 4, 3)
    pickle.loads(pickle.dumps(data))


def test_serialize_graph_batch():
    data = GraphBatch.random_batch(100, 5, 4, 3)
    pickle.loads(pickle.dumps(data))
