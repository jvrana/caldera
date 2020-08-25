from caldera.data.utils import adj_matrix


def test_adj_matrix(random_data_example):
    M = adj_matrix(random_data_example)
