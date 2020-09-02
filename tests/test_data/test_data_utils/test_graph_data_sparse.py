import torch

from caldera.data import GraphBatch
from caldera.data import GraphData
from caldera.data.utils import to_sparse_coo_matrix


def test_to_coo_matrix():
    data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
    W = to_sparse_coo_matrix(data)
    assert W.size() == torch.Size([data.num_edges, data.num_edges, 4])


def test_to_coo_matrix_with_fill():
    data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
    W = to_sparse_coo_matrix(data, fill_value=1)
    print(W)
    assert W.size() == torch.Size([data.num_edges, data.num_edges])


def test_create_edge_mask():
    """Attempt to create a fast edge mask from sparse matrix.

    Results: There is no advantage to doing this over repeated calls of `long_isin`.

    :return:
    """
    from caldera.utils import scatter_coo

    data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)

    ij = torch.cat([data.edges])

    sparse_mask = scatter_coo(ij, 1, expand=True, size=(data.num_nodes, data.num_nodes))

    import torch_scatter

    from caldera.utils import torch_coo_to_scipy_coo
    from scipy.sparse import csr_matrix
    from caldera.utils import long_isin

    matrix = csr_matrix(torch_coo_to_scipy_coo(sparse_mask.coalesce()))

    nodes = torch.randint(1, 10, (1000,))

    import time

    _nodes = torch.cat([nodes] * 1000)
    t1 = time.time()
    res = matrix[_nodes]
    res.todense()
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    for _n in range(1000):
        res = matrix[nodes]
        a = torch.BoolTensor([False] * 1000)

    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    edges = data.edges.flatten()
    for _ in range(1000):
        res = long_isin(edges, nodes)
    t2 = time.time()
    print(t2 - t1)
    # print(res)
    # from caldera.utils import long_isin
    # t1 = time.time()
    # for i in range(1000):
    #     long_isin(data.edges.flatten(), nodes)
    # t2 = time.time()
    # print(t2 - t1)
    # t1 = time.time()
    # for i in range(1000):
    #     matrix[nodes]
    # t2 = time.time()
    # print(t2 - t1)
