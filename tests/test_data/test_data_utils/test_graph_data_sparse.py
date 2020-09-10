import pytest
import torch

from caldera.data import GraphData
from caldera.data.utils import to_sparse_coo_matrix
from caldera.utils import scatter_coo


def test_to_coo_matrix():
    data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
    W = to_sparse_coo_matrix(data)
    assert W.size() == torch.Size([data.num_edges, data.num_edges, 4])


def test_to_coo_matrix_with_fill():
    data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
    W = to_sparse_coo_matrix(data, fill_value=1)
    print(W)
    assert W.size() == torch.Size([data.num_edges, data.num_edges])


@pytest.mark.benchmark
class TestBenchmarkScatterCoo:
    def test_benchmark_scatter_coo(self, benchmark):
        def run():
            data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
            ij = torch.cat([data.edges])
            sparse_mask = scatter_coo(
                ij, 1, expand=True, size=(data.num_nodes, data.num_nodes)
            )

        benchmark(run)


#
#     matrix = csr_matrix(torch_coo_to_scipy_coo(sparse_mask.coalesce()))
#     nodes = torch.randint(1, 10, (1000,))
#
#     # def _test_indexing_from_sparse(n_nodes):
#     #     _nodes = torch.cat([nodes] * n_nodes)
#     #     def run():
#     #         res = matrix[_nodes]
#     #         res.todense()
#     #     return run
#     #
#     # def _test_indexing_each_row(n_nodes):
#     #     def run():
#     #         for _n in range(n_nodes):
#     #             res = matrix[nodes]
#     #             a = torch.BoolTensor([False] * n_nodes)
#     #     return run
#
#     def _test_long_is_in(n_nodes):
#         edges = data.edges.flatten()
#         def run():
#             for _ in range(n_nodes):
#                 res = long_isin(edges, nodes)
#         return run
#
#     benchmark(_test_long_is_in(100))
#
# def test_benchmark_indexing(self, benchmark):
#     """Attempt to create a fast edge mask from sparse matrix.
#
#     Results: There is no advantage to doing this over repeated calls of `long_isin`.
#
#     :return:
#     """
#
#     data = GraphData.random(5, 4, 3, min_edges=1000, min_nodes=1000)
#     ij = torch.cat([data.edges])
#     sparse_mask = scatter_coo(ij, 1, expand=True, size=(data.num_nodes, data.num_nodes))
#     matrix = csr_matrix(torch_coo_to_scipy_coo(sparse_mask.coalesce()))
#     nodes = torch.randint(1, 10, (1000,))
#
#     # def _test_indexing_from_sparse(n_nodes):
#     #     _nodes = torch.cat([nodes] * n_nodes)
#     #     def run():
#     #         res = matrix[_nodes]
#     #         res.todense()
#     #     return run
#     #
#     def _test_indexing_each_row(n_nodes):
#         def run():
#             for _n in range(n_nodes):
#                 res = matrix[nodes]
#                 a = torch.BoolTensor([False] * n_nodes)
#         return run
#
#     # def _test_long_is_in(n_nodes):
#     #     edges = data.edges.flatten()
#     #     def run():
#     #         for _ in range(n_nodes):
#     #             res = long_isin(edges, nodes)
#     #     return run
#
#     benchmark(_test_indexing_each_row(100))
#     # benchmark(_test_indexing_each_row(100))
#     # benchmark(_test_indexing_from_sparse(100))
#
