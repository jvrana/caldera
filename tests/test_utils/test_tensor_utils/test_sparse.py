import pytest
import torch

from caldera.utils.sparse import scatter_coo


class TestScatterCoo:

    def test_scatter_coo_2D(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            0.1, 0.2, 0.3, 0.4
        ])
        matrix = scatter_coo(indices, values)
        assert matrix.size() == torch.Size([4, 5])

    def test_scatter_coo_size_2D_size(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            0.1, 0.2, 0.3, 0.4
        ])
        matrix = scatter_coo(indices, values, size=(10, 10))
        assert matrix.size() == torch.Size([10, 10])

    def test_scatter_coo_size_2D_size_ellipsis(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            0.1, 0.2, 0.3, 0.4
        ])
        matrix = scatter_coo(indices, values, size=(10, 10, ...))
        assert matrix.size() == torch.Size([10, 10])

    def test_scatter_coo_3D(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5]
        ])
        matrix = scatter_coo(indices, values)
        assert matrix.size() == torch.Size([4, 5, 2])

    def test_scatter_coo_3D(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5]
        ])
        matrix = scatter_coo(indices, values, size=(10, 10, ...))
        assert matrix.size() == torch.Size([10, 10, 2])

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Size has wrong dimensions")
    def test_scatter_coo_3D_fail(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5]
        ])
        scatter_coo(indices, values, size=(10, 10))

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Tensors have wrong dimensions")
    def test_scatter_coo_size_mismatch(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            [0, 1, 3]
        ])
        scatter_coo(indices, values)

    def test_scatter_coo_infer_second_dim(self):
        indices = torch.LongTensor([0, 1, 2, 3])
        values = torch.tensor([10, 20, 30, 40])
        matrix = scatter_coo(indices, values)
        assert matrix.size() == torch.Size([1, 4])

    def test_scatter_coo_5D(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1],
            [1, 2, 3, 4]
        ])
        values = torch.tensor([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5]
        ])
        matrix = scatter_coo(indices, values, size=(10, 10, 10, ...))
        assert matrix.size() == torch.Size([10, 10, 10, 2])

    def test_scatter_coo_2D_expand_with_num(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = 1
        matrix = scatter_coo(indices, values, expand=True)
        assert matrix.size() == torch.Size([4, 5])

    def test_scatter_coo_2D_expand_with_0D_tensor(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor(1)
        matrix = scatter_coo(indices, values, expand=True)
        assert matrix.size() == torch.Size([4, 5])

    def test_scatter_coo_2D_expand_with_1D_tensor(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([1])
        matrix = scatter_coo(indices, values, expand=True)
        assert matrix.size() == torch.Size([4, 5, 1])

    def test_scatter_coo_2D_expand_with_1D_tensor2(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([1, 2, 3])
        matrix = scatter_coo(indices, values, expand=True)
        assert matrix.size() == torch.Size([4, 5, 3])

    def test_scatter_coo_2D_expand_with_2D_tensor(self):
        indices = torch.LongTensor([
            [0, 1, 2, 3],
            [4, 3, 2, 1]
        ])
        values = torch.tensor([
            [1, 2, 3],
            [3, 4, 5]
        ])
        matrix = scatter_coo(indices, values, expand=True)
        assert matrix.size() == torch.Size([4, 5, 2, 3])
    #
#
# @pytest.mark.parametrize(
#     ("n", "m", "o"), [(2, 10, 3), (3, 10, 3), (1, 1, 1), (2, 10, 0), (0, 10, 3)]
# )
# @pytest.mark.parametrize("size", [(10, 10, ...), (20, 20, ...), None])
# def test_scatter_coo(n, m, o, size):
#     s1 = (n, m)
#     if n is None:
#         s1 = (m,)
#     s2 = (m, o)
#     if o is None:
#         s2 = (m,)
#     indices = torch.randint(1, 10, s1)
#     values = torch.randn(s2)
#     matrix = scatter_coo(indices, values, size=size)
#     print(matrix.size)
#     # if size is not None:
#     #     assert matrix.size() == size
#
#
# @pytest.mark.parametrize(
#     ("n", "m", "o"),
#     [(None, 10, None), (1, 10, None), (3, 10, None), (None, 10, 1), (None, 10, 3)],
# )
# @pytest.mark.parametrize("size", [(10, 10), (20, 20), None,])
# def test_scatter_coo_1dim(n, m, o, size):
#     s1 = (n, m)
#     if n is None:
#         s1 = (m,)
#     s2 = (m, o)
#     if o is None:
#         s2 = (m,)
#     indices = torch.randint(1, 10, s1)
#     values = torch.randn(s2)
#     matrix = scatter_coo(indices, values)
#     print(matrix)
#
#
# @pytest.mark.parametrize(("n", "m"), [(2, 10)])
# @pytest.mark.parametrize("values", [0, torch.tensor(0), torch.tensor([0, 1, 2])])
# @pytest.mark.parametrize("size", [(10, 10), (20, 20), None])
# def test_scatter_fill(n, m, values, size):
#     s1 = (n, m)
#     if n is None:
#         s1 = (m,)
#
#     indices = torch.randint(1, 10, s1)
#     if hasattr(values, "shape") and size is not None:
#         size = size + tuple(values.shape)
#     matrix = scatter_coo_fill(indices, values, size=size)
#     if size is not None:
#         assert matrix.size() == size
#
#
# # def test_scatter_coo_explicit():
