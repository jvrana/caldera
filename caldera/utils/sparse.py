import torch
from scipy.sparse import coo_matrix


def torch_coo_to_scipy_coo(m: torch.sparse.Tensor) -> coo_matrix:
    data = m.values().numpy()
    indices = m.indices()
    return coo_matrix((data, (indices[0], indices[1])), tuple(m.size()))
