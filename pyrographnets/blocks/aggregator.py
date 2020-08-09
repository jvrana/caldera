from functools import wraps

import torch_scatter
from torch import nn


class Aggregator(nn.Module):
    """Aggregation layer."""

    def __init__(self, aggregator: str, dim: int = None, dim_size: int = None):
        super().__init__()

        self.valid_aggregators = {
            "mean": torch_scatter.scatter_mean,
            "max": self.scatter_max,
            "min": self.scatter_min,
            "add": torch_scatter.scatter_add,
        }

        if aggregator not in self.valid_aggregators:
            raise ValueError(
                "Aggregator '{}' not not one of the valid aggregators {}".format(
                    aggregator, self.valid_aggregators
                )
            )
        self.aggregator = aggregator
        self.kwargs = dict(dim=dim, dim_size=dim_size)

    def forward(self, x, indices, **kwargs):
        func_kwargs = dict(self.kwargs)
        func_kwargs.update(kwargs)
        func = self.valid_aggregators[self.aggregator]
        result = func(x, indices, **func_kwargs)
        return result

    @staticmethod
    @wraps(torch_scatter.scatter_max)
    def scatter_max(*args, **kwargs):
        return torch_scatter.scatter_max(*args, **kwargs)[0]

    @staticmethod
    @wraps(torch_scatter.scatter_min)
    def scatter_min(*args, **kwargs):
        return torch_scatter.scatter_min(*args, **kwargs)[0]