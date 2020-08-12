from functools import wraps
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch_scatter
from torch import nn

from caldera.defaults import CalderaDefaults as D


@wraps(torch_scatter.scatter_max)
def scatter_max(*args, **kwargs):
    return torch_scatter.scatter_max(*args, **kwargs)[0]


@wraps(torch_scatter.scatter_min)
def scatter_min(*args, **kwargs):
    return torch_scatter.scatter_min(*args, **kwargs)[0]


class AggregatorBase(nn.Module):
    """Aggregation layer."""

    valid_aggregators = {
        "mean": torch_scatter.scatter_mean,
        "max": scatter_max,
        "min": scatter_min,
        "add": torch_scatter.scatter_add,
    }


# TODO: make aggregation selection trainable
class Aggregator(AggregatorBase):
    """Aggregation layer."""

    def __init__(self, aggregator: str, dim: int = None, dim_size: int = None):
        super().__init__()

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

    def __repr__(self):
        return "{}(func='{}')".format(self.__class__.__name__, self.aggregator)


# TODO: This is where the network decides the type of causal relationship.
#       Exposing this could help with network 'explainability'
#       You could apply a loss function so the network is encouraged
#       to pick a single explanation for the data.


class MultiAggregator(AggregatorBase):
    def __init__(
        self,
        input_size: int,
        aggregators: Union[Tuple[str, ...], List[str]],
        dim: int = None,
        dim_size: int = None,
        activation_function=D.activation,
    ):
        """A differentiable and trainable way to select the aggregation
        function.

        :param input_size:
        :param aggregators:
        :param dim:
        :param dim_size:
        """
        super().__init__()
        for aggregator in aggregators:
            if aggregator not in self.valid_aggregators:
                raise ValueError(
                    "Aggregator '{}' not not one of the valid aggregators {}".format(
                        aggregator, self.valid_aggregators
                    )
                )
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, len(aggregators)), activation_function()
        )
        self.aggregators = torch.nn.ModuleDict(
            {agg: Aggregator(agg) for agg in aggregators}
        )

        self.kwargs = dict(dim=dim, dim_size=dim_size)

    def forward(self, x, indices, **kwargs):
        func_kwargs = dict(self.kwargs)
        func_kwargs.update(kwargs)

        # stack each aggregation function
        stacked = torch.stack(
            [agg(x, indices, **func_kwargs) for agg in self.aggregators.values()]
        )

        # get the weights
        weights = self.layers(x)

        # match shape of aggregated matrix
        scatter_weights = self.valid_aggregators["add"](weights, indices, **func_kwargs)

        # weight each 'function' by the learned weights
        return torch.sum(
            torch.mul(stacked, scatter_weights.expand(1, -1, -1).T), axis=0
        )
