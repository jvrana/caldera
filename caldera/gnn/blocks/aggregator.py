from abc import ABC
from functools import wraps
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch_scatter
from torch import nn

from caldera.defaults import CalderaDefaults as D

AggregatorCallableType = Callable[
    [torch.FloatTensor, torch.LongTensor], torch.FloatTensor
]


@wraps(torch_scatter.scatter_max)
def scatter_max(*args, **kwargs):
    return torch_scatter.scatter_max(*args, **kwargs)[0]


@wraps(torch_scatter.scatter_min)
def scatter_min(*args, **kwargs):
    return torch_scatter.scatter_min(*args, **kwargs)[0]


class AggregatorBase(nn.Module, ABC):
    """Aggregation layer."""

    valid_aggregators = {
        "mean": torch_scatter.scatter_mean,
        "max": scatter_max,
        "min": scatter_min,
        "add": torch_scatter.scatter_add,
    }

    @classmethod
    def _resolve_aggregator(cls, aggregator: Union[str, AggregatorCallableType]):
        if aggregator not in cls.valid_aggregators and not callable(aggregator):
            raise ValueError(
                "Aggregator '{}' not not one of the valid aggregators {} or a callable of {}".format(
                    aggregator, list(cls.valid_aggregators), AggregatorCallableType
                )
            )
        if isinstance(aggregator, str):
            aggregator = cls.valid_aggregators[aggregator]
        return aggregator


# TODO: make aggregation selection trainable
class Aggregator(AggregatorBase):
    """Aggregation layer."""

    def __init__(
        self,
        aggregator: Union[str, AggregatorCallableType],
        dim: int = None,
        dim_size: int = None,
    ):
        """Initializes an aggregator function. Supported functions include:

        mean
            scatter_mean

        max
            scatter_max

        min
            scatter_min

        add
            scatter_add

        Callable
            A new callable can be provided

        :param aggregator: The aggregator function name.
                           Select from 'max', 'min', 'mean', or 'add'.
                           Alternatively, provide a new aggregating function.
        :param dim: Optional dimensions attribute to apply to the aggregating function.
                    See documentation on scatter functions for more information.
        :param dim_size: Optional dimension size to apply to the aggregating function.
                         See documentation on scatter functions for more information.
        """
        super().__init__()
        self.aggregator_alias = aggregator
        self.aggregator = self._resolve_aggregator(aggregator)
        self.kwargs = dict(dim=dim, dim_size=dim_size)

    def forward(self, x, indices, **kwargs):
        func_kwargs = dict(self.kwargs)
        func_kwargs.update(kwargs)
        result = self.aggregator(x, indices, **func_kwargs)
        return result

    @staticmethod
    @wraps(torch_scatter.scatter_max)
    def scatter_max(*args, **kwargs):
        return torch_scatter.scatter_max(*args, **kwargs)[0]

    @staticmethod
    @wraps(torch_scatter.scatter_min)
    def scatter_min(*args, **kwargs):
        return torch_scatter.scatter_min(*args, **kwargs)[0]

    def __str__(self):
        return "{}(func={})".format(self.__class__.__name__, self.aggregator_alias)

    def __repr__(self):
        return str(self)


# TODO: This is where the network decides the type of causal relationship.
#       Exposing this could help with network 'explainability'
#       You could apply a loss function so the network is encouraged
#       to pick a single explanation for the data.


class UniformSoftmax(torch.nn.Softmax):
    def __init__(self, *args, **kwargs):
        """Initializes a uniform softmax function such that the result is
        uniform for all rows in output dimension. This will sum the tensor
        along the provided dimension (default: 0), apply Softmax, and then
        broadcast the tensor to the original data shape.

        For example:

        .. code-block::

            from torch import nn

            x = torch.tensor([
                [0.1, 0.2, 0.3],
                [0.1, 0.0, 0.9]
            ])

            out = nn.UniformSoftmax(0)(x)
            print(out)
            # tensor([[0.2119, 0.2119, 0.5761],
            #         [0.2119, 0.2119, 0.5761]])

            out = nn.Softmax(0)(x)
            print(out)
            # tensor([[0.5000, 0.5498, 0.3543],
            #         [0.5000, 0.4502, 0.6457]])

        :param args: Softmax args
        :param kwargs: Softmax kwargs
        """
        super().__init__(*args, **kwargs)

    def forward(self, data):
        dim = self.dim or 0
        repeats = [1] * len(data.shape)
        repeats[dim] = data.shape[dim]
        data = data.sum(dim)
        data = super().forward(data)
        data = data.repeat(repeats)
        return data


class ArgMax(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, data):
        dim = self.dim
        out = torch.zeros(data.shape, requires_grad=True)
        return out.scatter(
            dim, data.max(dim).indices.unsqueeze(1), torch.ones(data.shape)
        )


# TODO: change terminology of Attention, is this 'really' attention???
class MultiAggregator(AggregatorBase):

    LOCAL_ATTENTION = "local"
    GLOBAL_ATTENTION = "global"

    def __init__(
        self,
        input_size: int,
        aggregators: Union[
            Tuple[Union[str, AggregatorCallableType], ...],
            List[Union[str, AggregatorCallableType]],
        ],
        dim: int = None,
        dim_size: int = None,
        activation=D.activation,
        attention: str = LOCAL_ATTENTION,
        hard_select: bool = False,
    ):
        """A differentiable and trainable way to select the aggregation
        function. This layer will feed data into data into a linear and
        activation function, apply the scatter_add function, and then a softmax
        function to get weights to apply to the stacked activation functions.

        This is used in the cases where it is uncertain which aggregation function
        to use. This module provides a trainable way to learn which of these functions
        to use and how to use them. This can be done in one of several *modes* of operation
        using the `attention` argument:

        `attention=local (default)`
            This will train an aggregation function that weights each function
            *per entry in the batch*. For example, if we had three aggregation
            functions (`max, min, mean`) and data of shape (5, 10), weights
            like the following may be produced:

            .. code-block::

                [
                    [0.4, 0.25, 0.35]
                    [0.33, 0.67, 0.0],
                    [0.41, 0.19, 0.4]
                    [0.33, 0.33, 0.34],
                    [0.4, 0.21, 0.39]
                ]

            With the row dimension being the same size as the data batch size and
            the column dimensions being the weights for each aggregation function.
            In the top row, if we have functions `max, min, mean` and the weights
            `[0.4, 0.2, 0.4]` means to apply 0.4max + 0.25min + 0.35mean
            to the first entry in the input data. Note that weights for each
            row entry can be different, meaning the network is training
            to select the aggregating function based on the data itself.

        `attention=global (default)`
            This will train a classifier to weight the aggregation functions
            *for the entire batch*. For example, if we had three aggregation
            functions (`max, min, mean`) and data of shape (5, 10), weights
            like the following may be produced:

            .. code-block::

                [
                    [0.4, 0.25, 0.35]
                    [0.4, 0.25, 0.35],
                    [0.4, 0.25, 0.35],
                    [0.4, 0.25, 0.35],
                    [0.4, 0.25, 0.35],
                ]

            Fo all rows, if we have functions `max, min, mean` and the weights
            `[0.4, 0.2, 0.4]` means to apply 0.4max + 0.25min + 0.35mean
            to the first entry in the input data. Note that the weights
            are applied uniformly throughout the batch.

        `attention=local, hard_select=True`
            This will train a classifier to select a *single* aggregation functions
            *for each entry in the batch*. For example, if we had three aggregation
            functions (`max, min, mean`) the following weights may be produced.
            Note how only one aggregation function is selected, but the function
            may be different for each entry.

            .. code-block::

                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    ...
                ]

        `attention=global, hard_select=True (default)`
            This will train a classifier to select a *single* aggregation functions
            *for the entire batch*. For example, if we had three aggregation
            functions (`max, min, mean`) the following weights may be produced.
            Note how only one aggregation function is selected, but the function
            is the same.
            .. code-block::

                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    ...
                ]

        .. note::
            This will apply weights to the aggregator functions on a batch-by-batch
            bases. However, in some cases, one may want the module to pick a *single*
            aggregating function to use throughout the whole batch. In this case,
            set `per_batch=True` to apply weights uniformly through the whole batch.
            This will produce weights as in:



        :param input_size: Input size of the tensor to feed into the linear and activation layer
                           to produce `weights` to select the aggregating functions.
        :param aggregators: List of Aggregators. Select from 'max', 'min', 'mean', or 'add'.
                            Alternatively, provide a new aggregating function.
        :param dim: Optional dimensions attribute to apply to the aggregating function.
                    See documentation on scatter functions for more information.
        :param dim_size: Optional dimension size to apply to the aggregating function.
                         See documentation on scatter functions for more information.
        :param activation: The Activation function to apply after the linear layer
        :param uniform: If True, will train to learn a single set of weights to
                        apply to the aggregation functions. Default: false
        """
        super().__init__()
        self.weight_layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, len(aggregators)),
            activation(),
        )
        if attention == self.LOCAL_ATTENTION:
            self.attention_layer = torch.nn.Softmax(0)
        elif attention == self.GLOBAL_ATTENTION:
            self.attention_layer = UniformSoftmax(0)
        if hard_select:
            self.attention_layer = torch.nn.Sequential(self.attention_layer, ArgMax())
        self.aggregators: Dict[str, Aggregator] = torch.nn.ModuleDict(
            {str(agg): Aggregator(agg) for agg in aggregators}
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
        weights = self.weight_layers(x)

        # match shape of aggregated matrix
        scatter_weights = self.valid_aggregators["add"](weights, indices, **func_kwargs)
        scatter_weights = self.attention_layer(scatter_weights)

        # weight each 'function' by the learned weights
        return torch.sum(
            torch.mul(stacked, scatter_weights.expand(1, -1, -1).T), axis=0
        )

    def __str__(self):
        return "{}(aggregators={})".format(
            self.__class__.__name__, list(self.aggregators)
        )

    def __repr__(self):
        return str(self)
