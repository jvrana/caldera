from functools import wraps
from typing import Tuple, Any, Dict, Type

import torch


class FlexDim:
    def __init__(self, pos: int = 0, dim: int = 1):
        """Flexible dimension to be used in conjunction with `FlexBlock`

        :param pos: position of the input arguments that contains the input data
        :param dim: dimension to use for the input shape
        """
        self.pos = pos
        self.dim = dim

    def resolve(self, input_args, input_kwargs):
        d = input_args[self.pos].shape[self.dim]
        if d == 0:
            raise ValueError("Dimension cannot be zero")
        return d


class FlexBlock(torch.nn.Module):
    def __init__(self, module_fn, *args, **kwargs):
        super().__init__()
        self.module = module_fn
        self.args = args
        self.kwargs = kwargs
        self.resolved_module = None

    def resolve_args(self, input_args: Tuple[Any, ...], input_kwargs: Dict[str, Any]):
        rargs = []
        for i, a in enumerate(self.args):
            if isinstance(a, FlexDim):
                rargs.append(a.resolve(input_args, input_kwargs))
            elif a is FlexDim:
                raise ValueError("Found {}. Initialize FlexDim to use flexible dimensions, `Flex.d()` or `FlexDim()`".format(
                a))
            else:
                rargs.append(a)
        return rargs

    def resolve_kwargs(self, input_args: Tuple[Any, ...], input_kwargs: Dict[str, Any]):
        return self.kwargs

    def resolve(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        resolved_args = self.resolve_args(args, kwargs)
        resolved_kwargs = self.resolve_kwargs(args, kwargs)
        self.resolved_module = self.module(*resolved_args, **resolved_kwargs)

    def forward(self, *args, **kwargs):
        if self.resolved_module is None:
            self.resolve(args, kwargs)
        return self.resolved_module(*args, **kwargs)


class Flex:
    d = FlexDim

    def __init__(self, module_type: Type[torch.nn.Module]):
        """Initialize a module as a FlexBlock with flexible dimensions.

        Usage:

        .. code-block:: python

            Flex(torch.nn.Linear)(Flex.d(), 25)

        :param module_type: module type (e.g. `torch.nn.Linear`
        """
        self.module_type = module_type

        self.__call__ = wraps(module_type.__init__)(self.__class__.__call__)

    def __call__(self, *args, **kwargs) -> torch.nn.Module:
        """Initialize the flexible module.

        :param args: the initialization arguments
        :param kwargs: the initialization keyword arguments
        :return: initialized torch.nn.Module
        """
        return FlexBlock(self.module_type, *args, **kwargs)