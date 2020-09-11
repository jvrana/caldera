from functools import wraps
from typing import Any
from typing import Dict
from typing import Generator
from typing import Tuple
from typing import Type
from typing import TypeVar

import torch

from caldera.exceptions import CalderaException

T = TypeVar("T")


class InvalidFlexZeroDimension(Exception):
    """Flex dimension cannot be 0."""


class InvalidFlexNegativeDimension(Exception):
    """Flex dimension cannot be negative."""


class ResolveError(Exception):
    """There was an error resolving the module."""


class FlexDim:
    def __init__(self, pos: int = 0, dim: int = 1):
        """Flexible dimension to be used in conjunction with `FlexBlock`

        :param pos: position of the input arguments that contains the input data
        :param dim: dimension to use for the input shape
        """
        self.pos = pos
        self.dim = dim

    # TODO: resolve input_kwargs for FlexDim?
    def resolve(self, input_args, input_kwargs):
        d = input_args[self.pos].shape[self.dim]
        if d == 0:
            raise InvalidFlexZeroDimension("Dimension cannot be zero.")
        if d < 0:
            raise InvalidFlexNegativeDimension("Dimension cannot be less than zero.")
        return d

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.pos, self.dim)


class FlexBlock(torch.nn.Module):
    """Flexible Block that is resolve upon calling of `forward` with an
    example."""

    def __init__(self, module_fn: Type[torch.nn.Module], *args, **kwargs):
        """A Flexible torch.nn.Module whose dimensions are resolved when
        provided with an example.

        :param module_fn:
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.module = module_fn
        self.args = args
        self.kwargs = kwargs
        self.resolved_module = None
        self._apply_history = None
        self.__resolved = False

    @property
    def is_resolved(self):
        """Returns whether this block has been resolved with an example. Note
        that before an example is presented, this block will have no
        parameters.

        :return: whether this block has been resolved
        """
        return self.__resolved

    def resolve_args(self, input_args: Tuple[Any, ...], input_kwargs: Dict[str, Any]):
        rargs = []
        for i, a in enumerate(self.args):
            if isinstance(a, FlexDim):
                rargs.append(a.resolve(input_args, input_kwargs))
            elif a is FlexDim:
                raise ValueError(
                    "Found {}. Initialize FlexDim to use flexible dimensions, `Flex.d()` or `FlexDim()`".format(
                        a
                    )
                )
            else:
                rargs.append(a)
        return rargs

    def resolve_kwargs(self, input_args: Tuple[Any, ...], input_kwargs: Dict[str, Any]):
        return self.kwargs

    def resolve(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        resolved_args = self.resolve_args(args, kwargs)
        resolved_kwargs = self.resolve_kwargs(args, kwargs)
        self.__resolved = True
        try:
            self.resolved_module = self.module(*resolved_args, **resolved_kwargs)
        except Exception as e:
            raise ResolveError(
                "There was an error resolving module {}: {}".format(self.module, str(e))
            ) from e
        if self._apply_history:
            self._play_apply()

    def forward(self, *args, **kwargs):
        if not self.is_resolved:
            self.resolve(args, kwargs)
        return self.resolved_module(*args, **kwargs)

    def _record_apply(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        """Records the `_apply` function to the `_apply_history` storage."""
        if self._apply_history is None:
            self._apply_history = []
        self._apply_history.append((args, kwargs))

    def _play_apply(self):
        """Plays back the `_apply` function from the `_apply_history`
        storage."""
        try:
            for args, kwargs in self._apply_history:
                super()._apply(*args, **kwargs)
        except Exception as e:
            raise CalderaException(
                "An error occurred while trying to "
                " replay `_apply` in a {}. Try resolving the module"
                " by providing an example before using"
                " methods that use `_apply` (such as `to()`)".format(self.__class__)
            ) from e
        self._apply_history = None

    def _apply(self, *args, **kwargs) -> None:
        """Special override method. In the case that `_apply` is called and
        this module has not been resolved, the arguments for the `_apply`
        function are stored. These are reapplied after they have been resolved.

        :param args:
        :param kwargs:
        :return:
        """
        if not self.is_resolved:
            self._record_apply(args, kwargs)
        else:
            super()._apply(*args, **kwargs)

    # def _get_name(self):
    #     return "{}({})".format(
    #         self.__class__.__name__,
    #         self.module.__name__
    #     )

    def __repr__(self):
        if self.is_resolved:
            return super().__repr__()
        else:
            s = "{c}(\n\t(unresolved_module): {m}({args}, {kwargs}\n)".format(
                c=self._get_name(),
                m=self.module.__name__,
                args=", ".join([str(a) for a in self.args]),
                kwargs=",".join(str(k) + "=" + str(v) for k, v in self.kwargs.items()),
            )
            return s


def _iter_modules_of_type(
    module: torch.nn.Module, module_type: Type[T]
) -> Generator[T, None, None]:
    for m in module.modules():
        if issubclass(m.__class__, module_type):
            yield m


def _iter_flex_blocks(module: torch.nn.Module) -> Generator[FlexBlock, None, None]:
    yield from _iter_modules_of_type(module, FlexBlock)


class Flex:
    """Flex."""

    d = FlexDim

    def __init__(self, module_type: Type[torch.nn.Module]):
        """Initialize a module as a FlexBlock with flexible dimensions.

        Usage:

        .. code-block::

            Flex(torch.nn.Linear)(Flex.d(), 25)

        :param module_type: module type (e.g. `torch.nn.Linear`
        """
        self.module_type = module_type

        self.__call__ = wraps(module_type.__init__)(self.__class__.__call__)

    def __call__(self, *args, **kwargs) -> FlexBlock:
        """Initialize the flexible module.

        :param args: the initialization arguments
        :param kwargs: the initialization keyword arguments
        :return: initialized torch.nn.Module
        """
        return FlexBlock(self.module_type, *args, **kwargs)

    @staticmethod
    def has_flex_blocks(module: torch.nn.Module):
        for m in _iter_flex_blocks(module):
            if issubclass(m.__class__, FlexBlock):
                return True

    @staticmethod
    def has_unresolved_flex_blocks(module: torch.nn.Module):
        for m in _iter_flex_blocks(module):
            if not m.is_resolved:
                return True
        return False
