import pytest
import itertools
from copy import copy as do_copy
from typing import Type, Tuple, Iterable, Union, List, Callable, Optional
from contextlib import AbstractContextManager
from functools import wraps
from inspect import signature


class IgnoreContextManager(AbstractContextManager):
    """Context manager for ignoring exceptions."""

    default_exceptions = (AssertionError,)

    def __init__(self, name, ignore: bool = False, exceptions: Tuple[Type[Exception], ...] = default_exceptions):
        if exceptions is None:
            exceptions = self.default_exceptions
        self.name = name
        self.ignore = ignore
        self.ignore_exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ignore:
            for e in self.ignore_exceptions:
                if isinstance(exc_val, e):
                    return True

    def __repr__(self):
        return "{}(ignore={})".format(self.name, self.ignore)


class ContextContainer(AbstractContextManager):
    """
    Container for keeping a list of ContextManagers.
    """

    def __init__(self, contexts: Iterable[IgnoreContextManager], mask: Tuple[bool, ...] = None, copy: bool = True):
        if copy:
            self.contexts = tuple(do_copy(c) for c in contexts)
        else:
            self.contexts = tuple(contexts)
        self.popped = None
        if mask:
            self.apply(mask)
        self.collected_exceptions = []

    def context_dict(self):
        return {c.name: c for c in self.contexts}

    def __getitem__(self, item):
        return self.context_dict()[item]

    def values(self):
        return list(itertools.product([True, False]), repeat=len(self.contexts))

    def apply(self, mask):
        for context, b in zip(self.contexts, mask):
            context.ignore = b

    def pop(self):
        c = self.contexts[0]
        self.contexts = tuple(list(self.contexts)[1:])
        return c

    def __enter__(self):
        self.popped = self.pop()
        return self.popped.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.popped is None:
            return
        res = self.popped.__exit__(exc_type, exc_val, exc_tb)
        if not res:
            self.collected_exceptions.append(exc_val)

    def raise_all(self):
        if self.collected_exceptions:
            s = '\n'.join([str(e) for e in self.collected_exceptions])
            raise Exception(s)

    @staticmethod
    def get_name(contexts):
        return 'case(' + ','.join([str(c.name) for c in contexts if c.ignore is False]) + ")"

    def __repr__(self):
        return self.get_name(pytest_contexts)


def override_signature(from_key, to_key):
    """Create a new function with the signature replaced"""

    def wrapped(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        # override signature
        sig = signature(f)
        new_params = []
        for p in sig.parameters.values():
            if p.name == from_key:
                p = p.replace(name=to_key)
            new_params.append(p)
        _wrapped.__signature__ = sig.replace(parameters=new_params)
        return _wrapped

    return wrapped


def change_key(from_key: str, to_key: str) -> Callable:
    """Decorator to swap the kwargs key of a function."""
    def wrapped(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):
            assert to_key not in kwargs
            if from_key in kwargs:
                kwargs[to_key] = kwargs[from_key]
                del kwargs[from_key]
            return f(*args, **kwargs)

        return _wrapped

    return wrapped


def signature_swap(function_key: str, fixture_key: str) -> Callable:
    """Perform a sneaky signature swap for a function so that indirect fixtures may be applied to the function
    with custom names."""
    def wrapped(f):
        return override_signature(function_key, fixture_key)(change_key(fixture_key, function_key)(f))

    return wrapped


@pytest.fixture
def _context_manager_test_cases(request):
    """Create new context managers for test cases"""
    return ContextContainer(*request.param)

choices = [
    'accumulate',
    'expand'
]

def pytest_contexts(n,
                    case_names: Union[List[str], Tuple[str, ...]],
                    ids=None,
                    exceptions: Optional[Tuple[Type[Exception], ...]] = None,
                    mode: str = choices[0]):
    """
    Decorator to provide pytest with test cases.

    Usage:

    .. code-block:: python

        @contexts('cases', ['check_x', 'check_y'])
        def test_check(cases):
            with cases:
                pass # testing code for first case ('check_x'). Implicitly `pops` the context manager.
            # test code that is always rund
            with cases:
                pass # testing code for second case ('check_y'). Implicitly `pops` the context manager.

    .. code-block:: python

        @contexts('cases', ['check_x', 'check_y'])
        def test_check(cases):
            with cases['check_x']:
                pass # testing code for first case ('check_x')
            # test code that is always rund
            with cases['check_y']:
                pass # testing code for second case ('check_y')


    :param n:
    :param case_names:
    :param ids:
    :return:
    """
    assert mode in choices
    context_list = tuple([IgnoreContextManager(n, exceptions=exceptions) for n in case_names])
    if mode == 'expand':
        masks = list(itertools.product([True, False], repeat=len(context_list)))
    else:
        masks = [True] * len(context_list)
    values = list(itertools.product(
        [context_list] * len(case_names),
        masks
    ))

    if ids is None:
        def ids(args):
            context_list, mask = args
            return 'case(' + ','.join([str(c.name) for c, m in zip(context_list, mask) if m is False]) + ")"

    wrapper = pytest.mark.parametrize(
        _context_manager_test_cases.__name__, values, ids=ids, indirect=True
    )

    def conclude(f):
        @wraps(f)
        def _wrapped(*args, **kwargs):
            result = f(*args, **kwargs)
            for v in list(kwargs.values()) + list(args):
                if hasattr(v, 'raise_all'):
                    v.raise_all()
            return result
        return _wrapped

    def new_wrapper(f):
        return wrapper(conclude(signature_swap(n, _context_manager_test_cases.__name__)(f)))

    return new_wrapper