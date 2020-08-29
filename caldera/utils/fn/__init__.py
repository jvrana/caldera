"""
fn.py

Functional programming methods
"""

import functools
from typing import Callable, Any
import inspect


def signature_args_len(f: Callable):
    return len(inspect.signature(f).parameters)


def curry(f: Callable, *cargs, **ckwargs) -> Any:
    """
    Creates flexible curried functions

    Usage:

    .. code-block:: python

        def foo(a, b, c, d=1):
            return (a+b+c0*d

        f1 = curry(foo)(5)
        f2 = f1(6, d=2)
        r = f2(4)
        assert r == (4 + 5 + 6)*2

    :param f:
    :return:
    """
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if len(args) == signature_args_len(f):
            return f(*args, **kwargs)
        else:
            return curry(functools.partial(f, *args, **kwargs))
    return wrapped(*cargs, **ckwargs)


def pipe(*funcs):
    """
    Create a functional pipe.

    :param funcs:
    :return:
    """
    def wrapped(data):
        result = data
        for f in funcs:
            result = f(result)
        return result
    return wrapped


def fn_partial(*args, **kwargs):
    """
    Partial function wrapper.

    Usage:

    .. code-block:: python

        fp = fn_partial(a=1)
        f1 = fp(foo1)
        f2 = fp(foo2)

    :param args:
    :param kwargs:
    :return:
    """
    def wrapped(f):
        return functools.partial(f, *args, **kwargs)
    return wrapped


def fn_curry_partial(*args, **kwargs):
    def wrapped(f):
        return curry(f)(*args, **kwargs)
    return wrapped