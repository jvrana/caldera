import functools
import inspect
from typing import Callable
from typing import TypeVar
from typing import Union

T = TypeVar("T")
CurryCallable = Union[Callable[..., T], Callable[..., Callable[..., T]]]
CurryReturn = Union[CurryCallable, T]


def signature_args_len(f: Callable):
    return len(inspect.signature(f).parameters)


def flex_curry(f: CurryCallable, *cargs, **ckwargs) -> CurryReturn:
    """Creates flexible curried functions.

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
        num_args = signature_args_len(f)
        if len(args) == num_args:
            return f(*args, **kwargs)
        else:
            return curry(functools.partial(f, *args, **kwargs))

    if cargs or ckwargs:
        return wrapped(*cargs, **ckwargs)
    return wrapped


def curry(f: CurryCallable) -> CurryReturn:
    if signature_args_len(f) == 1:
        return f
    else:

        @functools.wraps(f)
        def wrapped(arg):
            return curry(functools.partial(f, arg))

        return wrapped
