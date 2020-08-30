"""
fn.py

Functional programming methods.

Generally, functions are named by `verb-noun-qualifier`

`verbs`

**ignore** - returned function ignores certain elements of the iterator, returning an iterator of the remaining elements.
**return** - returns the result of the function, rather than another iterator.
**group** - returned function returns an iterator of groups (i.e. iterables of each element `T`, such as `List[T]`),
instead of
**map** - applies a new function to elements of the iterator

`nouns`

**each** - returned function applies to each element in the iterator
**all** - returned function applies to the entire iterator, rather than to each element

`qualifier`

**until** - iterates until some condition is met

**Special Methods (i.e. piping methods)**

`pipe`
`side_effect`

"""
import functools
from typing import Callable, Any, Iterable, Generator, TypeVar, Tuple
import itertools
import inspect

T = TypeVar('T')

def signature_args_len(f: Callable):
    sig = inspect.signature(f)
    return len(inspect.signature(f).parameters)


def flex_curry(f: Callable, *cargs, **ckwargs) -> Any:
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
        num_args = signature_args_len(f)
        if len(args) == num_args:
            return f(*args, **kwargs)
        else:
            return curry(functools.partial(f, *args, **kwargs))
    if cargs or ckwargs:
        return wrapped(*cargs, **ckwargs)
    return wrapped


def curry(f: Callable):
    if signature_args_len(f) == 1:
        return f
    else:
        @functools.wraps(f)
        def wrapped(arg):
            return curry(functools.partial(f, arg))
        return wrapped

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


def fpartial(*args, **kwargs):
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

def curry_partial(*args, **kwargs):
    def wrapped(f):
        return curry(f)(*args, **kwargs)
    return wrapped

# pipe(
#     fn.tee_all(),
#     fn.enumerate_each(),
#     fn.zip_with([
#
#         lambda x: x
#     ])
#     fn.condition_all(lambda x: x[0] == 0)(
#         fn.ignore_count
#     )
# )

def map_each(f, *args, **kwargs):
    def wrapped(arr):
        for a in arr:
            yield f(a, *args, **kwargs)
    return wrapped

def map_all(f, *args, **kwargs):
    """Apply the function to the iterator itself.

    .. note::
        Of course, this could be done just as easily by supplying the provided
        function itself. This keeps the syntax common with other methods."""
    def wrapped(arr):
        return f(arr, *args, **kwargs)
    return f

def map_all_if(condition, if_func, else_func = None):
    def wrapped(arr):
        for a in arr:
            if condition(a):
                yield if_func(a)
            elif else_func is not None:
                yield else_func(a)
            else:
                yield a
    return wrapped

def zip_each_with(*arrs):
    def _zip(a):
        return zip(a, *arrs)
    return map_all(_zip, *arrs)


def zipmap_each_with(*funcs):
    return pipe(
        zip_each_with(*funcs),
        map_each(lambda x: x[1](x[0]))
    )


def iter_each_unique():
    def wrapped(arr):
        seen = set()
        for a in arr:
            if a not in seen:
                seen.add(a)
                yield a
    return wrapped


def zip_all():
    def wrapped(arr):
        return zip(*arr)
    return wrapped


def reduce_each(f):
    return map_each(curry(functools.reduce(f)))


def reduce_all(f):
    return map_all(curry(functools.reduce(f)))


iter_map_items = functools.partial(map_each, zipped=True)


def group_each_consecutive(condition):

    def wrapped(arr):
        chunk = []
        v = ...
        for a in arr:
            new_v = condition(a)
            if v is ... or new_v == v:
                chunk.append(a)
            else:
                yield chunk
                chunk = [a]
            v = new_v
        if len(chunk):
            yield chunk

    return wrapped


def group_each_until(condition, group_if=None, yield_final: bool = True):
    def wrapped(arr):
        chunk = []
        for a in arr:
            if group_if:
                if group_if(a):
                    chunk.append(a)
            else:
                chunk.append(a)

            if condition(chunk):
                yield chunk
                chunk = []

        if yield_final or condition(chunk):
            yield chunk

    return wrapped


def group_each_into_chunks(chunk_size: int) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    """
    Returns a new function the groups iterables by chunks of the specified size.
    Last chunk is returned by default.

    .. doctest::

        >>> from bentham.fn import group_each_into_chunks
        >> a = [1, 2, 3, 4]
        >> f = fn_chunks(3)
        >> print(f(a))
        [[1, 2, 3], [4]]

    :param chunk_size: size of each chunk
    :return: generator of chunks
    """
    return group_each_until(lambda g: len(g) >= chunk_size and len(g), yield_final=True)


def repeat_all(n: int):
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        return itertools.chain(itertools.tee(arr, n))
    return wrapped


def enumerate_each():
    """Non-tee version of enumerate"""
    def wrapped(arr: Iterable[T]) -> Generator[Tuple[int, T], None, None]:
        i = 0
        for a in arr:
            yield (i, a)
            i += 1
    return wrapped

_enumerate = enumerate_each()


def iter_count(n: int = 1):
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        for i, a in _enumerate(arr):
            yield a
            if i >= n:
                return
    return wrapped


def fn_apply_effect(f: Callable[[Iterable[T]], Any]) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    """
    Apply function as a side effect.

    :param f:
    :return:
    """
    def wrapped(arr):
        result = f(arr)
        result = list(result)
        return arr
    return wrapped


def side_effect(*funcs):
    """
    Tee the iterator and apply a new series of functions without affecting
    the primary iterator.

    .. code-blocK:

        def iter_print(arr):
            for a in arr:
                print(a)
                yield a

        func = fn.pipe(
            fn.side_effect(iter_print, fn.map_each(lambda x: x**2), iter_print),
            fn.map_each(lambda x: x**2)
        )

    :return:
    """
    def wrapped(*pipe_funcs):
        piped = pipe(pipe_funcs)
        def _wrapped(arr):
            a, _ = itertools.tee(arr)
            piped(a)
            list(piped)
            yield arr
        return _wrapped
    return wrapped




def index_each(i, default=...):
    if default is not ...:
        return map_each(lambda x: x.get(i, default))
    return map_each(lambda x: x[i])


# def side_effect(f: Callable[[Iterable[T]], Any] = None, consumer: bool = True) -> Callable[[Iterable[T]], Generator[T, None, None]]:
#     """
#     Apply function as a side effect.
#
#     :param f:
#     :return:
#     """
#     if f is None:
#         f = lambda arr: iter(arr)
#
#     def wrapped(arr):
#         result = f(arr)
#         if consumer is True:
#             list(result)
#         elif callable(consumer):
#             consumer(result)
#         return arr
#     return wrapped


def return_first():
    def wrapped(arr):
        for a in arr:
            return a
    return wrapped


def return_as(rtype):
    def wrapped(arr):
        return rtype(arr)
    return wrapped


def ignore_each_count(n: int = 1) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    """

    pipe(
        fn_side_effect(fn_next(n))
    )
    :param n:
    :return:
    """
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        for i, a in _enumerate(arr):
            if i >= n:
                yield a
    return wrapped


def return_all(f: Callable = None):
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        if f:
            return f(arr)
        return arr
    return wrapped


def condition(cond):
    def wrapped(f):
        def _wrapped(arr):
            return f(arr)
        return _wrapped
    return wrapped

def yield_all(f: Callable = None):
    def wrapped(arr):
        if f:
            yield f(arr)
        yield arr
    return wrapped


def ignore_each_until(f: Callable[[T], bool]) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        start = False
        for a in arr:
            if not start and f(a):
                start = True
            if start:
                yield a
    return wrapped


def iter_each_until(f: Callable[[T], bool]) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        for a in arr:
            if f(a):
                return
            yield a
    return wrapped


def step(step_size: int = 1) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        for i, a in _enumerate(arr):
            if i % step_size == 0:
                yield a
    return wrapped


def tee_all(n: int = 2):

    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        for a in itertools.tee(arr, n):
            yield a

    return wrapped

def trycatch(*exceptions, catch_yields=...):
    """
    Catches the provided exceptions. If one of the exceptions is raised, will stop iteration.
    If provided with `catch_yields`, the value will be yielded upon exception.

    :param exceptions:
    :param catch_yields:
    :return:
    """
    def wrapped(arr):
        while True:
            try:
                yield next(arr)
            except Exception as e:
                for valid_exception in exceptions:
                    if isinstance(e, valid_exception):
                        if catch_yields is not ...:
                            yield catch_yields
                        return
                raise e
    return wrapped


def pairwise_each() -> Callable[[Iterable[T]], Generator[Tuple[T, T], None, None]]:
    def wrapped(arr: Iterable[T]) -> Generator[Tuple[T, T], None, None]:
        a, b = tee_all()
        next(b)
        return zip(a, b)
    return wrapped


def filter_each(f: Callable[[T], bool], inverse: bool = False) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        for a in arr:
            if not inverse and f(a):
                yield a
            elif inverse and not f(a):
                yield a
        return wrapped
    return wrapped


def chain_each():
    def wrapped(arr):
        for a in arr:
            for _a in a:
                yield _a
    return wrapped


from collections import OrderedDict


def group_each_by_key(f):

    def wrapped(arr):
        a, b = itertools.tee(arr)
        data = OrderedDict()
        for _a in a:
            k = f(_a)
            data.setdefault(k, list())
            data[k].append(_a)
        for x in data.items():
            yield x
    return wrapped

