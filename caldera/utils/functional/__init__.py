"""functional.py.

Functional programming methods.

Generally, functions are named by `verb-noun-qualifier`

`verbs`

**ignore** - returned function ignores certain elements of the iterator, returning an iterator of the remaining
elements.
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

`compose`
"""
import functools
import itertools
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

T = TypeVar("T")
S = TypeVar("S")

GT = Generator[T, None, None]
GS = Generator[S, None, None]


def compose(*funcs: Tuple[Callable[[GT], GS], ...]) -> Callable[[GT], GS]:
    """Create a function composition.

    :param funcs:
    :return:
    """

    def wrapped(data):
        result = data
        for f in funcs:
            try:
                result = f(result)
            except Exception as e:
                msg = str(e)
                msg += "\nduring composition:"
                msg += "\n f: {}".format(f)
                raise type(e)(msg) from e
        return result

    return wrapped


def map_each(f: Callable[[T], S], *args, **kwargs) -> Callable[[GS], GT]:
    def wrapped(arr):
        for a in arr:
            yield f(a, *args, **kwargs)

    return wrapped


def map_all(f: Callable[[GT], GS], *args, **kwargs) -> Callable[[GT], GS]:
    """Apply the function to the iterator itself.

    .. note::     Of course, this could be done just as easily by
    supplying the provided     function itself. This keeps the syntax
    common with other methods.
    """

    def wrapped(arr: GT) -> GS:
        return f(arr, *args, **kwargs)

    return wrapped


def map_all_if(condition, if_func, else_func=None):
    def wrapped(arr):
        for a in arr:
            if condition(a):
                yield if_func(a)
            elif else_func is not None:
                yield else_func(a)
            else:
                yield a

    return wrapped


def zip_each_with(*arrs, first: bool = False):
    def _zip(a, *arrs):
        if first:
            return zip(a, *arrs)
        else:
            return zip(*arrs, a)

    return map_all(_zip, *arrs)


def zipmap_each_with(*funcs, first: bool = True):
    return compose(zip_each_with(funcs, first=first), map_each(lambda x: x[1](x[0])))


def iter_each_unique():
    def wrapped(arr):
        seen = set()
        for a in arr:
            if a not in seen:
                seen.add(a)
                yield a

    return wrapped


def iter_reverse():
    def wrapped(arr):
        return iter(list(arr)[::-1])

    return wrapped


def zip_all():
    def wrapped(arr):
        return zip(*arr)

    return wrapped


def reduce_each(f):
    def _reduce(arr):
        return functools.reduce(f, arr)

    return map_each(_reduce)


def reduce_all(f):
    def _reduce(arr):
        return functools.reduce(f, arr)

    return map_all(_reduce)


iter_map_items = functools.partial(map_each, zipped=True)


def group_each_consecutive(condition: Callable[[T], bool]):
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


def group_each_into_chunks(
    chunk_size: int,
) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    """Returns a new function the groups iterables by chunks of the specified
    size. Last chunk is returned by default.

    .. doctest::

        >>> from bentham.functional import group_each_into_chunks
        >> a = [1, 2, 3, 4]
        >> f = fn_chunks(3)
        >> print(f(a))
        [[1, 2, 3], [4]]

    :param chunk_size: size of each chunk
    :return: generator of chunks
    """
    return group_each_until(lambda g: len(g) >= chunk_size and len(g), yield_final=True)


def _validate_positive_int(n: int):
    if n is not None:
        if not isinstance(n, int):
            raise TypeError(
                "`n` must be an integer greater than 0 or None, not {}".format(n)
            )
        elif n < 0:
            raise ValueError(
                "`n` must be an integer greater than 0 or None, not {}".format(n)
            )


def repeat_all(n: Optional[int]) -> Callable[[GT], GT]:
    if n is not None:
        _validate_positive_int(n)

    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        c = itertools.tee(arr, 1)[0]
        i = 0
        while n is None or i < n:
            b, c = itertools.tee(c)
            yield from b
            i += 1

    return wrapped


def counter(i: Optional[int] = 0) -> Generator[int, None, None]:
    while True:
        yield i
        i += 1


def enumerate_each():
    """Non-tee version of enumerate."""
    return zip_each_with(counter())


_enumerate = enumerate_each()


def iter_count(n: int = 1) -> Callable[[GT], GT]:
    if n == 0:
        return empty_generator
    _validate_positive_int(n)
    return compose(
        enumerate_each(), until(lambda x: x[0] < n - 1, yield_last=True), index_each(1)
    )


def iter_next():
    return iter_count(1)


def consume(f: Callable[[GT], Any]) -> Callable[[GT], GT]:
    """Applies a function to the iterator and then consumes.

    :param f:
    :return:
    """

    def wrapped(arr):
        list(f(arr))
        return arr

    return wrapped


def empty_generator(*args):
    return
    yield


def tee_pipe(*funcs: Tuple[Callable[[GT], GS], ...]) -> Callable[[GT], GT]:
    """Tee the iterator and apply a new series of functions without affecting
    the primary iterator.

    .. code-blocK:

        def iter_print(arr):
            for a in arr:
                print(a)
                yield a

        func = functional.pipe(
            functional.side_effect(iter_print, functional.map_each(lambda x: x**2), iter_print),
            functional.map_each(lambda x: x**2)
        )

    :return:
    """

    piped = compose(*funcs)

    def _wrapped(arr):
        a, b = itertools.tee(arr)
        piped(a)
        return b

    return _wrapped


def tee_pipe_yield(*funcs):
    piped = compose(*funcs)
    def _wrapped(arr):
        a, b = itertools.tee(arr)
        yield piped(a)
        yield b
    return _wrapped


def asterisk(f):
    def wrapped(arr):
        return f(*arr)
    return wrapped


def tee_consume(*funcs):
    piped = compose(*funcs)

    def _wrapped(arr):
        list(piped(arr))
        return arr

    return _wrapped


def apply_each(func):
    def wrapped(arr):
        for a in arr:
            func(a)
            yield a

    return wrapped


def index_each(i: int) -> Callable[[GT], GS]:
    return map_each(lambda x: x[i])


def get_each(i: int, default=...) -> Callable[[GT], GT]:
    if default is ...:
        return map_each(lambda x: x.get(i))
    else:
        return map_each(lambda x: x.get(i, default))


def return_first() -> Callable[[GT], T]:
    return compose(iter_count(1), return_all())


def return_as(rtype: Type[S]) -> Callable[[GT], Generator[S, None, None]]:
    return return_all(rtype)


def ignore_each_count(n: int = 1) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    """pipe( fn_side_effect(fn_next(n)) )

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


def yield_all(f: Callable = None):
    def wrapped(arr):
        if f:
            yield f(arr)
        else:
            yield arr

    return wrapped


def ignore_each_until(
    f: Callable[[T], bool]
) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        start = False
        for a in arr:
            if not start and f(a):
                start = True
            if start:
                yield a

    return wrapped


def iter_each_until(
    f: Callable[[T], bool]
) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        for a in arr:
            if f(a):
                yield a
                return
            yield a

    return wrapped


def iter_step(step_size: int = 1) -> Callable[[Iterable[T]], Generator[T, None, None]]:
    _validate_positive_int(step_size)
    return compose(
        enumerate_each(), filter_each(lambda x: x[0] % step_size == 0), index_each(1)
    )


def tee_all(n: int = 2):
    def wrapped(arr: Iterable[T]) -> Generator[T, None, None]:
        yield from itertools.tee(arr, n)

    return wrapped


def trycatch(*exceptions, catch_yields=...):
    """Catches the provided exceptions. If one of the exceptions is raised,
    will stop iteration. If provided with `catch_yields`, the value will be
    yielded upon exception.

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


def pairwise_each() -> Callable[[GT], Generator[Tuple[T, T], None, None]]:
    def wrapped(arr: GT) -> Generator[Tuple[T, T], None, None]:
        a, b = itertools.tee(arr)
        next(b)
        return zip(a, b)

    return wrapped


def filter_each(f: Callable[[T], bool], inverse: bool = False) -> Callable[[GT], GT]:
    def wrapped(arr: GT) -> GT:
        for a in arr:
            v = f(a)
            if not inverse and v:
                yield a
            elif inverse and not v:
                yield a

    return wrapped


def until(
    f: Callable[[T], bool], inverse: bool = False, yield_last: bool = False
) -> Callable[[GT], GT]:
    def wrapped(arr: GT) -> GT:
        for a in arr:
            v = f(a)
            if not inverse and v:
                yield a
            elif inverse and not v:
                yield a
            else:
                if yield_last:
                    yield a
                return
        return wrapped

    return wrapped


def chain_each() -> Callable[[Generator[GT, None, None]], GT]:
    def wrapped(arr: Generator[GT, None, None]) -> GT:
        for a in arr:
            yield from a
    return wrapped


def group_each_by_key(f: Callable[[T], Any]) -> Callable[[GT], Generator[Iterable[T], None, None]]:
    def wrapped(arr):
        a, b = itertools.tee(arr)
        data = OrderedDict()
        for _a in a:
            k = f(_a)
            data.setdefault(k, list())
            data[k].append(_a)
        yield from data.items()
    return wrapped
