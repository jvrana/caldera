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
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar


T = TypeVar("T")
S = TypeVar("S")

GT = Generator[T, None, None]
GS = Generator[S, None, None]


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


class Functional:
    @staticmethod
    def compose(*funcs: Tuple[Callable[[GT], GS], ...]) -> Callable[[GT], GS]:
        """Create a function composition.

        :param funcs:
        :return:
        """

        def _compose(data):
            result = data
            for i, f in enumerate(funcs):

                def _composition_part(r):
                    try:
                        return f(result)
                    except Exception as e:
                        raise e
                        msg = str(e)
                        msg += "\nduring composition:"
                        msg += "\n ({}) f: {}".format(i, f)
                        msg += "\n args: {} {}".format(result, result.__class__)
                        print(msg)
                        raise type(e)(msg) from e

                result = _composition_part(result)
            return result

        return _compose

    @staticmethod
    def map_each(f: Callable[[T], S], *args, **kwargs) -> Callable[[GS], GT]:
        def _map_each(arr):
            for a in arr:
                yield f(a, *args, **kwargs)

        return _map_each

    @classmethod
    def map_all(cls, f: Callable[[GT], GS], *args, **kwargs) -> Callable[[GT], GS]:
        """Apply the function to the iterator itself.

        .. note::     Of course, this could be done just as easily by
        supplying the provided     function itself. This keeps the
        syntax common with other methods.
        """

        def _map_all(arr: GT) -> GS:
            return f(arr, *args, **kwargs)

        return _map_all

    @staticmethod
    def map_all_if(condition, if_func, else_func=None):
        def _map_all_if(arr):
            for a in arr:
                if condition(a):
                    yield if_func(a)
                elif else_func is not None:
                    yield else_func(a)
                else:
                    yield a

        return _map_all_if

    @classmethod
    def zip_each_with(cls, *arrs, first: bool = False):
        """.. warning::

            If provided with generators or generator functions,
            calling this function will consume all the provided iterables during
            iteration. The provided generators should never
            be used again elsewhere.

        :param arrs:
        :return:
        """

        def _zip_each_with(a):
            if first:
                yield from zip(a, *arrs)
            else:
                yield from zip(*arrs, a)

        return _zip_each_with

    @staticmethod
    def cat(*arrs):
        """.. warning::

            If provided with generators or generator functions,
            calling this function will consume all the provided iterables
            immediately. The provided generators should never
            be used again elsewhere.

        :param arrs:
        :return:
        """
        arrs = [list(a) for a in arrs]

        def _cat(a):
            yield a
            yield from arrs

        return _cat

    @classmethod
    def zipmap_each_with(cls, *funcs, first: bool = True):
        def _zipmap_each_with(arr):

            yield from cls.compose(
                cls.zip_each_with(funcs, first=first),
                cls.map_each(lambda x: x[1](x[0])),
            )(arr)

        return _zipmap_each_with

    @staticmethod
    def iter_each_unique():
        def _iter_each_unique(arr):
            seen = set()
            for a in arr:
                if a not in seen:
                    seen.add(a)
                    yield a

        return _iter_each_unique

    @staticmethod
    def iter_reverse():
        def _iter_reverse(arr):
            yield from iter(list(arr)[::-1])(arr)

        return _iter_reverse

    @staticmethod
    def zip_all(reverse: bool = False):
        def _zip_all(arr):
            if reverse:
                yield from zip(*list(arr)[::-1])
            else:
                yield from zip(*arr)

        return _zip_all

    @classmethod
    def reduce_each(cls, f):
        def reducer(arr):
            return functools.reduce(f, arr)

        def _reduce_each(arr):
            yield from cls.map_each(reducer)(arr)

        return _reduce_each

    @classmethod
    def reduce_all(cls, f):
        def reducer(arr):
            return functools.reduce(f, arr)

        def _reduce_all(arr):
            yield from cls.map_all(reducer)(arr)

        return _reduce_all

    # iter_map_items = functools.partial(map_each, zipped=True)

    @staticmethod
    def group_each_consecutive(condition: Callable[[T], bool]):
        def _group_each_consecutive(arr):
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

        return _group_each_consecutive

    @staticmethod
    def group_each_until(
        condition: Callable[[List[T]], bool],
        group_if: Optional[Callable[[T], bool]] = None,
        yield_final: bool = True,
    ) -> Callable[[GT], Generator[List[T], None, None]]:
        def _group_each_until(arr):
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

        return _group_each_until

    @classmethod
    def group_each_into_chunks(
        cls, chunk_size: int
    ) -> Callable[[Iterable[T]], Generator[T, None, None]]:
        """Returns a new function the groups iterables by chunks of the
        specified size. Last chunk is returned by default.

        .. doctest::

            >>> from bentham.functional import group_each_into_chunks
            >> a = [1, 2, 3, 4]
            >> f = fn_chunks(3)
            >> print(f(a))
            [[1, 2, 3], [4]]

        :param chunk_size: size of each chunk
        :return: generator of chunks
        """

        def _group_each_into_chunks(arr):
            yield from cls.group_each_until(
                lambda g: len(g) >= chunk_size and len(g), yield_final=True
            )(arr)

        return _group_each_into_chunks

    @staticmethod
    def repeat_all(n: Optional[int]) -> Callable[[GT], GT]:
        if n is not None:
            _validate_positive_int(n)

        def _repeat_all(arr: Iterable[T]) -> Generator[T, None, None]:
            c = itertools.tee(arr, 1)[0]
            i = 0
            while n is None or i < n:
                b, c = itertools.tee(c)
                yield from b
                i += 1

        return _repeat_all

    @staticmethod
    def counter(i: Optional[int] = 0) -> Generator[int, None, None]:
        while True:
            yield i
            i += 1

    @classmethod
    def enumerate_each(cls, *, reverse: bool = False):
        """Non-tee version of enumerate."""

        def _enumerate_each(arr):
            if not reverse:
                for i, a in enumerate(arr):
                    yield i, a
            else:
                for i, a in enumerate(arr):
                    yield a, i

        return _enumerate_each

    @classmethod
    def iter_count(cls, n: int = 1):
        if n == 0:
            return cls.empty_generator
        _validate_positive_int(n)

        def _iter_count(arr):
            yield from cls.compose(
                cls.enumerate_each(),
                cls.until(lambda x: x[0] < n - 1, yield_last=True),
                cls.index_each(1),
            )(arr)

        return _iter_count

    @classmethod
    def iter_next(cls):
        return cls.iter_count(1)

    @staticmethod
    def consume(f: Callable[[GT], Any]) -> Callable[[GT], GT]:
        """Applies a function to the iterator and then consumes.

        :param f:
        :return:
        """

        def _consume(arr):
            list(f(arr))
            return arr

        return _consume

    @staticmethod
    def empty_generator(*_):
        return
        yield

    @classmethod
    def tee_pipe(cls, *funcs: Tuple[Callable[[GT], GS], ...]) -> Callable[[GT], GT]:
        """Tee the iterator and apply a new series of functions without
        affecting the primary iterator.

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

        piped = cls.compose(*funcs)

        def _tee_pipe(arr):
            a, b = itertools.tee(arr)
            piped(a)
            return b

        return _tee_pipe

    @classmethod
    def tee_pipe_yield(cls, *funcs):
        piped = cls.compose(*funcs)

        def _tee_pipe_yield(arr):
            a, b = itertools.tee(arr)
            yield piped(a)
            yield b

        return _tee_pipe_yield

    @classmethod
    def tee_consume(cls, *funcs):
        piped = cls.compose(*funcs)

        def _tee_consume(arr):
            list(piped(arr))
            return arr

        return _tee_consume

    @staticmethod
    def apply_each(func):
        def _apply_each(arr):
            for a in arr:
                func(a)
                yield a

        return _apply_each

    @classmethod
    def index_each(cls, i: int) -> Callable[[GT], GS]:
        def _index_each(arr):
            yield from cls.map_each(lambda x: x[i])(arr)

        return _index_each

    @classmethod
    def index_from(cls, d: Dict) -> Callable[[GT], GS]:
        def _index_from(arr):
            yield from cls.map_each(lambda x: d[x])(arr)

        return _index_from

    @classmethod
    def get_each(cls, i: int, default=...) -> Callable[[GT], GT]:
        def _get_each(arr):
            if default is ...:
                yield from cls.map_each(lambda x: x.get(i))(arr)
            else:
                yield from cls.map_each(lambda x: x.get(i, default))(arr)

        return _get_each

    @classmethod
    def getattr_each(cls, k: str):
        def _getattr_each(arr):
            yield from cls.map_each(lambda x: getattr(x, k))(arr)

        return _getattr_each

    @classmethod
    def get_from(cls, d: Dict, default=...) -> Callable[[GT], GT]:
        def _get_from(arr):
            if default is ...:
                yield from cls.map_each(lambda x: d.get(x))(arr)
            else:
                yield from cls.map_each(lambda x: d.get(x, default))(arr)

        return _get_from

    @staticmethod
    def _reverse(x):
        if isinstance(x, tuple):
            return tuple(list(x)[::-1])
        elif isinstance(x, list):
            return x[::-1]
        else:
            return list(x)[::-1]

    @classmethod
    def reverse_each(cls):
        def _reverse_each(arr):
            for a in arr:
                yield cls._reverse(a)

        return _reverse_each

    @staticmethod
    def ignore_each_count(
        n: int = 1,
    ) -> Callable[[Iterable[T]], Generator[T, None, None]]:
        """pipe( fn_side_effect(fn_next(n)) )

        :param n:
        :return:
        """

        def _ignore_each_count(arr: Iterable[T]) -> Generator[T, None, None]:
            for i, a in enumerate(arr):
                if i >= n:
                    yield a

        return _ignore_each_count

    @staticmethod
    def yield_all(*funcs: Tuple[Callable, ...]):
        def _yield_all(arr):
            if not funcs:
                yield arr
            else:
                for _f in funcs:
                    yield _f(arr)

        return _yield_all

    @staticmethod
    def ignore_each_until(
        f: Callable[[T], bool]
    ) -> Callable[[Iterable[T]], Generator[T, None, None]]:
        def _ignore_each_until(arr: Iterable[T]) -> Generator[T, None, None]:
            start = False
            for a in arr:
                if not start and f(a):
                    start = True
                if start:
                    yield a

        return _ignore_each_until

    @staticmethod
    def iter_each_until(
        f: Callable[[T], bool]
    ) -> Callable[[Iterable[T]], Generator[T, None, None]]:
        def _iter_each_until(arr: Iterable[T]) -> Generator[T, None, None]:
            for a in arr:
                if f(a):
                    yield a
                    return
                yield a

        return _iter_each_until

    @staticmethod
    def iter_step(
        step_size: int = 1,
    ) -> Callable[[Iterable[T]], Generator[T, None, None]]:
        _validate_positive_int(step_size)

        def _iter_step(arr):
            for i, a in enumerate(arr):
                if i % step_size == 0:
                    yield a

        return _iter_step

    @staticmethod
    def tee_all(n: int = 2):
        def _tee_all(arr: Iterable[T]) -> Generator[T, None, None]:
            yield from itertools.tee(arr, n)

        return _tee_all

    @staticmethod
    def trycatch(*exceptions, catch_yields=...):
        """Catches the provided exceptions. If one of the exceptions is raised,
        will stop iteration. If provided with `catch_yields`, the value will be
        yielded upon exception.

        :param exceptions:
        :param catch_yields:
        :return:
        """

        def _trycath(arr):
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

        return _trycath

    @staticmethod
    def pairwise_each() -> Callable[[GT], Generator[Tuple[T, T], None, None]]:
        def _pairwise_each(arr: GT) -> Generator[Tuple[T, T], None, None]:
            a, b = itertools.tee(arr)
            next(b)
            return zip(a, b)

        return _pairwise_each

    @staticmethod
    def filter_each(
        f: Callable[[T], bool], inverse: bool = False
    ) -> Callable[[GT], GT]:
        def _filter_each(arr: GT) -> GT:
            for a in arr:
                v = f(a)
                if not inverse and v:
                    yield a
                elif inverse and not v:
                    yield a

        return _filter_each

    @staticmethod
    def until(
        f: Callable[[T], bool], inverse: bool = False, yield_last: bool = False
    ) -> Callable[[GT], GT]:
        def _until(arr: GT) -> GT:
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

        return _until

    @staticmethod
    def chain_each() -> Callable[[Generator[GT, None, None]], GT]:
        def _chain_each(arr: Generator[GT, None, None]) -> GT:
            for a in arr:
                yield from a

        return _chain_each

    @staticmethod
    def group_each_by_key(
        f: Callable[[T], S]
    ) -> Callable[[GT], Generator[Tuple[S, T], None, None]]:
        def _group_each_by_key(arr):
            a, b = itertools.tee(arr)
            data = OrderedDict()
            for _a in a:
                k = f(_a)
                data.setdefault(k, list())
                data[k].append(_a)
            yield from data.items()

        return _group_each_by_key

    @staticmethod
    def star(f):
        """Return a function equivalent to `lambda args: f(*args)`

        :param f:
        :return:
        """

        def _star(arr):
            return f(*arr)

        return _star

    @staticmethod
    def count_each():
        def _count_each(arr):
            for i, arr in enumerate(arr):
                yield i

        return _count_each

    @staticmethod
    def apply(func):
        def _apply(arr):
            func(arr)
            return arr

        return _apply
