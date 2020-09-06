import itertools
from typing import Iterable
from typing import Tuple
from typing import TypeVar

T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _first(i: Iterable[T]) -> T:
    """Select the first element in an iterable."""
    return next(x for x in itertools.tee(i)[0])
