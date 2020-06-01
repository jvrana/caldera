import itertools
from pyrographnets.utils.scatter_group import scatter_group


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _first(i):
    """Select the first element in an iterable"""
    return next((x for x in itertools.tee(i)[0]))