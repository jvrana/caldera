from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import TypeVar
from typing import Union

T = TypeVar("T")
S = TypeVar("S")
K = TypeVar("K")
V = TypeVar("V")


def dict_join(
    a: Dict[K, T],
    b: Dict[K, S],
    out: Optional[Dict] = None,
    join_fn: Callable[[T, S], V] = None,
    default_a: T = ...,
    default_b: S = ...,
    keys: Optional[Iterable[K]] = None,
    mode: str = "union",  # Literal["union", "left", "right", "intersection"]
) -> Dict[K, Union[T, S, V]]:
    """Join two dictionaries. This function merges two dictionaries is various
    ways. For example, a dictionary of `Dict[str, List]` can be merged such
    that, if the two dictionaries share the same key, the lists are
    concatenated. The join function can be applied to the union of all keys
    (default) by (`mode="union"`), the intersection of the dictionary
    (`mode="intersection"), only the keys in the left dictionary
    (`mode="left"`), or keys only in the right dictionary (`mode="right"`).

    .. code-block::

        import operator
        d1 = {'a': [1,2,3], 'b': [1,2]
        d2 = {'a': [1,2], 'c': [10,20]
        d3 = dict_join(d1, d2, join_fn=operator.add)
        print(d3)
        # {'a': [1,2,3,1,2], 'b': [1,2], 'c': [10,20]}

    This can be done such that the first dictionary is updated instead of returning a new dictionary:

    .. code-block::

        d1 = {'a': [1,2,3], 'b': [1,2]
        d2 = {'a': [1,2], 'c': [10,20]
        dict_join(d1, d2, d2, join_fn=lambda a, b: a + b
        print(d2)
        # {'a': [1,2,3,1,2], 'b': [1,2], 'c': [10,20]}

        functools.partial(dict_join, join_fv=operator.add

    :param a: First dictionary
    :param b: Second dictionary
    :param out: The target dictionary. If None, creates a new dictionary.
    :param join_fn: Join function when both dictionaries share the same key. If not provided, will use the value
        provided by the second dictionary.
    :param default_a: Default value to use in the case a key is missing from the first dictionary. If defined as
        `Ellipsis` (or `...`), defaults will be ignored.
    :param default_b: Default value to use in the case a key is missing from the second dictionary. If defined as
        `Ellipsis` (or `...`), defaults will be ignored.
    :param keys: If provided, explicitly join only on specified keys.
    :param mode: If keys are None, mode specifies which keys to join. "union" (default), means join
        takes the union of keys from both dictionaries. "intersection" means take intersection of keys
        for both dictionaries. "left" means use only keys in the first dictionary. "right" means use only keys
        from the second dictionary.
    :return:
    """
    if out is None:
        out = dict()

    if join_fn is None:

        def join_fn(v1, v2):
            if v2 is not ...:
                return v2
            if v1 is not ...:
                return v1

    if keys is None:
        if mode == "union":
            keys = set(a).union(set(b))
        elif mode == "intersection":
            keys = set(a).intersection(set(b))
        elif mode == "left":
            keys = set(a)
        elif mode == "right":
            keys = set(b)
        else:
            raise ValueError("mode '{}' not recognized.".format(mode))

    for k in keys:
        v1 = a.get(k, default_a)
        v2 = b.get(k, default_b)
        if v1 is ...:
            v3 = v2
        elif v2 is ...:
            v3 = v1
        else:
            v3 = join_fn(v1, v2)
        out[k] = v3
    return out
