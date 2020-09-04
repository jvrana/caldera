import random
from typing import overload
from typing import Tuple
from typing import Union


@overload
def _resolve_range(x: Union[Tuple[float, float], float]) -> float:
    pass


def _rand_float(a: float, b: float):
    return a + (b - a) * random.random()


@overload
def _resolve_range(x: Union[Tuple[float, float], float]) -> float:
    ...


def _resolve_range(x: Union[Tuple[int, int], int]) -> int:
    if isinstance(x, int) or isinstance(x, float):
        return x
    elif isinstance(x, tuple):
        if isinstance(x[0], int):
            return random.randint(*x)
        elif isinstance(x[0], float):
            return _rand_float(*x)
    else:
        raise TypeError
