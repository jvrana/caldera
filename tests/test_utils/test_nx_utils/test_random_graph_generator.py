from caldera.utils.nx.generators import random_graph

import random
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from omegaconf import MISSING


class IntNumber(ABC):
    """Abstract class castable to integer."""

    @abstractmethod
    def __int__(self):
        pass


class FloatNumber(ABC):
    """Abstract class castable to float."""

    @abstractmethod
    def __float__(self):
        pass


@dataclass
class Distribution(IntNumber, FloatNumber):
    low: Any = MISSING
    high: Optional[Any] = None
    name: str = MISSING


@dataclass
class Uniform(Distribution):
    """Uniform distribution class. Automatically type casts to float or int.

    .. code-block::

        x = Uniform(1, 10)
        for _ in range(10):
            print(float(x))
            print(int(x))

    .. code-block::

        x = Uniform(1.1)
        for _ in range(10):
            assert int(x) == 1
            assert float(x) == 1.1
    """

    low: float = 1
    high: Optional[float] = None
    name: str = "uniform"

    def validate(self):
        if self.high is not None:
            assert self.high >= self.low

    def __call__(self) -> float:
        if self.high is None:
            return self.low
        return random.random() * (self.high - self.low) + self.low

    def __float__(self):
        return self()

    def __int__(self):
        return int(float(self))


def test_random_graph():
    g = random_graph(Uniform(1, 100), density=0.3, weight=Uniform(1, 10))

    for w in g.edges(data='weight'):
        print(w)