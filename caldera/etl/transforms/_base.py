from abc import ABC
from abc import abstractmethod
from typing import Generator
from typing import List
from typing import overload
from typing import Tuple

import networkx as nx


class NetworkxTransformBase(ABC):
    @abstractmethod
    def transform(self, graphs):
        pass

    @overload
    def __call__(self, graph: List[nx.Graph]) -> List[nx.Graph]:
        ...

    @overload
    def __call__(self, graphs: Tuple[nx.Graph, ...]) -> Tuple[nx.Graph, ...]:
        ...

    @overload
    def __call__(self, graphs: nx.Graph) -> nx.Graph:
        ...

    def __call__(
        self, graphs: Generator[nx.Graph, None, None]
    ) -> Generator[nx.Graph, None, None]:
        if isinstance(graphs, nx.Graph):
            results = self.transform([graphs])
            return list(results)[0]
        else:
            results = self.transform(graphs)
        if isinstance(graphs, list):
            return list(results)
        elif isinstance(graphs, tuple):
            return tuple(results)
        return results
