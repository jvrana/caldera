from abc import ABC
from abc import abstractmethod


class NetworkxTransformBase(ABC):
    @abstractmethod
    def transform(self, graphs):
        pass

    def __call__(self, graphs):
        results = self.transform(graphs)
        if isinstance(graphs, list):
            return list(results)
        elif isinstance(graphs, tuple):
            return list(results)
        return results
