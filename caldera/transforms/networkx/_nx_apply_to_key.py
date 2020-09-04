from typing import Callable
from typing import Dict
from typing import TypeVar
from typing import Union

from ._nx_feature_transform import NetworkxTransformFeatureData


T = TypeVar("T")
S = TypeVar("S")
K = TypeVar("K")


class NetworkxApplyToFeature(NetworkxTransformFeatureData):
    def __init__(
        self,
        key: str,
        node_func: Callable[[S], T] = None,
        edge_func: Callable[[S], T] = None,
        glob_func: Callable[[S], T] = None,
    ):
        """Transformation to apply a function to the a keyed value for a
        networkx graph.

        :param key: key to find the value to apply the function to.
        :param node_func:
        :param edge_func:
        :param glob_func:
        """

        def apply_to_key(f: Callable[[S], T]) -> Union[Callable[[Dict[K, S]], T], None]:
            if f is None:
                return None

            def _apply_to_key(d: Dict[K, S]) -> T:
                d[key] = f(d[key])

            return _apply_to_key

        super().__init__(
            node_transform=apply_to_key(node_func),
            edge_transform=apply_to_key(edge_func),
            global_transform=apply_to_key(glob_func),
        )
