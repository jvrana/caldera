from ._nx_feature_transform import NetworkxTransformFeatures
from caldera.utils import functional


class NetworkxNodesToStr(NetworkxTransformFeatures):
    def __init__(self):
        """Convert node keys to strings."""
        super().__init__(
            node_transform=functional.map_each(lambda x: (str(x[0]), x[1]))
        )
