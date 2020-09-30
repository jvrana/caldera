from ._nx_apply_to_key import NetworkxApplyToKey


class NetworkxFlattenNodeFeature(NetworkxApplyToKey):
    def __init__(self, key: str):
        super().__init__(key, node_func=lambda x: x.flatten())


class NetworkxFlattenEdgeFeature(NetworkxApplyToKey):
    def __init__(self, key: str):
        super().__init__(key, edge_func=lambda x: x.flatten())


class NetworkxFlattenGlobalFeature(NetworkxApplyToKey):
    def __init__(self, key: str):
        super().__init__(key, glob_func=lambda x: x.flatten())
