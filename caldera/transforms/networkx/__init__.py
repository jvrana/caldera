r"""
Data (:mod:`caldera.transform`)
===============================

.. currentmodule:: caldera.transforms.networkx

Methods for transforming :class:`networkx.Graph`s

.. autosummary::
    :toctree: generated/

    NetworkxApplyToFeature
    NetworkxAttachNumpyFeatures
    NetworkxAttachNumpyOneHot
    NetworkxSetDefaultFeature
    NetworkxTransformFeatureData
    NetworkxTransformFeatures
    NetworkxFlattenEdgeFeature
    NetworkxFlattenGlobalFeature
    NetworkxFlattenNodeFeature
    NetworkxNodesToStr
    NetworkxToDirected
    NetworkxToUndirected
    NetworkxFilterDataKeys
    NetworkxDeepCopyFeatures
"""
from ._nx_apply_to_graph import NetworkxApply
from ._nx_apply_to_key import NetworkxApplyToFeature
from ._nx_attach_np_features import NetworkxAttachNumpyFeatures
from ._nx_attach_np_features import NetworkxAttachNumpyOneHot
from ._nx_deepcopy import NetworkxDeepCopyFeatures
from ._nx_defaults import NetworkxSetDefaultFeature
from ._nx_feature_transform import NetworkxTransformFeatureData
from ._nx_feature_transform import NetworkxTransformFeatures
from ._nx_filter_keys import NetworkxFilterDataKeys
from ._nx_flatten_feature import NetworkxFlattenEdgeFeature
from ._nx_flatten_feature import NetworkxFlattenGlobalFeature
from ._nx_flatten_feature import NetworkxFlattenNodeFeature
from ._nx_nodes_to_str import NetworkxNodesToStr
from ._nx_to_directed import NetworkxToDirected
from ._nx_to_undirected import NetworkxToUndirected


__all__ = [
    "NetworkxAttachNumpyFeatures",
    "NetworkxAttachNumpyOneHot",
    "NetworkxSetDefaultFeature",
    "NetworkxTransformFeatureData",
    "NetworkxTransformFeatures",
    "NetworkxNodesToStr",
    "NetworkxToDirected",
    "NetworkxToUndirected",
    "NetworkxApplyToFeature",
    "NetworkxFlattenNodeFeature",
    "NetworkxFlattenEdgeFeature",
    "NetworkxFlattenGlobalFeature",
    "NetworkxFilterDataKeys",
    "NetworkxApply",
    "NetworkxDeepCopyFeatures",
]
