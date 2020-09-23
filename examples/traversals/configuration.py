"""
Single file to store configuration methods, types, and classes
"""

from typing import Tuple, Optional, List
from omegaconf import MISSING
from omegaconf import DictConfig
from hydra.conf import ConfigStore
from omegaconf import OmegaConf
from dataclasses import dataclass, field


@dataclass
class GraphLayerConfig:
    """Configuration for an edge, node, or global graph layer"""

    size: int = 1
    depth: int = 1
    layer_norm: bool = "${network.layer_norm}"
    activation: str = "${network.activation}"
    dropout: str = "${network.activation}"
    layers: List[int] = field(default_factory=lambda: [1, 2])

@dataclass
class GraphNetConfig:
    """Configuration for caldera Graph Network"""

    node_encode: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    edge_encode: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    glob_encode: GraphLayerConfig = field(default_factory=GraphLayerConfig)

    node_core: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    edge_core: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    glob_core: GraphLayerConfig = field(default_factory=GraphLayerConfig)

    node_decode: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    edge_decode: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    glob_decode: GraphLayerConfig = field(default_factory=GraphLayerConfig)

    node_out: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    edge_out: GraphLayerConfig = field(default_factory=GraphLayerConfig)
    glob_out: GraphLayerConfig = field(default_factory=GraphLayerConfig)

    # defaults
    dropout: Optional[float] = None
    activation: Optional[str] = None
    layer_norm: bool = True

    aggregator_activation: str = "leakyrelu"

    # architecture
    pass_global_to_edge: bool = True
    pass_global_to_node: bool = True

    # aggregators
    edge_block_to_node_aggregators: Tuple[str, ...] = tuple()
    global_block_to_node_aggregators: Tuple[str, ...] = tuple()
    global_block_to_edge_aggregators: Tuple[str, ...] = tuple()


@dataclass
class Config:
    network: GraphNetConfig = field(default_factory=GraphNetConfig)


OmegaConf.register_resolver("mul", lambda x,y: [x]*y)


def initialize_config():
    """Initialize structured configuration"""
    cs = ConfigStore()
    cs.store(group="network", name="default", node=GraphNetConfig)
    cs.store(name="config", node=Config)


initialize_config()

__all__ = [
    "initialize_config",
    "Config",
    "GraphNetConfig",
    "GraphLayerConfig",
    "DictConfig",
    "MISSING",
    "OmegaConf",
]
