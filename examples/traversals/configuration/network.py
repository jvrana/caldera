from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Tuple

import torch
from torch.nn import LeakyReLU

from .tools import ConfigObj


@dataclass
class LayerConfig(ConfigObj):
    """Configuration for an edge, node, or global graph layer."""

    size: int = 1
    depth: int = 1
    layer_norm: bool = "${network.layer_norm}"
    activation: Optional[str] = "${network.activation}"
    dropout: float = "${network.dropout}"

    @property
    def layers(self):
        return [self.size] * self.depth

@dataclass
class Connectivity(ConfigObj):

    x0_to_edge: bool = False
    x1_to_edge: bool = False
    g_to_edge: bool = False
    g_to_node: bool = False
    edge_to_node: bool = False
    edge_to_glob: bool = False
    node_to_glob: bool = False


@dataclass
class NetComponentConfig(ConfigObj):

    node: LayerConfig = field(default_factory=LayerConfig)
    edge: LayerConfig = field(default_factory=LayerConfig)
    glob: LayerConfig = field(default_factory=LayerConfig)


@dataclass
class NetConfig(ConfigObj):
    """Configuration for caldera Graph Network."""

    model: str = "Network"
    encode: NetComponentConfig = field(default_factory=NetComponentConfig)
    core: NetComponentConfig = field(default_factory=NetComponentConfig)
    out: NetComponentConfig = field(default_factory=NetComponentConfig)

    # defaults
    dropout: Optional[float] = None
    activation: Optional[str] = LeakyReLU.__name__
    layer_norm: bool = True

    aggregator_activation: str = LeakyReLU.__name__

    # architecture
    pass_global_to_edge: bool = True
    pass_global_to_node: bool = True
    connectivity: Connectivity = field(default_factory=Connectivity)

    # aggregators
    edge_block_to_node_aggregators: Tuple[str, ...] = tuple(["add"])
    global_block_to_node_aggregators: Tuple[str, ...] = tuple(["add"])
    global_block_to_edge_aggregators: Tuple[str, ...] = tuple(["add"])

    @staticmethod
    def get_activation(activation: str):
        if activation is None:
            return torch.nn.Identity
        return getattr(torch.nn, activation)
