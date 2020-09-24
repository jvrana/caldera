"""Single file to store configuration methods, types, and classes."""
import random
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

import torch
from hydra.conf import ConfigStore
from omegaconf import DictConfig
from omegaconf import MISSING
from omegaconf import OmegaConf
from torch.nn import LeakyReLU

C = TypeVar("C")


def dataclass_to_dict(dataklass: C) -> Dict[str, Any]:
    """Converts a dataclass to a nested dictionary.

    :param dataklass: The dataclass instance
    :return: dictionary
    """
    data = {}
    if is_dataclass(dataklass):
        for field in fields(dataklass):
            value = getattr(dataklass, field.name)
            if is_dataclass(value):
                value = dataclass_to_dict(value)
            data[field.name] = value
    return data


def dataclass_from_dict(dataklasstype: Type[C], data: Dict[str, Any]) -> C:
    """Converts a nested dictionary to a dataclass.

    :param dataklasstype: the dataclass
    :param data: nested data as dictionary
    :return: dataclass instance
    """
    if is_dataclass(dataklasstype):
        for field in fields(dataklasstype):
            if is_dataclass(field.type):
                value = dataclass_from_dict(field.type, data[field.name])
            else:
                value = data[field.name]
            data[field.name] = value
        return dataklasstype(**data)


def validate_all_accessible(cfg: DictConfig):
    for k in cfg:
        v = cfg[k]
        if isinstance(v, DictConfig):
            validate_all_accessible(v)


def validate_structured(klass: Type, cfg: DictConfig):
    OmegaConf.structured(klass).update(cfg)


class ConfigObj:
    """Generic configuraiton object that performs validation.

    Can convert from dataclass to DictConfig.
    """

    @classmethod
    def validate_cfg(cls, cfg: DictConfig):
        validate_structured(cls, cfg)
        validate_all_accessible(cfg)

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, ConfigObj):
                value.validate()

    @classmethod
    def from_dict_config(cls, cfg: DictConfig):
        cls.validate_cfg(cfg)
        return dataclass_from_dict(cls, OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return dataclass_from_dict(cls, data)

    def to_dict(self):
        return dataclass_to_dict(self)


@dataclass
class GraphLayerConfig(ConfigObj):
    """Configuration for an edge, node, or global graph layer."""

    size: int = 1
    depth: int = 1
    layer_norm: bool = "${network.layer_norm}"
    activation: str = "${network.activation}"
    dropout: float = "${network.dropout}"

    @property
    def layers(self):
        return [self.size] * self.depth


@dataclass
class GraphNetConfig(ConfigObj):
    """Configuration for caldera Graph Network."""

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
    activation: Optional[str] = LeakyReLU.__name__
    layer_norm: bool = True

    aggregator_activation: str = LeakyReLU.__name__

    # architecture
    pass_global_to_edge: bool = True
    pass_global_to_node: bool = True

    # aggregators
    edge_block_to_node_aggregators: Tuple[str, ...] = tuple(["add"])
    global_block_to_node_aggregators: Tuple[str, ...] = tuple(["add"])
    global_block_to_edge_aggregators: Tuple[str, ...] = tuple(["add"])

    @staticmethod
    def get_activation(activation: str):
        return getattr(torch.nn, activation)


@dataclass
class HyperParameters(ConfigObj):

    lr: float = 1e-3  #: the learning rate
    train_core_processing_steps: int = (
        10  #: number of core processing steps for training step
    )
    eval_core_processing_steps: int = (
        10  #: number of core processing steps for evaluation step
    )


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
class Distribution(ConfigObj, IntNumber, FloatNumber):
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


@dataclass
class DiscreteUniform(Uniform):
    """Discrete uniform distribution class. Automatically type casts to float
    or int.

    .. code-block::

        x = Uniform(1, 10)
        for _ in range(10):
            print(float(x))
            print(int(x))

    .. code-block::

        x = Uniform(1.1)
        for _ in range(10):
            assert int(x) == 1
            assert float(x) == 1.
    """

    low: int = 1
    high: Optional[int] = None
    name: str = "discrete_uniform"

    def __call__(self) -> int:
        if self.high is None:
            return self.low
        return random.randint(self.low, self.high)

    def __int__(self):
        return self()

    def __float__(self):
        return float(int(self))


@dataclass
class DataGenConfig(ConfigObj):
    num_graphs: int = MISSING
    density: Uniform = field(default_factory=Uniform)
    num_nodes: DiscreteUniform = field(default_factory=DiscreteUniform)
    path_length: DiscreteUniform = field(default_factory=DiscreteUniform)
    composition_density: Uniform = field(default_factory=Uniform)
    batch_size: int = MISSING
    shuffle: bool = MISSING
    name: str = ""


@dataclass
class DataConfig(ConfigObj):
    eval: DataGenConfig = field(default_factory=lambda: DataGenConfig(name="eval"))
    train: DataGenConfig = field(default_factory=lambda: DataGenConfig(name="train"))


@dataclass
class Config(ConfigObj):
    """Configuration dataclass.

    .. note::

        To access properties from a omegaconf.DictConfig instance,
        call the following

        .. code-block::

            from caldera.utils.conf import dataclass_from_dict, dataclass_to_dict

            config = dataclass_from_dict(Config, dict(cfg))
    """

    network: GraphNetConfig = field(default_factory=GraphNetConfig)
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    data: DataConfig = field(default_factory=DataConfig)


def initialize_config():
    #     """Initialize structured configuration."""
    cs = ConfigStore()
    cs.store(name="config", node=Config)
    cs.store(group="network", name="default", node=GraphNetConfig)
    cs.store(group="hyperparameters", name="default", node=HyperParameters)
    cs.store(group="data", name="default", node=DataConfig)


# TODO: make this a metaclass and automatic?
def initialize_resolvers():
    OmegaConf.register_resolver(Uniform.name, Uniform)
    OmegaConf.register_resolver(DiscreteUniform.name, DiscreteUniform)


initialize_config()
initialize_resolvers()

__all__ = [
    "initialize_config",
    "Config",
    "GraphNetConfig",
    "GraphLayerConfig",
    "DictConfig",
    "MISSING",
    "OmegaConf",
]
