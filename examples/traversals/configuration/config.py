from dataclasses import dataclass
from dataclasses import field

from hydra.conf import ConfigStore
from omegaconf import OmegaConf

from .data import DataConfig
from .data import DiscreteUniform
from .data import Uniform
from .hyperparameters import HyperParamConfig
from .network import NetConfig
from .tools import ConfigObj


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

    network: NetConfig = field(default_factory=NetConfig)
    hyperparameters: HyperParamConfig = field(default_factory=HyperParamConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gpus: int = 1


def initialize_config():
    #     """Initialize structured configuration."""
    cs = ConfigStore()
    cs.store(name="config", node=Config)
    cs.store(group="network", name="default", node=NetConfig)
    cs.store(group="hyperparameters", name="default", node=HyperParamConfig)
    cs.store(group="data", name="default", node=DataConfig)


# TODO: make this a metaclass and automatic?
def initialize_resolvers():
    OmegaConf.register_resolver(Uniform.name, Uniform)
    OmegaConf.register_resolver(DiscreteUniform.name, DiscreteUniform)


initialize_config()
initialize_resolvers()
