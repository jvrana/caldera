import os
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Union

from hydra.conf import ConfigStore
from hydra.experimental import compose as hydra_compose_config
from hydra.experimental import initialize_config_dir
from omegaconf import DictConfig
from omegaconf import OmegaConf
from py.path import local

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
    log_level: str = "WARNING"
    offline: bool = False


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


def get_config(
    overrides: List[str] = None,
    config_path: str = "conf",
    config_name: str = "config",
    directory: str = None,
    as_config_class: bool = False,
) -> Union[DictConfig, Config]:
    """Get config (instead of running command line, as in a jupyter notebook.

    :param overrides: list of config overrides
    :param config_path: config directory path
    :param config_name: main config name
    :param directory:
    :return: DictConfig configuration
    """
    initialize_config()
    directory = directory or os.getcwd()
    with local(directory).as_cwd():
        overrides = overrides or []
        config_path = os.path.join(directory, config_path)
        with initialize_config_dir(config_path):
            cfg = hydra_compose_config(config_name=config_name, overrides=overrides)
    if as_config_class:
        cfg = Config.from_dict_config(cfg)
    return cfg


initialize_config()
initialize_resolvers()
