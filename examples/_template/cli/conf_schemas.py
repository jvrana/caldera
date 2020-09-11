from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from omegaconf import MISSING
from omegaconf import OmegaConf


@dataclass
class NetworkConfig:
    """Your parameters."""


@dataclass
class TrainConfig:
    """Your parameters."""


@dataclass
class DataLoaderConfig:
    """Your parameters."""


@dataclass
class DataConfig:
    """Your parameters."""


defaults = [
    # Load the config "mysql" from the config group "db"
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    network: NetworkConfig = MISSING
    train: TrainConfig = MISSING


# Create config group `db` with options 'mysql' and 'postgreqsl'
cs = ConfigStore.instance()
cs.store(name="network", node=NetworkConfig)
cs.store(name="config", node=Config)
