#################################################################################
# FIX `sys.path` and import modules
#################################################################################

import sys
import os

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.insert(0, SCRIPT_DIR)

from .utils import find_and_ins_syspath

try:
    import caldera
except ImportError:
    find_and_ins_syspath('caldera', 4)
    import caldera

#################################################################################
# MAIN
#################################################################################

from typing import Tuple
from typing import Optional, List, Any, Union
from dataclasses import dataclass, field
from omegaconf import MISSING, DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import hydra


@dataclass
class NetworkConfig:
    latent_sizes: Tuple[int, int, int] = (32, 32, 32)
    out_sizes: Tuple[int, int, int] = (1, 1, 1)
    latent_depths: Tuple[int, int, int] = (1, 1, 1)
    dropout: Optional[float] = None
    pass_global_to_edge: bool = True,
    pass_global_to_node: bool = True,


@dataclass
class TrainConfig:
    pass


@dataclass
class DataLoaderConfig:
    name: str
    num: int


@dataclass
class DataConfig:
    density: float
    path_length: Union[int, Tuple[int, int]]
    compose_density: float
    n_nodes: Union[int, Tuple[int, int]]


defaults = [
    # Load the config "mysql" from the config group "db"
    {"network": "default"}
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


@hydra.main(conf_paht="conf")
def train():
    pass

if __name__ == "__main__":
    train()