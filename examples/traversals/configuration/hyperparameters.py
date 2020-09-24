from dataclasses import dataclass

from .tools import ConfigObj


@dataclass
class HyperParamConfig(ConfigObj):

    lr: float = 1e-3  #: the learning rate
    train_core_processing_steps: int = (
        10  #: number of core processing steps for training step
    )
    eval_core_processing_steps: int = (
        10  #: number of core processing steps for evaluation step
    )
