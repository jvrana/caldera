# flake8: noqa
##########################################################
# Relative Imports
##########################################################
import sys
from os.path import isfile
from os.path import join

from pytorch_lightning.callbacks import ProgressBar


def find_pkg(name: str, depth: int):
    if depth <= 0:
        ret = None
    else:
        d = [".."] * depth
        path_parts = d + [name, "__init__.py"]

        if isfile(join(*path_parts)):
            ret = d
        else:
            ret = find_pkg(name, depth - 1)
    return ret


def find_and_ins_syspath(name: str, depth: int):
    path_parts = find_pkg(name, depth)
    if path_parts is None:
        raise RuntimeError("Could not find {}. Try increasing depth.".format(name))
    path = join(*path_parts)
    if path not in sys.path:
        sys.path.insert(0, path)


try:
    import caldera
except ImportError:
    find_and_ins_syspath("caldera", 3)

##########################################################
# Main
##########################################################

import copy
import hydra
from examples.traversals.training import TrainingModule
from examples.traversals.data import DataGenerator, DataConfig
from examples.traversals.configuration import Config
from examples.traversals.configuration.data import Uniform, DiscreteUniform
from typing import TypeVar
from pytorch_lightning import Trainer
import logging
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from rich.panel import Panel
from rich import print


logger = logging.getLogger("examples.traversals")
C = TypeVar("C")


def prime_the_model(model: TrainingModule, config: Config):
    logger.info("Priming the model with data")
    config_copy: DataConfig = copy.deepcopy(config.data)
    config_copy.train.num_graphs = 10
    config_copy.eval.num_graphs = 0
    data_copy = DataGenerator(config_copy)
    for a, b in data_copy.train_loader():
        model.model.forward(a, 10)
        break


def print_title():
    print(Panel("Training Example: [red]Traversal", title="[red]caldera"))


# TODO: @bind(Config)(class Foo)   Foo.from_dict_config(cfg)
@hydra.main(config_path="conf", config_name="config")
def main(hydra_cfg: DictConfig):
    print_title()

    # really unclear why hydra has so many unclear validation issues with structure configs using ConfigStore
    # this correctly assigns the correct structured config
    # and updates from the passed hydra config
    # annoying... but this resolves all these issues
    cfg = OmegaConf.structured(Config())
    cfg.update(hydra_cfg)

    # debug
    print(OmegaConf.to_yaml(cfg))

    # defaults
    cfg.hyperparameters.lr = 1e-3
    cfg.hyperparameters.train_core_processing_steps = 10
    cfg.hyperparameters.eval_core_processing_steps = 10

    cfg.data.train.num_graphs = 5000
    cfg.data.train.num_nodes = DiscreteUniform(10, 100)
    cfg.data.train.density = Uniform(0.01, 0.03)
    cfg.data.train.path_length = DiscreteUniform(5, 10)
    cfg.data.train.composition_density = Uniform(0.01, 0.02)
    cfg.data.train.batch_size = 512
    cfg.data.train.shuffle = False

    cfg.data.eval.num_graphs = 500
    cfg.data.eval.num_nodes = DiscreteUniform(10, 100)
    cfg.data.eval.density = Uniform(0.01, 0.03)
    cfg.data.eval.path_length = DiscreteUniform(5, 10)
    cfg.data.eval.composition_density = Uniform(0.01, 0.02)
    cfg.data.eval.batch_size = "${data.eval.num_graphs}"
    cfg.data.eval.shuffle = False

    # explicitly convert the DictConfig back to Config object
    # has the added benefit of performing validation upfront
    # before any expensive training or logging initiates
    config = Config.from_dict_config(cfg)

    # initialize the training module
    training_module = TrainingModule(config)

    logger.info("Priming the model with data")
    prime_the_model(training_module, config)
    print(training_module.model)

    logger.info("Generating data...")
    data = DataGenerator(config.data)
    data.init()
    from rich import print as rprint
    from rich import Panel

    logger.info("Beginning training...")

    trainer = Trainer(gpus=config.gpus)
    trainer.fit(
        training_module,
        train_dataloader=data.train_loader(),
        val_dataloaders=data.eval_loader(),
    )


if __name__ == "__main__":
    main()
