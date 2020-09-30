# flake8: noqa
##########################################################
# Relative Imports
##########################################################
import sys
from os.path import isfile
from os.path import join


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
from examples.traversals.loggers import logger
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich import print
from rich.syntax import Syntax


C = TypeVar("C")


def prime_the_model(model: TrainingModule, config: Config):
    logger.info("Priming the model with data")
    config_copy: DataConfig = copy.deepcopy(config.data)
    config_copy.train.num_graphs = 10
    config_copy.eval.num_graphs = 0
    data_copy = DataGenerator(config_copy, progress_bar=False)
    for a, b in data_copy.train_loader():
        model.model.forward(a, 10)
        break


def print_title():
    print(Panel("Training Example: [red]Traversal", title="[red]caldera", expand=False))


def print_model(model: TrainingModule):
    print(Panel("Network", expand=False))
    print(model)


def print_yaml(cfg: Config):
    print(Panel("Configuration", expand=False))
    print(Syntax(OmegaConf.to_yaml(cfg), "yaml"))


# def config_override(cfg: DictConfig):
#     # defaults
#     cfg.hyperparameters.lr = 1e-3
#     cfg.hyperparameters.train_core_processing_steps = 10
#     cfg.hyperparameters.eval_core_processing_steps = 10
#
#     cfg.data.train.num_graphs = 5000
#     cfg.data.train.num_nodes = DiscreteUniform(10, 100)
#     cfg.data.train.density = Uniform(0.01, 0.03)
#     cfg.data.train.path_length = DiscreteUniform(5, 10)
#     cfg.data.train.composition_density = Uniform(0.01, 0.02)
#     cfg.data.train.batch_size = 512
#     cfg.data.train.shuffle = False
#
#     cfg.data.eval.num_graphs = 500
#     cfg.data.eval.num_nodes = DiscreteUniform(10, 100)
#     cfg.data.eval.density = Uniform(0.01, 0.03)
#     cfg.data.eval.path_length = DiscreteUniform(5, 10)
#     cfg.data.eval.composition_density = Uniform(0.01, 0.02)
#     cfg.data.eval.batch_size = "${data.eval.num_graphs}"
#     cfg.data.eval.shuffle = False


@hydra.main(config_path="conf", config_name="config")
def main(hydra_cfg: DictConfig):

    print_title()

    logger.setLevel(hydra_cfg.log_level)
    if hydra_cfg.log_level.upper() == "DEBUG":
        verbose = True
    else:
        verbose = False
    # really unclear why hydra has so many unclear validation issues with structure configs using ConfigStore
    # this correctly assigns the correct structured config
    # and updates from the passed hydra config
    # annoying... but this resolves all these issues
    cfg = OmegaConf.structured(Config())
    cfg.update(hydra_cfg)

    # debug
    if verbose:
        print_yaml(cfg)

    from pytorch_lightning.loggers import WandbLogger

    # explicitly convert the DictConfig back to Config object
    # has the added benefit of performing validation upfront
    # before any expensive training or logging initiates
    config = Config.from_dict_config(cfg)

    wandb_logger = WandbLogger(project="pytorchlightning", offline=config.offline)

    # initialize the training module
    training_module = TrainingModule(config)

    logger.info("Priming the model with data")
    prime_the_model(training_module, config)
    logger.debug(Panel("Model", expand=False))

    if verbose:
        print_model(training_module)

    logger.info("Generating data...")
    data = DataGenerator(config.data)
    data.init()

    logger.info("Beginning training...")
    trainer = Trainer(gpus=config.gpus, logger=wandb_logger, check_val_every_n_epoch=5)

    trainer.fit(
        training_module,
        train_dataloader=data.train_loader(),
        val_dataloaders=data.eval_loader(),
    )


if __name__ == "__main__":
    main()
