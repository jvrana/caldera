# flake8: noqa
##########################################################
# Relative Imports
##########################################################
import sys
from os.path import isfile
from os.path import join
import argparse


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
from examples.traversals.configuration import Config, get_config
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
    data_copy = DataGenerator(config_copy, train_config=config.training, progress_bar=False)
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


def dry_run(config: Config):
    config.data.train.num_graphs = 10
    config.data.eval.num_graphs = 10
    config.logging.disabled = True
    config.logging.log_level = "DEBUG"
    config.logging.offline = True
    config.training.max_epochs = 5
    config.training.validate_plot = False
    config.training.check_val_every_n_epoch = 3
    config.cache_data = False
    return config


# @hydra.main(config_path="conf", config_name="config")
def main():
    hydra_cfg = get_config(overrides=sys.argv[1:])
    print_title()

    logger.setLevel(hydra_cfg.logging.log_level)
    if hydra_cfg.logging.log_level.upper() == "DEBUG":
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
    if config.dry_run:
        config = dry_run(config)

    # initialize logger
    if config.logging.disabled is False:
        wandb_logger = WandbLogger(project=config.logging.project, offline=config.logging.offline)
    else:
        wandb_logger = None

    # initialize the training module
    training_module = TrainingModule(config)

    logger.info("Priming the model with data")
    prime_the_model(training_module, config)
    logger.debug(Panel("Model", expand=False))

    if verbose:
        print_model(training_module)

    # data generation
    # TODO: save Dataset, not Dataloader!
    logger.info("Generating data...")
    data = DataGenerator.load(config.data, config.training)
    data.init()
    if config.cache_data:
        data.dump()

    # training
    logger.info("Beginning training...")
    trainer = Trainer(max_epochs=config.training.max_epochs, gpus=config.training.gpus, logger=wandb_logger, check_val_every_n_epoch=config.training.check_val_every_n_epoch)

    if not config.logging.disabled and config.logging.log_all:
        wandb_logger.experiment.watch(training_module.model, log='all')

    trainer.fit(
        training_module,
        train_dataloader=data.train_loader(),
        val_dataloaders=data.eval_loader(),
    )

    result = trainer.test(test_dataloaders=data.eval_loader())
    print(result)
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb')
    ns, _ = parser.parse_known_args(sys.argv[1:])
    # if ns.wandb:
    if True:
        sys.argv = [a.strip('--') for a in sys.argv if 'wandb' not in a]
    print(sys.argv)
    main()
