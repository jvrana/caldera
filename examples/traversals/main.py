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

from examples.traversals.configuration import (
    Config,
    GraphLayerConfig,
    GraphNetConfig,
    DictConfig,
    OmegaConf,
)
from examples.traversals.model import TrainingModule
import hydra


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    training_module = TrainingModule(cfg)

    # cfg = GraphNetConfig(
    #     edge=GraphLayerConfig(size=3, depth=2, out=1, layer_norm=True),
    #     node=GraphLayerConfig(size=3, depth=2, out=1, layer_norm=True),
    #     glob=GraphLayerConfig(size=3, depth=2, out=1, layer_norm=True),
    #     dropout=0.2,
    #     pass_global_to_edge=True,
    #     pass_global_to_node=True,
    #     activation='leakyrelu',
    #     out_activation='sigmoid',
    #     aggregator_activation=('min', 'max', 'mean', 'sum')
    # )


if __name__ == "__main__":
    main()
