import os
from multiprocessing import Pool
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import networkx as nx
import numpy as np
from rich.progress import Progress

from .configuration import DataConfig
from .configuration import DataGenConfig
from .progress import safe_update
from caldera.data import GraphData
from caldera.data import GraphDataLoader
from caldera.testing import annotate_shortest_path
from caldera.transforms import Compose
from caldera.transforms.networkx import NetworkxAttachNumpyBool
from caldera.transforms.networkx import NetworkxAttachNumpyOneHot
from caldera.transforms.networkx import NetworkxNodesToStr
from caldera.transforms.networkx import NetworkxSetDefaultFeature
from caldera.transforms.networkx import NetworkxToDirected
from caldera.utils.mp import multiprocess
from caldera.utils.nx.generators import chain_graph
from caldera.utils.nx.generators import compose_and_connect
from caldera.utils.nx.generators import random_graph
from caldera.utils.nx.generators import uuid_sequence

T = TypeVar("T")


def generate_shortest_path_example(n_nodes, density, path_length, compose_density=None):
    d = float(density)
    l = int(path_length)
    if compose_density is None:
        cd = d
    else:
        cd = float(compose_density)
    path = list(uuid_sequence(l))
    h = chain_graph(path, nx.Graph)
    g = random_graph(n_nodes, density=d)
    graph = compose_and_connect(g, h, cd)

    annotate_shortest_path(
        graph,
        True,
        True,
        source_key="source",
        target_key="target",
        path_key="shortest_path",
        source=path[0],
        target=path[-1],
    )

    preprocess = Compose(
        [
            NetworkxSetDefaultFeature(
                node_default={"source": False, "target": False, "shortest_path": False},
                edge_default={"shortest_path": False},
            ),
            NetworkxAttachNumpyOneHot(
                "node", "source", "_features", classes=[False, True]
            ),
            NetworkxAttachNumpyOneHot(
                "node", "target", "_features", classes=[False, True]
            ),
            NetworkxAttachNumpyOneHot(
                "edge", "shortest_path", "_target", classes=[False, True]
            ),
            NetworkxAttachNumpyOneHot(
                "node", "shortest_path", "_target", classes=[False, True]
            ),
            NetworkxSetDefaultFeature(
                node_default={
                    "_features": np.array([1.0, 0.0]),
                    "_target": np.array([1.0, 0.0]),
                },
                edge_default={
                    "_features": np.array([1.0, 0.0]),
                    "_target": np.array([1.0, 0.0]),
                },
                global_default={
                    "_features": np.array([1.0]),
                    "_target": np.array([1.0]),
                },
            ),
            NetworkxNodesToStr(),
            NetworkxToDirected(),
        ]
    )

    return preprocess([graph])[0]


def _star_generate_shortest_path_example(args):
    return generate_shortest_path_example(*args)


@multiprocess(on="graph")
def graph_to_data(graph, key):
    return GraphData.from_networkx(graph, feature_key=key)


class DataGenerator:
    def __init__(self, config: DataConfig, progress_bar: bool = True):
        self.config = config
        self._train_loader = None
        self._eval_loader = None
        self.progress_bar = progress_bar

    def _create_nxg(self, data_config):
        return generate_shortest_path_example(
            data_config.num_nodes,
            data_config.density,
            data_config.path_length,
            data_config.composition_density,
        )

    def _create_raw_data(
        self,
        data_config: DataGenConfig,
        desc: Optional[str] = None,
        callback: Callable = None,
    ) -> List[nx.DiGraph]:
        graphs = []

        def new_arg():
            return (
                data_config.num_nodes,
                data_config.density,
                data_config.path_length,
                data_config.composition_density,
            )

        def new_args(n):
            return [new_arg() for _ in range(n)]

        args = new_args(data_config.num_graphs)

        with Pool() as pool:
            results = pool.imap_unordered(
                _star_generate_shortest_path_example, args, chunksize=100
            )

            if desc is None:
                desc = "generating {} data ({} cpus)".format(
                    data_config.name, os.cpu_count()
                )
            while True:
                try:
                    result = next(results)
                    graphs.append(result)
                    if callback:
                        callback(result)
                except StopIteration:
                    break

        return graphs

    def _create_dataloader(
        self,
        input_feature_key: str,
        target_feature_key: str,
        config: DataGenConfig,
        progress_bar: Optional[bool] = None,
    ):
        progress_bar = progress_bar or self.progress_bar
        with Progress(auto_refresh=True, refresh_per_second=1) as progress:

            task1 = progress.add_task(
                "[red]Generating {} data...".format(config.name),
                total=config.num_graphs,
                visible=progress_bar,
            )
            task2 = progress.add_task(
                "[purple]Converting {} data...".format(config.name),
                total=config.num_graphs,
                visible=progress_bar,
            )

            update = safe_update(progress, every=0.1)

            graphs = self._create_raw_data(
                config, callback=lambda _: update(task1, advance=1, refresh=False)
            )
            input_datalist = []
            target_datalist = []

            for graph in graphs:
                input_datalist.append(
                    GraphData.from_networkx(graph, feature_key=input_feature_key)
                )
                update(task2, advance=0.5, refresh=False)
                target_datalist.append(
                    GraphData.from_networkx(graph, feature_key=target_feature_key)
                )
                update(task2, advance=0.5, refresh=False)

        loader = GraphDataLoader(
            input_datalist,
            target_datalist,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
        )
        return loader

    def _create_train_loader(self):
        return self._create_dataloader("_features", "_target", self.config.train)

    def _create_eval_loader(self):
        return self._create_dataloader("_features", "_target", self.config.eval)

    def train_loader(self) -> GraphDataLoader:
        if self._train_loader is None:
            self._train_loader = self._create_train_loader()
        return self._train_loader

    def eval_loader(self) -> GraphDataLoader:
        if self._eval_loader is None:
            self._eval_loader = self._create_eval_loader()
        return self._eval_loader

    def reset(self) -> None:
        self._train_loader = None
        self._eval_loader = None

    def init(self) -> Tuple[GraphDataLoader, GraphDataLoader]:
        self.train_loader()
        self.eval_loader()
        return self
