import hashlib
import json
import os
import pickle
from multiprocessing import Pool
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import networkx as nx
import numpy as np
from rich.progress import Progress

import caldera.utils.functional as fn
from .configuration import DataConfig
from .configuration import DataGenConfig
from .configuration import TrainingConfig
from .progress import safe_update
from caldera.data import GraphData
from caldera.data import GraphDataLoader
from caldera.dataset import GraphDataset
from caldera.testing import annotate_shortest_path
from caldera.transforms import Compose
from caldera.transforms import networkx as nxt
from caldera.utils.mp import multiprocess
from caldera.utils.nx.generators import chain_graph
from caldera.utils.nx.generators import compose_and_connect
from caldera.utils.nx.generators import random_graph
from caldera.utils.nx.generators import uuid_sequence

T = TypeVar("T")


def generate_shortest_path_example(
    n_nodes,
    density,
    path_length,
    weight,
    weight_key="weight",
    compose_density=None,
    composition_weight=None,
):
    d = float(density)
    l = int(path_length)
    if compose_density is None:
        cd = d
    else:
        cd = float(compose_density)
    path = list(uuid_sequence(l))
    h = chain_graph(path, nx.Graph)
    composition_weight = composition_weight or weight
    g = random_graph(n_nodes, density=d, weight=weight, weight_key=weight_key)
    graph = compose_and_connect(
        g, h, cd, lambda n1, n2: {weight_key: float(composition_weight)}
    )

    for _, _, edata in graph.edges(data=True):
        if "weight" not in edata:
            edata["weight"] = 1.0

    annotate_shortest_path(
        graph,
        True,
        True,
        source_key="source",
        target_key="target",
        path_key="shortest_path",
        source=path[0],
        target=path[-1],
        weight=weight_key,
    )

    preprocess = Compose(
        [
            nxt.NetworkxSetDefaultFeature(
                node_default={"source": False, "target": False, "shortest_path": False},
                edge_default={"shortest_path": False},
            ),
            nxt.NetworkxAttachNumpyBool("node", "source", "_features"),  # label nodes as 'start'
            nxt.NetworkxAttachNumpyBool("node", "target", "_features"),  # label nodes as 'end'

            # attached weight
            nxt.NetworkxAttachNumpyFeatures(
                "edge",
                "weight",
                "_features",
                encoding=fn.map_each(lambda x: np.array([x])),
            ),
            nxt.NetworkxAttachNumpyBool("edge", "shortest_path", "_target"),  # label edge as shortest_path
            nxt.NetworkxAttachNumpyBool("node", "shortest_path", "_target"),  # label node as shortest_path
            nxt.NetworkxSetDefaultFeature(
                global_default={
                    "_features": np.array([1.0]),
                    "_target": np.array([1.0]),
                },
            ),
            nxt.NetworkxNodesToStr(),
            nxt.NetworkxToDirected(),
        ]
    )

    g = preprocess([graph])[0]
    return g


def _star_generate_shortest_path_example(args):
    return generate_shortest_path_example(*args)


@multiprocess(on="graph")
def graph_to_data(graph, key):
    return GraphData.from_networkx(graph, feature_key=key)


class DataGenerator:
    def __init__(
        self,
        config: DataConfig,
        train_config: TrainingConfig,
        progress_bar: bool = True,
    ):
        self.config = config
        self.train_config = train_config
        self._train_loader = None
        self._eval_loader = None
        self._train_datasets: Tuple[GraphDataset] = None
        self._eval_datasets: Tuple[GraphDataset] = None
        self.progress_bar = progress_bar

    def _create_nxg(self, data_config):
        return generate_shortest_path_example(
            data_config.num_nodes,
            data_config.density,
            data_config.path_length,
            data_config.weight,
            "weight",
            data_config.composition_density,
            data_config.composition_weight,
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
                data_config.weight,
                "weight",
                data_config.composition_density,
                data_config.composition_weight,
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

    def _create_dataset(
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

        return GraphDataset(input_datalist), GraphDataset(target_datalist)

    def _create_dataloader(self, *datasets, **kwargs):
        loader = GraphDataLoader(*datasets, **kwargs)
        return loader

    def _create_train_loader(self):
        return GraphDataLoader(
            *self.train_datasets,
            batch_size=self.train_config.train_batch_size,
            shuffle=self.train_config.train_shuffle
        )
        return self._create_dataloader(
            "_features",
            "_target",
            self.config.train,
            batch_size=self.train_config.train_batch_size,
            shuffle=self.train_config.train_shuffle,
        )

    def _create_eval_loader(self):
        return GraphDataLoader(
            *self.eval_datasets,
            batch_size=self.train_config.eval_batch_size,
            shuffle=self.train_config.eval_shuffle
        )
        return self._create_dataloader(
            "_features",
            "_target",
            self.config.eval,
            batch_size=self.train_config.eval_batch_size,
            shuffle=self.train_config.eval_shuffle,
        )

    @property
    def train_datasets(self):
        if self._train_datasets is None:
            self._train_datasets = self._create_dataset(
                "_features", "_target", self.config.train
            )
        return self._train_datasets

    @property
    def eval_datasets(self):
        if self._eval_datasets is None:
            self._eval_datasets = self._create_dataset(
                "_features", "_target", self.config.eval
            )
        return self._eval_datasets

    def train_loader(self) -> GraphDataLoader:
        if self._train_loader is None:
            self._train_loader = self._create_train_loader()
        return self._train_loader

    def eval_loader(self) -> GraphDataLoader:
        if self._eval_loader is None:
            self._eval_loader = self._create_eval_loader()
        return self._eval_loader

    def reset(self) -> None:
        self._train_datasets = None
        self._train_loader = None
        self._eval_datasets = None
        self._eval_loader = None

    def init(self) -> Tuple[GraphDataLoader, GraphDataLoader]:
        self.train_loader()
        self.eval_loader()
        return self

    @staticmethod
    def _checksum(config):
        hashed = config.to_dict()
        hashed = json.dumps(hashed, sort_keys=True)
        hashed = hashlib.md5(hashed.encode("utf-8")).hexdigest()
        return hashed

    @classmethod
    def _checksum_filepath(cls, config):
        here = os.path.abspath(os.path.dirname(__file__))
        filename = cls._checksum(config) + ".data.pkl"
        filedir = os.path.join(here, "cached_data")
        if not os.path.isdir(filedir):
            os.mkdir(filedir)
        filepath = os.path.join(filedir, filename)
        return filepath

    def dump(self):
        filepath = self._checksum_filepath(self.config)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, config: DataConfig, train_config: TrainingConfig, *args, **kwargs):
        filepath = cls._checksum_filepath(config)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                loaded: cls = pickle.load(f)
                loaded.train_config = train_config
                loaded._eval_loader = None
                loaded._train_loader = None
                loaded.init()
                return loaded
        else:
            return cls(config, train_config, *args, **kwargs)
