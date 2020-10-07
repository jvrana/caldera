from typing import List

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import LightningModule

from .configuration import Config
from .configuration.tools import dataclass_to_dict
from .loggers import logger
from .loggers import warn_once
from .model import Network
from .plotting import comparison_plot
from .plotting import figs_to_pils
from caldera.data import GraphBatch


def requires_logger_experiment(f):
    def wrapped(self, *args, **kwargs):
        if self.logger is None:
            warn_once("No self.logger present. Skipping call to '{}'".format(f))
        elif not hasattr(self.logger, "experiment"):
            warn_once(
                "self.logger has not attr 'experiment'. Skipping call to '{}'".format(f)
            )
        else:
            return f(self, *args, **kwargs)

    return wrapped


class TrainingModule(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model: Network = Network(config.network)
        self.hparams = dataclass_to_dict(config)

        self._loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean", weight=torch.tensor([1.0, 100.0])
        )

    def do_loss(self, input_list: List[GraphBatch], target: GraphBatch):
        losses = []
        for input in input_list:
            target_edge_classes = target.e.argmax(1)
            target_node_classes = target.x.argmax(1)
            edge_loss = self._loss_fn(input.e, target_edge_classes)
            node_loss = self._loss_fn(input.x, target_node_classes)
            input_loss = edge_loss + node_loss
            losses.append(input_loss.unsqueeze(0))
        return torch.cat(losses).sum()

    @requires_logger_experiment
    def logger_experiment_update_hparams(self):
        self.logger.experiment.config.update(self.hparams)

    def on_fit_start(self):
        self.logger_experiment_update_hparams()

    def training_step(self, batch: GraphBatch, batch_idx: int) -> pl.TrainResult:
        input_batch, target_batch = batch
        out_batch_list = self.model.forward(
            input_batch,
            steps=self.config.hyperparameters.train_core_processing_steps,
            save_all=True,
        )
        loss = self.do_loss(out_batch_list, target_batch)

        result = pl.TrainResult(loss)
        result.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.experiment_log({"train_loss": loss})
        return pl.TrainResult(loss)

    @requires_logger_experiment
    def experiment_log(self, *args, **kwargs):
        self.logger.experiment.log(*args, **kwargs)

    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        out_batch_list = self.model.forward(
            input_batch, steps=self.config.hyperparameters.train_core_processing_steps
        )
        loss = self.do_loss(out_batch_list[-1:], target_batch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log("eval_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.experiment_log({"eval_loss": loss})

        if batch_idx == 0:
            self.validate_plot(batch)
        return result

    @requires_logger_experiment
    def validate_plot(self, batch, num_graphs=10):
        if self.logger is None:
            return
        x, y = batch
        x.to_data_list()
        y.to_data_list()
        x = GraphBatch.from_data_list(x.to_data_list()[:num_graphs])
        y = GraphBatch.from_data_list(y.to_data_list()[:num_graphs])

        y_hat = self.model.forward(x, 10)[-1]
        y_graphs = y.to_networkx_list()
        y_hat_graphs = y_hat.to_networkx_list()

        if self.config.training.validate_plot:
            figs = []
            for idx in range(len(y_graphs)):
                yg = y_graphs[idx]
                yhg = y_hat_graphs[idx]
                fig, axes = comparison_plot(yhg, yg)
                figs.append(fig)
            with figs_to_pils(figs) as pils:
                self.experiment_log({"image": [wandb.Image(pil) for pil in pils]})

    # def validation_step(self, batch: Tuple[GraphBatch, GraphBatch], batch_idx: int) -> pl.EvalResult:
    #     x, y = batch
    #     y_hat = self.model(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     acc = FM.accuracy(y_hat, y)
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log_dict({'val_acc': acc, 'val_loss': loss})
    #     return result

    # def test_step(self, batch, batch_idx) -> pl.EvalResult:
    #     result = self.validation_step(batch, batch_idx)
    #     result.rename_keys({'val_acc': 'test_acc', 'val_loss': 'test_loss'})
    #     return result

    def configure_optimizers(self):
        if self.config.training.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=self.config.hyperparameters.lr
            )
        else:
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.config.hyperparameters.lr
            )

        raise ValueError("optimizer not recognized")
