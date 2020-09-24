from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import functional as FM
from torch.nn import functional as F

from .configuration import Config
from .model import Network
from caldera.data import GraphBatch


class TrainingModule(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model: Network = Network(config.network)

    def training_step(self, batch: GraphBatch, batch_idx: int) -> pl.TrainResult:
        input_batch, target_batch = batch
        out_batch_list = self.model.forward(
            input_batch, steps=self.config.hyperparameters.train_core_processing_steps
        )

        loss = torch.tensor(0.0)
        for out_batch in out_batch_list:
            loss += F.mse_loss(target_batch.x, out_batch.x)
            loss += F.mse_loss(target_batch.e, out_batch.e)
            loss += F.mse_loss(target_batch.g, out_batch.g)

        result = pl.TrainResult(loss)
        result.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):

        input_batch, target_batch = batch
        out_batch_list = self.model.forward(
            input_batch, steps=self.config.hyperparameters.train_core_processing_steps
        )

        loss = torch.tensor(0.0)
        for out_batch in out_batch_list:
            loss += F.mse_loss(target_batch.x, out_batch.x)
            loss += F.mse_loss(target_batch.e, out_batch.e)
            loss += F.mse_loss(target_batch.g, out_batch.g)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return result

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
        return torch.optim.Adam(
            self.model.parameters(), lr=self.config.hyperparameters.lr
        )
