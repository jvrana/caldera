import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import LightningModule

from .configuration import Config
from .configuration.tools import dataclass_to_dict
from .model import Network
from .plotting import comparison_plot
from .plotting import fig_to_pil
from .plotting import figs_to_pils
from caldera.data import GraphBatch


class TrainingModule(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model: Network = Network(config.network)
        self.hparams = dataclass_to_dict(config)
        # self._loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def on_fit_start(self):
        self.logger.experiment.config.update(self.hparams)

    def training_step(self, batch: GraphBatch, batch_idx: int) -> pl.TrainResult:
        input_batch, target_batch = batch
        out_batch_list = self.model.forward(
            input_batch,
            steps=self.config.hyperparameters.train_core_processing_steps,
            save_all=True,
        )

        _loss_f = torch.nn.BCELoss()
        for out_batch in out_batch_list:
            node_loss = _loss_f(out_batch.x, target_batch.x)
            edge_loss = _loss_f(out_batch.e, target_batch.e)
            glob_loss = _loss_f(out_batch.g, target_batch.g)
        loss = (node_loss + edge_loss + glob_loss) / len(out_batch_list)

        result = pl.TrainResult(loss)
        result.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.logger.experiment.log({"train_loss": loss})
        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):

        input_batch, target_batch = batch
        out_batch_list = self.model.forward(
            input_batch, steps=self.config.hyperparameters.train_core_processing_steps
        )
        _loss_f = torch.nn.BCELoss()
        for out_batch in out_batch_list:
            node_loss = _loss_f(out_batch.x, target_batch.x)
            edge_loss = _loss_f(out_batch.e, target_batch.e)
            glob_loss = _loss_f(out_batch.g, target_batch.g)
        loss = (node_loss + edge_loss + glob_loss) / len(out_batch_list)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log(
            "eval_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        if batch_idx == 0:
            self.validate_plot(batch)
        return result

    def validate_plot(self, batch, num_graphs=10):
        x, y = batch
        x.to_data_list()
        y.to_data_list()
        x = GraphBatch.from_data_list(x.to_data_list()[:num_graphs])
        y = GraphBatch.from_data_list(y.to_data_list()[:num_graphs])

        y_hat = self.model.forward(x, 10)[-1]
        y_graphs = y.to_networkx_list()
        y_hat_graphs = y_hat.to_networkx_list()

        figs = []
        for idx in range(len(y_graphs)):
            yg = y_graphs[idx]
            yhg = y_hat_graphs[idx]
            fig, axes = comparison_plot(yhg, yg)
            figs.append(fig)
        with figs_to_pils(figs) as pils:
            self.logger.experiment.log({"image": [wandb.Image(pil) for pil in pils]})

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
