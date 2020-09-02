# flake8: noqa
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))

PACKAGE_PARENT = "../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import wandb
import torch
from caldera.data import GraphBatch
from typing import Tuple
from typing import Dict
from data import generate_data, create_loader, CircuitGenerator
from model import Network
from summary import loader_summary
from tqdm.auto import tqdm
from caldera.utils import _first
import pylab as plt
import pandas as pd
import seaborn as sns
import numpy as np
import random

import argparse

defaults = dict(
    latent_size_0=512,
    latent_size_1=512,
    latent_size_2=512,
    latent_depth_0=5,
    latent_depth_1=5,
    latent_depth_2=5,
    processing_steps=5,
    pass_global_to_node=True,
    pass_global_to_edge=True,
    lr_to_batch_size=5e-4 / 3e3,
    batch_size=3000,
    weight_decay=1e-3,
    epochs=1000,
    train_size=10000,
    dev_size=2000,
    log_every_epoch=10,
    dropout=0.2,
)


def create_data(
    train_size: int,
    dev_size: int,
    n_parts: int,
    train_part_range: Tuple[int, int],
    dev_part_range: Tuple[int, int],
) -> Dict:
    circuit_gen = CircuitGenerator(n_parts)

    data = generate_data(
        circuit_gen, train_size, train_part_range, dev_size, dev_part_range
    )

    return data, circuit_gen


def to(batch, device, **kwargs):
    return GraphBatch(
        batch.x.to(device, **kwargs),
        batch.e.to(device, **kwargs),
        batch.g.to(device, **kwargs),
        batch.edges.to(device, **kwargs),
        batch.node_idx.to(device, **kwargs),
        batch.edge_idx.to(device, **kwargs),
    )


def plot(target_data, out):
    x = target_data.x.cpu().detach().numpy().flatten()
    y = out.x.cpu().detach().numpy().flatten()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    df = pd.DataFrame({"x": x, "y": y})
    ax = sns.scatterplot("x", "y", data=df, ax=ax)
    ax.set_ylim(-5, 20)
    ax.set_xlim(-5, 20)
    return ax, fig


def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def train(**kwargs):
    """This is the documentation for the function.

    :param learning_rate:
    :return:
    """

    # Set up your default hyperparameters before wandb.init
    # so they get properly set in the sweep
    hyperparameter_defaults = dict(defaults)

    # Pass your defaults to wandb.init

    hyperparameter_defaults.update(kwargs)
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    # update config
    wandb.config.update(
        {"learning_rate": config["batch_size"] * config["lr_to_batch_size"]}
    )

    # Your model here ...
    device = "cuda:0"

    class Null:
        pass

    net_config = dict(
        latent_sizes=(
            config["latent_size_0"],
            config["latent_size_1"],
            config["latent_size_2"],
        ),
        depths=(
            config["latent_depth_0"],
            config["latent_depth_1"],
            config["latent_depth_2"],
        ),
        pass_global_to_edge=config["pass_global_to_edge"],
        pass_global_to_node=config["pass_global_to_node"],
        dropout=config["dropout"],
        edge_to_node_aggregators=config.get("aggregators", Null),
        edge_to_global_aggregators=config.get("aggregators", Null),
        node_to_global_aggregators=config.get("aggregators", Null),
        aggregator_activation=config.get("aggregator_activation", Null),
    )
    net_config = {k: v for k, v in net_config.items() if v is not Null}

    net_config["aggregator_activation"] = {
        "leakyrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }.get(net_config["aggregator_activation"])

    net = Network(**net_config)

    # create your data
    data, gen = create_data(
        config["train_size"], config["dev_size"], 20, (2, 6), (8, 20)
    )

    seed(0)
    train_loader = create_loader(
        gen, data["train"], config["batch_size"], shuffle=True, pin_memory=True
    )
    seed(1)
    eval_loader = create_loader(
        gen, data["train/dev"], None, shuffle=False, pin_memory=True
    )

    wandb.config.update(
        {
            "train": {"loader_summary": loader_summary(train_loader)},
            "eval": {"loader_summary": loader_summary(eval_loader)},
        }
    )

    with torch.no_grad():
        batch, _ = _first(train_loader)
        out = net(batch, 3)
    net.to(device, non_blocking=True)

    assert list(net.parameters())
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = torch.nn.MSELoss()
    log_every_epoch = config["log_every_epoch"]
    for epoch in tqdm(range(config["epochs"]), desc="epochs"):
        net.train()

        running_loss = torch.tensor(0.0, device=device)
        for batch_idx, (train_data, target_data) in enumerate(train_loader):

            optimizer.zero_grad()

            # train_data.contiguous()
            # target_data.contiguous()
            # TODO: why clone?
            #             target_data = target_data.clone()
            train_data = to(train_data, device, non_blocking=True)
            target_data = to(target_data, device, non_blocking=True)
            out = net(train_data, config["processing_steps"])
            assert out[-1].x.shape == target_data.x.shape

            # TODO: the scale of the loss is proporational to the processing steps and batch_size, should this be normalized???
            loss = torch.tensor(0.0).to(
                device=device, non_blocking=True, dtype=torch.float32
            )
            for _out in out:
                loss += loss_fn(_out.x, target_data.x)
            loss = loss / (target_data.x.shape[0] * 1.0 * len(out))

            loss.backward()

            optimizer.step()

            loss.detach_()
            running_loss = running_loss + loss

        if epoch % log_every_epoch == 0:
            wandb.log({"train_loss": running_loss.cpu()}, step=epoch)

        if epoch % log_every_epoch == 0:
            net.eval()
            with torch.no_grad():
                eval_data, eval_target = _first(eval_loader)
                eval_data.detach_()
                eval_target.detach_()
                eval_data = to(eval_data, device, non_blocking=True)
                eval_target = to(eval_target, device, non_blocking=True)
                eval_outs = net(eval_data, config["processing_steps"], save_all=False)
                eval_outs[-1].detach_()
                eval_loss = (
                    loss_fn(eval_outs[-1].x, eval_target.x) / eval_outs[-1].x.shape[0]
                )

                wandb.log({"eval_loss": eval_loss}, step=epoch)

                wandb.log(
                    {
                        "edge_attr": wandb.Histogram(eval_outs[-1].e.cpu()),
                        "node_attr": wandb.Histogram(eval_outs[-1].x.cpu()),
                    },
                    step=epoch,
                )

                # # plot values
                # ax, fig = plot(eval_target, eval_outs[-1])
                # wandb.log({"chart": fig}, step=epoch)


if __name__ == "__main__":
    train()
