"""Entry training script."""
import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from models import GAT
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import CORA_PARAMS, load_cora, load_train_config, make_dir_hierarchy
from utils.metrics import calc_accuracy


class GATTrainer:
    """Class used for training the GAT."""

    def __init__(self, config: Dict[str, Any]):
        """Constructor.

        Args:
            config (dict): configuration of the training, inference and the model.
        """
        self._paths = make_dir_hierarchy()
        self._setup_logger(os.path.join(self._paths["log_path"], "log.txt"))

        with open(os.path.join(self._paths["runs_path"], "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        node_features, node_labels, topology = load_cora(config["data_dir"])
        self._node_features = torch.tensor(node_features.todense(), device=self._device)
        self._node_labels = torch.tensor(node_labels, dtype=torch.long, device=self._device)
        self._topology = torch.tensor(topology, dtype=torch.float32, device=self._device)
        self._indices = {
            "train": torch.arange(*CORA_PARAMS["train_range"], dtype=torch.long, device=self._device),
            "val": torch.arange(*CORA_PARAMS["val_range"], dtype=torch.long, device=self._device),
            "test": torch.arange(*CORA_PARAMS["test_range"], dtype=torch.long, device=self._device)
        }

        self._model = GAT(**config["model_kwargs"]).to(self._device)
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = Adam(self._model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        # saves all metrics and losses during training
        self._aggregator = {}

    def _update_aggregator(self, name: str, value: float) -> None:
        aggregated_values = self._aggregator.get(name, [])
        aggregated_values.append(value)
        self._aggregator[name] = aggregated_values

    def _setup_logger(self, log_path: str) -> None:
        """Setup logging to print logs to both file and stdout.

        Timestamps are in GMT format due to lexicographic order. Example: 22-01-27-21-09-48.

        Args:
            log_path (str): path of log file.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s: [%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.Formatter.converter = time.gmtime

    def _prepare_log(self, epoch, ce, acc, name):
        epoch_length = len(str(self._config["epochs"]))
        total_length = len(f"epoch=[{self._config['epochs']}/{self._config['epochs']}]:")
        epoch_log = f"epoch=[{epoch:>{epoch_length}}/{self._config['epochs']}]:"
        ce_log = f"CE_{name}={ce:.5f}"
        acc_log = f"ACC_{name}={acc:.0%}"
        log = f"{epoch_log:<{total_length}} {ce_log:>16} {acc_log:>14}"

        return log

    def run_training(self):
        train_labels = self._node_labels.index_select(0, self._indices["train"])
        train_indices = self._indices["train"]
        graph_data = (self._node_features, self._topology)

        for epoch in range(1, self._config["epochs"] + 1):
            self._model.train()
            self._optimizer.zero_grad()

            pred_labels = self._model(graph_data)[0].index_select(0, train_indices)
            loss = self._criterion(pred_labels, train_labels)
            accuracy = calc_accuracy(pred_labels, train_labels)

            logging.info(self._prepare_log(epoch, loss.item(), accuracy, "train"))
            self._update_aggregator("CE_train", (epoch, loss.item()))
            self._update_aggregator("ACC_train", (epoch, accuracy))

            loss.backward()
            self._optimizer.step()

            if epoch % self._config["ckpt_freq"] == 0:
                self._dump_model(epoch)

            if epoch == 1 or epoch % self._config["val_freq"] == 0:
                self._run_val(epoch)

            if epoch == 1 or epoch % self._config["test_freq"] == 0:
                self._run_test(epoch)

    def _run_val(self, epoch: int):
        val_labels = self._node_labels.index_select(0, self._indices["val"])
        val_indices = self._indices["val"]
        graph_data = (self._node_features, self._topology)

        self._model.eval()
        pred_labels = self._model(graph_data)[0].index_select(0, val_indices)
        loss = self._criterion(pred_labels, val_labels)
        accuracy = calc_accuracy(pred_labels, val_labels)

        logging.info(self._prepare_log(epoch, loss.item(), accuracy, "val"))
        self._update_aggregator("CE_val", (epoch, loss.item()))
        self._update_aggregator("ACC_val", (epoch, accuracy))

    def _run_test(self, epoch: int):
        test_labels = self._node_labels.index_select(0, self._indices["test"])
        test_indices = self._indices["test"]
        graph_data = (self._node_features, self._topology)

        self._model.eval()
        pred_labels = self._model(graph_data)[0].index_select(0, test_indices)
        loss = self._criterion(pred_labels, test_labels)
        accuracy = calc_accuracy(pred_labels, test_labels)

        logging.info(self._prepare_log(epoch, loss.item(), accuracy, "test"))
        self._update_aggregator("CE_test", (epoch, loss.item()))
        self._update_aggregator("ACC_test", (epoch, accuracy))

    def _dump_model(self, epoch: int) -> None:
        """Dumps current checkpoint.

        Args:
            epoch (int): current epoch.
        """
        ckpt_path = os.path.join(self._paths["checkpoints_path"], f"gat_{epoch}.ckpt")
        torch.save(self._model.state_dict(), ckpt_path)

    @property
    def aggregator(self):
        """Getter for aggregator."""
        return self._aggregator


def plot_metrics(aggregator: Dict[str, List[Tuple[float, float]]], metric_name: str, axis):
    metric_name = metric_name.upper()
    suffixes = ["train", "val", "test"]
    colors = ["blue", "green", "red"]

    for suffix, color in zip(suffixes, colors):
        name = f"{metric_name}_{suffix}"
        metric_values = aggregator[name]
        x_coords = list(map(lambda p: p[0], metric_values))
        y_coords = list(map(lambda p: p[1], metric_values))

        axis.plot(x_coords, y_coords, label=suffix, color=color)

    axis.legend(loc="upper right")
    axis.set_title(metric_name)
    axis.grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", help="Plot metrics & losses.", action="store_true", default=False)
    args, _ = parser.parse_known_args()

    config = load_train_config(".\\configs\\config.json")
    trainer = GATTrainer(config)
    trainer.run_training()

    if args.plot:
        aggregator = trainer.aggregator
        fig, axes = plt.subplots(2)

        # plotting loss
        plot_metrics(aggregator, "CE", axes[0])
        # plotting accuracy
        plot_metrics(aggregator, "ACC", axes[1])

        plt.show()
