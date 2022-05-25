import logging
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from utils.utils import CORA_PARAMS, load_cora, load_train_config, make_dir_hierarchy
from models import GAT


class GATTrainer:
    def __init__(self, config):
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

    def run_training(self):
        train_labels = self._node_labels.index_select(0, self._indices["train"])
        train_indices = self._indices["train"]
        graph_data = (self._node_features, self._topology)

        for epoch in range(1, self._config["epochs"] + 1):
            self._model.train()
            self._optimizer.zero_grad()

            pred_labels = self._model(graph_data)[0].index_select(0, train_indices)
            loss = self._criterion(pred_labels, train_labels)
            logging.info(f"epoch={epoch}: CELoss={loss.item():.5f}")

            loss.backward()
            self._optimizer.step()
            self.run_val()

            if epoch % self._config["ckpt_freq"] == 0:
                self._dump_model(epoch)

    def run_val(self):
        self._model.eval()

    def run_test(self):
        self._model.eval()

    def _dump_model(self, epoch):
        ckpt_path = os.path.join(self._paths["checkpoints_path"], f"gat_{epoch}.ckpt")
        torch.save(self._model.state_dict(), ckpt_path)


if __name__ == "__main__":
    config = load_train_config(".\\configs\\config.json")
    trainer = GATTrainer(config)
    trainer.run_training()
