import torch
import torch.nn as nn
from torch.optim import Adam

from utils.utils import CORA_PARAMS, load_cora
from models import GAT


class GATTrainer:
    def __init__(self, config):
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

    def run_training(self):
        train_labels = self._node_labels.index_select(0, self._indices["train"])
        train_indices = self._indices["train"]
        graph_data = (self._node_features, self._topology)

        for epoch in range(1, self._config["epochs"] + 1):
            self._model.train()
            self._optimizer.zero_grad()

            pred_labels = self._model(graph_data)[0].index_select(0, train_indices)
            loss = self._criterion(train_labels, pred_labels)

            loss.backward()
            self._optimizer.step()
            self.run_val()

    def run_val(self):
        self._model.eval()

    def run_test(self):
        self._model.eval()


if __name__ == "__main__":    
    config = {"data_dir": ".\\data", "lr": 1e-4, "weight_decay": 1e-6, "model_kwargs": {}}
    trainer = GATTrainer(config)
    trainer.run_training()
