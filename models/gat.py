import torch
import torch.nn as nn


class GAT(nn.Module):
    def __init__(self):
        super().__init__()

        self._net = nn.Sequential(nn.Linear(10, 1))

    def forward(self):
        pass
