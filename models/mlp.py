"""Definition of MLP."""
from typing import List, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Class for simple Multi-Layer Perceptron (MLP)."""

    def __init__(self, input_dim: int, intermediate_dim: List[int], output_dim: int):
        """Constructor.

        Args:
            input_dim (int): input dimension.
            intermediate_dim (list): number of neurons for each intermediate layer.
            output_dim (int): output dimension.
        """
        super().__init__()

        layers = []
        if len(intermediate_dim) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            num_neurons = [input_dim] + intermediate_dim
            for prev, curr in zip(num_neurons[:-1], num_neurons[1:]):
                layers.append(nn.Linear(prev, curr))
                layers.append(nn.GELU())

            layers.append(nn.Linear(num_neurons[-1], output_dim))

        self._net = nn.Sequential(*layers)
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Input requires topology due to compatibility with other models.

        Args:
            data (tuple): input node features and graph topology.

        Returns:
            output (tuple): output node features and graph topology.
        """
        x, topology = data
        output = self._net(x)
        output = self._softmax(output)

        return output, topology
