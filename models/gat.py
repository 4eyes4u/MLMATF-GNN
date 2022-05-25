"""Definition of GAT."""
from typing import List, Tuple

from models.gatlayer import GATLayer
import torch
import torch.nn as nn


class GAT(nn.Module):
    """Class for Graph Attention Network (GAT)."""

    def __init__(self, num_heads_per_layer: List[int], num_features_per_layer: List[int], dropout_prob: float = 0.6,
                 add_skip_connection: bool = True, **kwargs):
        """Constructor of GAT.

        Args:
            num_heads_per_layer (list): number of heads for each layer.
            num_features_per_layer (list): number of features for each layer.
            dropout_prob (float): probability of dropout layer used in GAT's layers.
            add_skip_connection (bool): to add (True) or not (False) residual/skip connection.
        """
        super().__init__()

        num_layers = len(num_heads_per_layer)
        num_heads_per_layer = [1] + num_heads_per_layer

        layers = []
        for i in range(num_layers):
            layer = GATLayer(num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                             num_out_features=num_features_per_layer[i + 1],
                             num_heads=num_heads_per_layer[i + 1],
                             to_aggregate_heads=i < num_layers - 1,
                             dropout_prob=dropout_prob,
                             add_skip_connection=add_skip_connection)
            layers.append(layer)

        self._net = nn.Sequential(*layers)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Input needs to be fed as a tuple because GAT is defined as a sequential model.

        Args:
            data (tuple): input node features and graph topology.

        Returns:
            output (tuple): output node features and graph topology.
        """
        output = self._net(data)

        return output
