from typing import List

import torch
import torch.nn as nn

from models.gatlayer import GATLayer


class GAT(nn.Module):
    def __init__(self, num_heads_per_layer: List[int], num_features_per_layer: List[int], dropout_prob: bool = 0.6,
                 add_skip_connection: bool = True, **kwargs):
        super().__init__()

        num_layers = len(num_heads_per_layer)
        num_heads_per_layer = [1] + num_heads_per_layer

        layers = []
        for i in range(num_layers):
            layer = GATLayer(num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                             num_out_features=num_features_per_layer[i + 1],
                             num_heads=num_heads_per_layer[i + 1],
                             concat_heads=i < num_layers - 1,
                             dropout_prob=dropout_prob,
                             add_skip_connection=add_skip_connection)
            layers.append(layer)

        self._net = nn.Sequential(*layers)

    def forward(self, data):
        output = self._net(data)

        return output
