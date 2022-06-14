from typing import List, Tuple
import math

import torch
import torch.nn as nn


class DropoutWrapper(nn.Module):
    def __init__(self, fn: nn.Module, dropout_prob: float):
        super().__init__()

        self._fn = fn
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout_prob)

    def forward(self, data):
        topology = data[1]
        output = self._fn(data)[0]
        output = self._dropout(self._activation(output))

        return output, topology


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_params()

    def _init_params(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]):
        x, topology = data
        support = torch.matmul(x, self.weight)
        output = torch.matmul(topology, support)

        if self.bias is not None:
            output += self.bias

        return output, topology


class GCN(nn.Module):
    def __init__(self, num_features_per_layer: List[int], dropout_prob: float):
        super().__init__()

        layers = []
        for prev, curr in zip(num_features_per_layer[:-1: 2], num_features_per_layer[1::2]):
            layers.append(DropoutWrapper(GraphConvolution(prev, curr), dropout_prob))
        layers.append(GraphConvolution(num_features_per_layer[-2], num_features_per_layer[-1]))

        self._net = nn.Sequential(*layers)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]):
        output = self._net(data)

        return output
