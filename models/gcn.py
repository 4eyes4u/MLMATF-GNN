"""Definition of GCN."""
import math
from typing import List, Tuple

import torch
import torch.nn as nn


class DropoutWrapper(nn.Module):
    """Class that adds dropout layer on top of the module."""

    def __init__(self, fn: nn.Module, dropout_prob: float):
        """Constructor.

        Args:
            fn (nn.Module): module to be encapsulated.
            dropout_prob (float): probability of dropout.
        """
        super().__init__()

        self._fn = fn
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout_prob)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        It requires topology due to compability.

        Args:
            data (tuple): input node features and graph topology.

        Returns:
            output (tuple): output node features and graph topology.
        """
        topology = data[1]
        output = self._fn(data)[0]
        output = self._dropout(self._activation(output))

        return output, topology


class GraphConvolution(nn.Module):
    """Class for Graph Convolution (GC) layer.

    This operator can be seen as a transformation of the input with respect to graph topology.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Constructor of GC.

        Args:
            in_features (int): number of input features.
            out_features (int): number of output features.
            bias (bool): include bias (True) or not (False).
        """
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

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            data (tuple): input node features and graph topology.

        Returns:
            output (tuple): output node features and graph topology.
        """
        x, topology = data
        support = torch.matmul(x, self.weight)
        output = torch.matmul(topology, support)

        if self.bias is not None:
            output += self.bias

        return output, topology


class GCN(nn.Module):
    """Class for Graph Convolutional Network (GCN)."""

    def __init__(self, num_features_per_layer: List[int], dropout_prob: float):
        """Constructor of GCN.

        Args:
            num_features_per_layer (liist): number of input/output features for each GC layer.
            dropout_prob (float): probability of dropout.
        """
        super().__init__()

        layers = []
        for prev, curr in zip(num_features_per_layer[:-1: 2], num_features_per_layer[1::2]):
            layers.append(DropoutWrapper(GraphConvolution(prev, curr), dropout_prob))
        layers.append(GraphConvolution(num_features_per_layer[-2], num_features_per_layer[-1]))

        self._net = nn.Sequential(*layers)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Input needs to be fed as a tuple because GCN is defined as a sequential model.

        Args:
            data (tuple): input node features and graph topology.

        Returns:
            output (tuple): output node features and graph topology.
        """
        output = self._net(data)

        return output
