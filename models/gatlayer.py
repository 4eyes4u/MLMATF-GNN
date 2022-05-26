"""Definition of GAT layer."""
from typing import Tuple

import torch
import torch.nn as nn


class GATLayer(nn.Module):
    """Class for GAT's elementary layer."""

    def __init__(self, num_in_features: int, num_out_features: int, num_heads: int, dropout_prob: float = 0.6,
                 to_aggregate_heads: bool = False, add_skip_connection: bool = True):
        """Constructor of GAT layer.

        Args:
            num_in_features (int): number of input features.
            num_out_features (int): number of output features.
            num_heads (int): number of attention heads.
            dropout_prob (float): probability of dropout.
            to_aggregate_heads (bool): aggregate features of attention heads (True) or not (False).
            add_skip_connection (bool): to add (True) or not (False) residual/skip connection.
        """
        super().__init__()

        # head's dimension
        self._head_dim = 1
        self._num_in_features = num_in_features
        self._num_out_features = num_out_features
        self._num_heads = num_heads
        self._to_aggregate_heads = to_aggregate_heads
        self._add_skip_connection = add_skip_connection

        # not used during training - only for logging/visualization
        self._attention_weights = None

        # projection and scoring mappings
        self._proj_param = nn.Parameter(torch.Tensor(num_heads, num_in_features, num_out_features))
        self._scoring_source = nn.Parameter(torch.Tensor(num_heads, num_out_features, 1))
        self._scoring_target = nn.Parameter(torch.Tensor(num_heads, num_out_features, 1))

        if add_skip_connection:
            self._skip_proj = nn.Linear(num_in_features, num_heads * num_out_features, bias=False)

        self._activation = nn.LeakyReLU(negative_slope=0.2)
        self._softmax = nn.Softmax(dim=-1)
        self._dropout = nn.Dropout(p=dropout_prob)

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self._proj_param)
        nn.init.xavier_uniform_(self._scoring_source)
        nn.init.xavier_uniform_(self._scoring_target)

    def _aggregate_heads(self, attention_weights: torch.Tensor, in_node_features: torch.Tensor, out_node_features: torch.Tensor) \
            -> torch.Tensor:
        """Aggregate features of all attention heads.

        Aggregation is either cocatenation or mean.

        Args:
            attention_weights (torch.Tensor): attention coefficients.
            in_node_features (torch.Tensor): input node features.
            out_node_features (torch.Tensor): output node features.

        Returns:
            out_node_features (torch.Tensor): aggregated node features.
        """
        # for later inspection
        self._attention_weights = attention_weights

        # view will raise an exception if underlying memory isn't contiguous
        if not out_node_features.is_contiguous():
            out_node_features = out_node_features.contiguous()

        if self._add_skip_connection:
            if out_node_features.size(-1) == in_node_features.size(-1):
                out_node_features += in_node_features.unsqueeze(1)
            else:
                out_node_features += self._skip_proj(in_node_features).view(-1, self._num_heads, self._num_out_features)

        if self._to_aggregate_heads:
            out_node_features = out_node_features.view(-1, self._num_heads * self._num_out_features)
        else:
            out_node_features = torch.mean(out_node_features, dim=self._head_dim)

        out_node_features = self._activation(out_node_features)

        return out_node_features

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Input needs to be fed as a tuple because GAT is defined as a sequential model.

        Args:
            data (tuple): input node features and graph topology.

        Returns:
            out_node_features, topology (tuple): output node features and graph topology.
        """
        in_node_features, topology = data
        num_nodes = in_node_features.shape[0]
        assert topology.shape == (num_nodes, num_nodes), "Adjacency matrix has invalid shape."

        in_node_features = self._dropout(in_node_features)
        node_features_proj = torch.matmul(in_node_features.unsqueeze(0), self._proj_param)
        node_features_proj = self._dropout(node_features_proj)

        edge_scores_source = torch.bmm(node_features_proj, self._scoring_source)
        edge_scores_target = torch.bmm(node_features_proj, self._scoring_target)
        edge_scores = self._activation(edge_scores_source + edge_scores_target.transpose(1, 2))
        attention_weights = self._softmax(edge_scores + topology)

        out_node_features = torch.bmm(attention_weights, node_features_proj)
        out_node_features = out_node_features.transpose(0, 1)

        out_node_features = self._aggregate_heads(attention_weights, in_node_features, out_node_features)

        return out_node_features, topology
