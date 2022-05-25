import torch
import torch.nn as nn

from typing import Tuple


class GATLayer(nn.Module):
    def __init__(self, num_in_features: int, num_out_features: int, num_heads: int, dropout_prob: float = 0.6,
                 concat_heads: bool = False, add_skip_connection: bool = True):
        super().__init__()

        self._head_dim = 1
        self._num_in_features = num_in_features
        self._num_out_features = num_out_features
        self._num_heads = num_heads
        self._concat_heads = concat_heads
        self._add_skip_connection = add_skip_connection

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

    def _aggregate_heads(self, in_node_features: torch.Tensor, out_node_features: torch.Tensor) -> torch.Tensor:
        if not out_node_features.is_contiguous():
            out_node_features = out_node_features.contiguous()

        if self._add_skip_connection:
            if out_node_features.size(-1) == in_node_features.size(-1):
                out_node_features += in_node_features.unsqueeze(1)
            else:
                out_node_features += self._skip_proj(in_node_features).view(-1, self._num_heads, self._num_out_features)

        if self._concat_heads:
            out_node_features = out_node_features.view(-1, self._num_heads * self._num_out_features)
        else:
            out_node_features = torch.mean(out_node_features, dim=self._head_dim)

        out_node_features = self._activation(out_node_features)

        return out_node_features

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        in_node_features, topology = data
        num_nodes = in_node_features.shape[0]
        assert topology.shape == (num_nodes, num_nodes), "Adjacency matrix has invalid shape."

        in_node_features = self._dropout(in_node_features)
        node_features_proj = torch.matmul(in_node_features.unsqueeze(0), self._proj_param)
        node_features_proj = self._dropout(node_features_proj)

        edge_scores_source = torch.bmm(node_features_proj, self._scoring_source)
        edge_scores_target = torch.bmm(node_features_proj, self._scoring_target)
        edge_scores = self._activation(edge_scores_source + edge_scores_target.transpose(1, 2))
        attention_coeffs = self._softmax(edge_scores + topology)

        out_node_features = torch.bmm(attention_coeffs, node_features_proj)
        out_node_features = out_node_features.transpose(0, 1)

        out_node_features = self._aggregate_heads(in_node_features, out_node_features)

        return out_node_features, topology
