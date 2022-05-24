import torch
import torch.nn as nn


class GATLayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_heads, add_skip_connection=True):
        self._num_heads = num_heads
        self._num_in_features = num_in_features
        self._num_out_features = num_out_features

        self._proj_param = nn.Parameter(torch.Tensor(num_heads, num_in_features, num_out_features))
        self._scoring_source = nn.Parameter(torch.Tensor(num_heads, num_out_features, 1))
        self._scoring_target = nn.Parameter(torch.Tensor(num_heads, num_out_features, 1))

        if add_skip_connection:
            self._skip_proj = nn.Linear(num_in_features, num_heads * num_out_features, bias=False)

    def init_params(self):
        nn.init.xavier_uniform_(self._proj_param)
        nn.init.xavier_uniform_(self._scoring_source)
        nn.init.xavier_uniform_(self._scoring_target)