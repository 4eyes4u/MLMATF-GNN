import json
import networkx as nx
import os
import pickle
from typing import Any, Dict, Union

import numpy as np
import scipy.sparse as sparse


CORA_PARAMS = {
    "train_range": [0, 140],
    "val_range": [140, 640],
    "test_range": [1708, 2708],
    "num_features": 1433,
    "num_classes": 7
}


def load_train_config(path: str) -> Dict[str, Any]:
    """Reads JSON config file and returns it.

    Args:
        path (str): config path.

    Returns:
        config (dict): config.
    """
    with open(path, "r") as f:
        config = json.load(f)

    return config


def read_from_binary(path: str):
    """Reads data from binary file.

    Args:
        path (str): path of the binary file.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def normalize_node_features(node_features: Union[np.ndarray, sparse.csr.csr_matrix]):
    """Normalize each row to the unit row.

    Args:
        node_features (union): matrix of node features.

    Returns:
        node_features_normalized (union): normalized node features.
    """
    node_features_sum = np.array(np.sum(node_features, axis=1))

    node_features_inv_sum = np.squeeze(1 / node_features_sum)
    node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1

    node_features_inv_sum_diag = sparse.diags(node_features_inv_sum)
    node_features_normalized = node_features_inv_sum_diag.dot(node_features)

    return node_features_normalized


def load_cora(data_dir: str):
    """Loads CORA dataset.

    Args:
        data_dir (str): data directory.

    Returns:
        node_features, node_labels, topology (tuple): pre-processed data.
    """
    # reading raw data
    node_features = read_from_binary(os.path.join(data_dir, "node_features.csr"))
    node_labels = read_from_binary(os.path.join(data_dir, "node_labels.npy"))
    adjacency_list = read_from_binary(os.path.join(data_dir, "adjacency_list.dict"))

    node_features = normalize_node_features(node_features)

    # making dense adjacency matrix ready for softmax to be applied
    topology = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list))
    topology = topology.todense().astype(np.float32)
    # adding loops
    topology += np.identity(topology.shape[0])
    # handling parallel edges
    topology[topology > 0] = 1
    # making topology ready for softmax (-np.inf -> 0; 0 -> 1)
    topology[topology == 0] = -np.inf
    topology[topology == 1] = 0

    return node_features, node_labels, topology
