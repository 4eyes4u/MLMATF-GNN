"""Utility script."""
import datetime
import json
import os
import pickle
from typing import Any, Dict, Union

import networkx as nx
import numpy as np
import scipy.sparse as sparse


CORA_PARAMS = {
    "train_range": [0, 140],
    "val_range": [140, 640],
    "test_range": [1708, 2708],
    "num_features": 1433,
    "num_classes": 7
}


def generate_unique_name() -> str:
    """Generates (unique) name that is used for naming local runs.

    Returns:
        run_name (str): unique name.
    """
    run_name = datetime.datetime.utcnow().strftime("%y-%m-%d-%H-%M-%S-%f")

    return run_name


def make_dir_hierarchy() -> Dict[str, str]:
    """Creating all necessary directories that the current run will use.

    Returns:
        paths (argparse.Namespace): dictionary with created paths and other information.
    """
    run_name = generate_unique_name()

    # directory for storing information of current run
    runs_path = os.path.join("logs", run_name)
    os.makedirs(runs_path, exist_ok=True)

    # directory for storing log (including loss information)
    log_path = os.path.join(runs_path, "log")
    os.makedirs(log_path, exist_ok=True)

    # directory for storing checkpoints
    checkpoints_path = os.path.join(runs_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    paths = {
        "run_name": run_name,
        "runs_path": runs_path,
        "log_path": log_path,
        "checkpoints_path": checkpoints_path
    }

    return paths


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


def normalize_features(features: Union[np.ndarray, sparse.csr.csr_matrix]):
    """Normalize each row to the unit row.

    Args:
        node_features (union): matrix of node features.

    Returns:
        node_features_normalized (union): normalized node features.
    """
    features_sum = np.array(np.sum(features, axis=1))

    features_inv_sum = np.squeeze(1.0 / features_sum)
    features_inv_sum[np.isinf(features_inv_sum)] = 1

    features_inv_sum_diag = sparse.diags(features_inv_sum)
    features_normalized = features_inv_sum_diag.dot(features)

    return features_normalized


def load_cora(data_dir: str, topology_normalization: str):
    """Loads CORA dataset.

    Args:
        data_dir (str): data directory.
        topology_normalization (str): how to normalize topology.

    Returns:
        node_features, node_labels, topology (tuple): pre-processed data.
    """
    # reading raw data
    node_features = read_from_binary(os.path.join(data_dir, "node_features.csr"))
    node_labels = read_from_binary(os.path.join(data_dir, "node_labels.npy"))
    adjacency_list = read_from_binary(os.path.join(data_dir, "adjacency_list.dict"))

    node_features = normalize_features(node_features)

    # making dense adjacency matrix ready for softmax to be applied
    topology = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list))
    topology = topology.todense().astype(np.float32)
    # adding loops
    topology += np.identity(topology.shape[0])

    if topology_normalization == "softmax":
        # making topology ready for softmax (-np.inf -> 0; 0 -> 1)
        topology[topology == 0] = -np.inf
        topology[topology == 1] = 0
    elif topology_normalization == "unit":
        topology = normalize_features(topology)

    return node_features, node_labels, topology
