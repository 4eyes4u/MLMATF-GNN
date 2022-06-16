"""Metrics module."""
import torch


def calc_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates accuracy.

    Args:
        preds (torch.Tensor): predictions.
        labels (torch.Tensor): GT labels.

    Returns:
        accuracy (float): accuracy.
    """
    if len(preds.size()) == 2:
        preds = torch.argmax(preds, dim=1)
    elif len(preds.size()) > 2:
        raise RuntimeError("Invalid prediction shape")

    num_hits = torch.sum(preds == labels)
    accuracy = num_hits / labels.size(0)

    return accuracy
