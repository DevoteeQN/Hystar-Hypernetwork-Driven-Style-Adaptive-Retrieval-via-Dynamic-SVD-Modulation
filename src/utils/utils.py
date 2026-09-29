"""Small utilities shared by training and evaluation."""

from __future__ import annotations

import random

import numpy as np
import torch


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_paired_accuracy(similarity: torch.Tensor, k: int = 1) -> float:
    """Accuracy when row i's positive target is column i."""
    if similarity.ndim != 2:
        raise ValueError("similarity must be a 2D matrix")
    rows, cols = similarity.shape
    if rows != cols:
        raise ValueError("paired retrieval expects a square similarity matrix")
    k = min(k, cols)
    topk = similarity.topk(k, dim=1).indices
    target = torch.arange(rows, device=similarity.device).unsqueeze(1)
    return (topk == target).any(dim=1).float().mean().item()


def count_trainable_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
