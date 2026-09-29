"""Contrastive objectives used by Hystar."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def sinkhorn(kernel: torch.Tensor, max_iter: int = 50, eps: float = 1e-9) -> torch.Tensor:
    """Normalize a non-negative matrix toward a doubly-stochastic transport plan."""
    batch = kernel.size(0)
    u = kernel.new_ones(batch) / batch
    v = kernel.new_ones(batch) / batch
    kernel = kernel.clamp_min(eps)
    for _ in range(max_iter):
        u = 1.0 / (kernel @ v).clamp_min(eps)
        v = 1.0 / (u @ kernel).clamp_min(eps)
    return torch.diag(u) @ kernel @ torch.diag(v)


class StyleNCELoss(nn.Module):
    """OT-weighted contrastive loss used by Hystar.

    ``negative_weight`` corresponds to ``gamma`` in the paper.  The OT
    term below is written in a paper-aligned form while remaining numerically
    equivalent to the original research implementation.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        ot_lambda: float = 1.0,
        sinkhorn_iters: int = 50,
        negative_weight: float = 80.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.ot_lambda = ot_lambda
        self.sinkhorn_iters = sinkhorn_iters
        self.negative_weight = negative_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        if query_features.shape != target_features.shape:
            raise ValueError(
                f"query_features and target_features must have the same shape, got "
                f"{query_features.shape} and {target_features.shape}"
            )

        query_features = F.normalize(query_features, p=2, dim=1, eps=1e-7)
        target_features = F.normalize(target_features, p=2, dim=1, eps=1e-7)
        similarity = query_features @ target_features.t()

        batch = similarity.size(0)
        eye = torch.eye(batch, device=similarity.device, dtype=torch.bool)

        # Paper-aligned notation.  The research implementation uses the
        # signed OT score c_ij = -(1 - sim_ij) and forms a Gibbs kernel
        # K_ij = exp(-c_ij / lambda).  Therefore
        #
        #     K_ij = exp((1 - sim_ij) / lambda),
        #
        # which is exactly the weighting term written in the paper.  Keeping
        # the explicit minus signs here makes the connection to the standard
        # Sinkhorn / entropic-OT form clear without changing any computation.
        ot_cost = -(1.0 - similarity)
        ot_cost = ot_cost.masked_fill(eye, 1e9)
        kernel = torch.exp(-ot_cost / self.ot_lambda).masked_fill(eye, 0.0)
        transport = sinkhorn(kernel, max_iter=self.sinkhorn_iters)
        weights = transport * self.negative_weight + 1e-8

        positive_logits = torch.diag(similarity).unsqueeze(1) / self.temperature
        negative_logits = similarity / self.temperature + torch.log(weights)
        negative_logits = negative_logits.masked_fill(eye, -1e9)
        logits = torch.cat([positive_logits, negative_logits], dim=1)

        labels = torch.zeros(batch, dtype=torch.long, device=similarity.device)
        return self.cross_entropy(logits, labels)
