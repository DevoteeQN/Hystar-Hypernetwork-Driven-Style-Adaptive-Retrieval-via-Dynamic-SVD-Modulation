"""Lightweight hypernetwork used to predict query-conditioned singular-value updates.

The original research tree used torchmeta's MetaLinear/MetaSequential, but the
meta-learning parameter override path was never used by Hystar. This compact
version is functionally equivalent for normal forward passes and removes the
large torchmeta dependency.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class HyperGenerator(nn.Module):
    """MLP that maps a style embedding to singular-value increments."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        nonlinearity: str = "relu",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        activation = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "softplus": nn.Softplus,
        }
        if nonlinearity not in activation:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.extend([linear, activation[nonlinearity]()])
            prev_dim = dim

        out = nn.Linear(prev_dim, output_dim)
        nn.init.kaiming_normal_(out.weight, nonlinearity="linear")
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.mlp = nn.Sequential(*layers)

        if normalize:
            self.register_buffer("feature_mean", torch.zeros(output_dim))
            self.register_buffer("feature_std", torch.ones(output_dim))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not self.normalize:
            raise RuntimeError("set_stats() requires normalize=True")
        self.feature_mean.copy_(mean.detach())
        self.feature_std.copy_(std.detach())

    def forward(self, style_embedding: torch.Tensor) -> torch.Tensor:
        if style_embedding.ndim == 1:
            style_embedding = style_embedding.unsqueeze(0)
        output = self.mlp(style_embedding)
        if self.normalize:
            output = output * self.feature_std + self.feature_mean
        return output
