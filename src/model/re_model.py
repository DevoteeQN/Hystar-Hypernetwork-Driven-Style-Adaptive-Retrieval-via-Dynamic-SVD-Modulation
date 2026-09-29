"""Core Hystar CLIP model."""

from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
from torch import nn
import torch.nn.functional as F

from LoRACLIP.loraclip.loralib import layers as lora
from src.model.style_clip import apply_hystar_modulation
from src.utils.loss import StyleNCELoss


@dataclass
class HystarConfig:
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    style_dim: int = 768
    dynamic_layers: tuple[int, ...] = (3, 6, 9, 12)
    svd_rank: int | None = None
    temperature: float = 0.07
    gamma: float = 80.0
    ot_lambda: float = 1.0
    sinkhorn_iters: int = 50
    dino_repo: str = "facebookresearch/dinov2"
    dino_model: str = "dinov2_vitb14"


class HystarCLIP(nn.Module):
    """Hypernetwork-driven style-adaptive retrieval with a CLIP backbone."""

    def __init__(self, config: HystarConfig) -> None:
        super().__init__()
        self.config = config

        clip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(
            model_name=config.clip_model,
            pretrained=config.clip_pretrained,
        )
        self.clip = apply_hystar_modulation(
            clip,
            encoder_type="visual",
            rank=config.svd_rank,
            dynamic_layers=config.dynamic_layers,
            style_dim=config.style_dim,
        )
        self.tokenizer = open_clip.get_tokenizer(config.clip_model)

        # DINOv2 acts as the frozen style encoder. Using torch.hub removes the
        # need to vendor the full DINOv2 repository in this release package.
        self.gram_encoder = torch.hub.load(config.dino_repo, config.dino_model)
        self.gram_encoder.eval()
        self.gram_encoder.requires_grad_(False)

        self.style_nce = StyleNCELoss(
            temperature=config.temperature,
            ot_lambda=config.ot_lambda,
            sinkhorn_iters=config.sinkhorn_iters,
            negative_weight=config.gamma,
        )

    @torch.no_grad()
    def extract_style(self, images: torch.Tensor) -> torch.Tensor:
        features = self.gram_encoder.forward_features(images)
        cls_token = features["x_norm_clstoken"]
        return F.normalize(cls_token, p=2, dim=1)

    def _set_style_embedding(self, style_embedding: torch.Tensor) -> None:
        for module in self.clip.visual.modules():
            if isinstance(module, (lora.Linear, lora.MultiheadAttention)):
                module.set_z(style_embedding)

    def encode_image(self, images: torch.Tensor, style_embedding: torch.Tensor | None = None) -> torch.Tensor:
        if style_embedding is None:
            style_embedding = self.extract_style(images)
        self._set_style_embedding(style_embedding)
        return self.clip.encode_image(images)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(tokens)

    def forward(
        self,
        data: torch.Tensor,
        modality: str = "image",
        style_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if modality == "image":
            return self.encode_image(data, style_embedding=style_embedding)
        if modality == "text":
            return self.encode_text(data)
        raise ValueError("modality must be 'image' or 'text'")

    def retrieval_loss(self, styled_query: torch.Tensor, natural_target: torch.Tensor) -> torch.Tensor:
        return self.style_nce(styled_query, natural_target)
