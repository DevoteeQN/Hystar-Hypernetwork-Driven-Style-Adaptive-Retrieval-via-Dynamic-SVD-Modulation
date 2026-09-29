"""Inject Hystar's dynamic/static SVD modulation into an OpenCLIP ViT."""

from __future__ import annotations

from collections.abc import Sequence

from LoRACLIP.loraclip.loralib import layers as lora


def freeze_all_except_modulation(model):
    """Freeze the backbone and train only delta/gate/hypernetwork parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "delta" in name or "gate" in name:
            parameter.requires_grad = True
    return model


def apply_hystar_modulation(
    model,
    encoder_type: str = "visual",
    rank: int | None = None,
    dynamic_layers: Sequence[int] = (3, 6, 9, 12),
    mlp: bool = True,
    attention: bool = True,
    style_dim: int = 768,
):
    """Replace selected attention and MLP projections with SVD-modulated layers.

    Attention layers in ``dynamic_layers`` use a hypernetwork-conditioned
    singular-value update. MLP projections use globally learned static
    singular-value offsets. Layer indices are zero-based; (3, 6, 9, 12)
    correspond to the 4th, 7th, 10th, and 13th transformer blocks.
    """
    if encoder_type == "visual":
        encoder = model.visual.transformer
    elif encoder_type == "text":
        encoder = model.transformer
    else:
        raise ValueError("encoder_type must be 'visual' or 'text'")

    dynamic_layers = set(dynamic_layers)

    for layer_idx, block in enumerate(encoder.resblocks):
        if attention and hasattr(block, "attn") and layer_idx in dynamic_layers:
            old_attn = block.attn
            new_attn = lora.MultiheadAttention(
                r=rank,
                embed_dim=old_attn.embed_dim,
                num_heads=old_attn.num_heads,
                dropout=old_attn.dropout,
                bias=old_attn.in_proj_bias is not None,
                add_zero_attn=old_attn.add_zero_attn,
                kdim=old_attn.kdim,
                vdim=old_attn.vdim,
                use_hyper=True,
                batch_first=old_attn.batch_first,
                latent_dim=style_dim,
            )
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            new_attn.init_svd()
            block.attn = new_attn

        if mlp and hasattr(block, "mlp"):
            for name in ("c_fc", "c_proj"):
                old_linear = getattr(block.mlp, name)
                new_linear = lora.Linear(
                    in_features=old_linear.in_features,
                    out_features=old_linear.out_features,
                    bias=old_linear.bias is not None,
                    r=rank,
                    use_hyper=False,
                    latent_dim=style_dim,
                )
                new_linear.load_state_dict(old_linear.state_dict(), strict=False)
                new_linear.init_svd()
                setattr(block.mlp, name, new_linear)

    return freeze_all_except_modulation(model)
