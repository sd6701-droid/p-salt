from __future__ import annotations

from copy import deepcopy


VIT_ARCHITECTURES: dict[str, dict[str, float | int]] = {
    "vit_tiny": {
        "embed_dim": 192,
        "encoder_depth": 12,
        "encoder_heads": 3,
        "decoder_dim": 384,
        "decoder_depth": 4,
        "decoder_heads": 6,
        "mlp_ratio": 4.0,
    },
    "vit_small": {
        "embed_dim": 384,
        "encoder_depth": 12,
        "encoder_heads": 6,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 8,
        "mlp_ratio": 4.0,
    },
    "vit_base": {
        "embed_dim": 768,
        "encoder_depth": 12,
        "encoder_heads": 12,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 8,
        "mlp_ratio": 4.0,
    },
    "vit_large": {
        "embed_dim": 1024,
        "encoder_depth": 24,
        "encoder_heads": 16,
        "decoder_dim": 768,
        "decoder_depth": 8,
        "decoder_heads": 12,
        "mlp_ratio": 4.0,
    },
    "vit_h": {
        "embed_dim": 1280,
        "encoder_depth": 32,
        "encoder_heads": 16,
        "decoder_dim": 1024,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "mlp_ratio": 4.0,
    },
    "vit_huge": {
        "embed_dim": 1280,
        "encoder_depth": 32,
        "encoder_heads": 16,
        "decoder_dim": 1024,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "mlp_ratio": 4.0,
    },
}


def resolve_model_config(model_cfg: dict) -> dict:
    cfg = deepcopy(model_cfg)
    architecture = cfg.pop("architecture", None)
    if architecture is None:
        return cfg
    if architecture not in VIT_ARCHITECTURES:
        choices = ", ".join(sorted(VIT_ARCHITECTURES))
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {choices}")
    preset = deepcopy(VIT_ARCHITECTURES[architecture])
    preset.update(cfg)
    return preset
