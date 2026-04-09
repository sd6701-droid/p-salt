from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .vision_transformer import VideoTransformerEncoder


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(Path(path).expanduser(), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(ckpt)!r}")

    for key in ("model", "teacher", "student", "state_dict"):
        nested = ckpt.get(key)
        if isinstance(nested, dict):
            ckpt = nested
            break

    state_dict = {str(k): v for k, v in ckpt.items() if torch.is_tensor(v)}
    if not state_dict:
        raise ValueError("No tensor weights found in the DINOv2 checkpoint")
    return state_dict


def _infer_depth(state_dict: dict[str, torch.Tensor]) -> int:
    max_idx = -1
    prefix = "blocks."
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        idx_text = remainder.split(".", 1)[0]
        if idx_text.isdigit():
            max_idx = max(max_idx, int(idx_text))
    return max_idx + 1


def _inflate_patch_weight(
    weight_2d: torch.Tensor,
    target_patch_size: int,
    target_tubelet_size: int,
) -> torch.Tensor:
    resized = weight_2d
    if weight_2d.shape[-1] != target_patch_size or weight_2d.shape[-2] != target_patch_size:
        resized = F.interpolate(
            weight_2d,
            size=(target_patch_size, target_patch_size),
            mode="bicubic",
            align_corners=False,
        )
    inflated = resized.unsqueeze(2)
    if target_tubelet_size > 1:
        inflated = inflated.repeat(1, 1, target_tubelet_size, 1, 1) / float(target_tubelet_size)
    return inflated.contiguous()


def initialize_video_encoder_from_dinov2(
    encoder: VideoTransformerEncoder,
    checkpoint_path: str | Path,
) -> dict[str, int | str]:
    state_dict = _load_state_dict(checkpoint_path)

    source_patch = state_dict["patch_embed.proj.weight"]
    source_embed_dim = int(source_patch.shape[0])
    source_depth = _infer_depth(state_dict)
    target_depth = len(encoder.blocks)
    target_embed_dim = encoder.embed_dim

    if source_embed_dim != target_embed_dim:
        raise ValueError(
            "The provided DINOv2 checkpoint is not compatible with this student encoder. "
            f"checkpoint_embed_dim={source_embed_dim}, student_embed_dim={target_embed_dim}. "
            "A dinov2_vits14 checkpoint can initialize a 384-dim student, but not a vit_large "
            "student with 1024-dim embeddings."
        )
    if source_depth > target_depth:
        raise ValueError(
            "The provided DINOv2 checkpoint has more transformer blocks than the target student "
            f"encoder: checkpoint_depth={source_depth}, student_depth={target_depth}."
        )

    target_state = encoder.state_dict()
    mapped: dict[str, torch.Tensor] = {}

    mapped["patch_embed.proj.weight"] = _inflate_patch_weight(
        state_dict["patch_embed.proj.weight"],
        target_patch_size=encoder.patch_size,
        target_tubelet_size=encoder.tubelet_size,
    )
    mapped["patch_embed.proj.bias"] = state_dict["patch_embed.proj.bias"]
    if "norm.weight" in state_dict and "norm.weight" in target_state:
        mapped["norm.weight"] = state_dict["norm.weight"]
    if "norm.bias" in state_dict and "norm.bias" in target_state:
        mapped["norm.bias"] = state_dict["norm.bias"]

    for idx in range(source_depth):
        src_prefix = f"blocks.{idx}."
        dst_prefix = f"blocks.{idx}."
        block_pairs = {
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "attn.qkv.weight": "attn.in_proj_weight",
            "attn.qkv.bias": "attn.in_proj_bias",
            "attn.proj.weight": "attn.out_proj.weight",
            "attn.proj.bias": "attn.out_proj.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "mlp.fc1.weight": "mlp.net.0.weight",
            "mlp.fc1.bias": "mlp.net.0.bias",
            "mlp.fc2.weight": "mlp.net.2.weight",
            "mlp.fc2.bias": "mlp.net.2.bias",
        }
        for src_suffix, dst_suffix in block_pairs.items():
            src_key = src_prefix + src_suffix
            dst_key = dst_prefix + dst_suffix
            if src_key not in state_dict or dst_key not in target_state:
                continue
            if tuple(state_dict[src_key].shape) != tuple(target_state[dst_key].shape):
                raise ValueError(
                    f"Shape mismatch while loading DINOv2 weights for '{dst_key}': "
                    f"checkpoint_shape={tuple(state_dict[src_key].shape)} "
                    f"student_shape={tuple(target_state[dst_key].shape)}"
                )
            mapped[dst_key] = state_dict[src_key]

    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    # The expected missing keys are the student-only random leftovers when the target
    # encoder is deeper than the source checkpoint.
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading mapped DINOv2 weights: {unexpected}")

    return {
        "loaded_tensors": len(mapped),
        "source_depth": source_depth,
        "student_depth": target_depth,
        "source_embed_dim": source_embed_dim,
        "student_embed_dim": target_embed_dim,
        "missing_tensors_after_load": len(missing),
        "checkpoint_path": str(Path(checkpoint_path).expanduser()),
    }
