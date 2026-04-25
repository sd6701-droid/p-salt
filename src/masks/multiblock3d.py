from __future__ import annotations

import math
import random
from typing import Sequence

import torch

from .default import sample_token_mask


def _sample_scale(scale: float | Sequence[float]) -> float:
    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)):
        if len(scale) != 2:
            raise ValueError(f"scale range must have length 2, got {scale}")
        return random.uniform(float(scale[0]), float(scale[1]))
    return float(scale)


def _sample_block_dims(
    grid_h: int,
    grid_w: int,
    area_scale: float | Sequence[float],
    aspect_ratio_range: tuple[float, float],
) -> tuple[int, int]:
    target_area = max(1.0, _sample_scale(area_scale) * grid_h * grid_w)
    aspect_ratio = random.uniform(*aspect_ratio_range)
    block_h = int(round(math.sqrt(target_area / aspect_ratio)))
    block_w = int(round(math.sqrt(target_area * aspect_ratio)))
    return max(1, min(grid_h, block_h)), max(1, min(grid_w, block_w))

def _trim_masked_tokens_to_batch_min(mask: torch.Tensor) -> torch.Tensor:
    masked_counts = mask.sum(dim=1)
    min_masked = int(masked_counts.min().item())
    num_tokens = int(mask.size(1))
    
    # Edge case: if min is 0 or all masked, force a sensible target
    if min_masked == 0:
        # Some sample had no masks at all. Pick a target near 90%.
        min_masked = int(0.9 * num_tokens)
    if min_masked >= num_tokens:
        min_masked = num_tokens - 1  # leave at least one visible
    
    trimmed = torch.zeros_like(mask)
    for idx in range(mask.size(0)):
        masked_ids = torch.nonzero(mask[idx], as_tuple=False).flatten()
        if masked_ids.numel() < min_masked:
            # This sample didn't have enough masks. Add random ones.
            visible_ids = torch.nonzero(~mask[idx], as_tuple=False).flatten()
            n_add = min_masked - masked_ids.numel()
            perm = torch.randperm(visible_ids.numel(), device=mask.device)[:n_add]
            extra = visible_ids[perm]
            kept = torch.cat([masked_ids, extra])
        else:
            perm = torch.randperm(masked_ids.numel(), device=mask.device)[:min_masked]
            kept = masked_ids[perm]
        trimmed[idx, kept] = True
    return trimmed

def _sample_profile_mask(
    *,
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    spatial_scale: float | Sequence[float],
    temporal_scale: float | Sequence[float],
    aspect_ratio_range: tuple[float, float],
    num_blocks: int,
    device: torch.device | None,
) -> torch.Tensor:
    return sample_spatiotemporal_block_mask(
        batch_size=batch_size,
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        spatial_scale=spatial_scale,
        temporal_scale=temporal_scale,
        aspect_ratio_range=aspect_ratio_range,
        num_blocks=num_blocks,
        device=device,
    )

def sample_spatiotemporal_block_mask(
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    spatial_scale: float | Sequence[float],
    temporal_scale: float | Sequence[float],
    aspect_ratio_range: tuple[float, float],
    num_blocks: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, grid_t, grid_h, grid_w, dtype=torch.bool, device=device)
    
    for idx in range(batch_size):
        for _ in range(num_blocks):
            # Sample fresh dimensions for THIS block
            block_t = max(1, min(grid_t, int(round(grid_t * _sample_scale(temporal_scale)))))
            block_h, block_w = _sample_block_dims(grid_h, grid_w, spatial_scale, aspect_ratio_range)
            
            start_t = 0 if block_t == grid_t else random.randint(0, grid_t - block_t)
            start_h = 0 if block_h == grid_h else random.randint(0, grid_h - block_h)
            start_w = 0 if block_w == grid_w else random.randint(0, grid_w - block_w)
            mask[idx, start_t : start_t + block_t, start_h : start_h + block_h, start_w : start_w + block_w] = True
    
    return mask.flatten(1)   # Return raw mask, do NOT trim here


def sample_multi_block_mask(
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    short_spatial_scale: float | Sequence[float],
    long_spatial_scale: float | Sequence[float],
    temporal_scale: float | Sequence[float],
    aspect_ratio_range: tuple[float, float],
    short_num_blocks: int = 8,
    long_num_blocks: int = 2,
    profile_sampling: str = "random",
    device: torch.device | None = None,
) -> torch.Tensor:
    if short_num_blocks <= 0 or long_num_blocks <= 0:
        raise ValueError("multiblock num_blocks values must be positive")

    profile = profile_sampling.lower()
    if profile == "short":
        mask = _sample_profile_mask(
            batch_size=batch_size,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            spatial_scale=short_spatial_scale,
            temporal_scale=temporal_scale,
            aspect_ratio_range=aspect_ratio_range,
            num_blocks=short_num_blocks,
            device=device,
        )
    elif profile == "long":
        mask = _sample_profile_mask(
            batch_size=batch_size,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            spatial_scale=long_spatial_scale,
            temporal_scale=temporal_scale,
            aspect_ratio_range=aspect_ratio_range,
            num_blocks=long_num_blocks,
            device=device,
        )
    elif profile in {"random", "mixed", "both", "all", "vjepa"}:
        short_mask = _sample_profile_mask(
            batch_size=batch_size,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            spatial_scale=short_spatial_scale,
            temporal_scale=temporal_scale,
            aspect_ratio_range=aspect_ratio_range,
            num_blocks=short_num_blocks,
            device=device,
        )
        long_mask = _sample_profile_mask(
            batch_size=batch_size,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            spatial_scale=long_spatial_scale,
            temporal_scale=temporal_scale,
            aspect_ratio_range=aspect_ratio_range,
            num_blocks=long_num_blocks,
            device=device,
        )
        mask = _trim_masked_tokens_to_batch_min(short_mask | long_mask)
    elif profile in {"either", "choose_one"}:
        spatial_scale, num_blocks = random.choice(
            [
                (short_spatial_scale, short_num_blocks),
                (long_spatial_scale, long_num_blocks),
            ]
        )
        mask = _sample_profile_mask(
            batch_size=batch_size,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            spatial_scale=spatial_scale,
            temporal_scale=temporal_scale,
            aspect_ratio_range=aspect_ratio_range,
            num_blocks=num_blocks,
            device=device,
        )
    else:
        raise ValueError(f"Unknown multiblock profile_sampling '{profile_sampling}'")

    return mask
