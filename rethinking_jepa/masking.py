from __future__ import annotations

import math
import random

import torch


def sample_token_mask(
    batch_size: int,
    num_tokens: int,
    mask_ratio: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    num_masked = max(1, min(num_tokens - 1, int(num_tokens * mask_ratio)))
    noise = torch.rand(batch_size, num_tokens, device=device)
    ids = noise.argsort(dim=1)
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, ids[:, :num_masked], True)
    return mask


def _sample_block_dims(
    grid_h: int,
    grid_w: int,
    area_scale: float,
    aspect_ratio_range: tuple[float, float],
) -> tuple[int, int]:
    target_area = max(1.0, area_scale * grid_h * grid_w)
    aspect_ratio = random.uniform(*aspect_ratio_range)
    block_h = int(round(math.sqrt(target_area / aspect_ratio)))
    block_w = int(round(math.sqrt(target_area * aspect_ratio)))
    return max(1, min(grid_h, block_h)), max(1, min(grid_w, block_w))


def sample_spatiotemporal_block_mask(
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    spatial_scale: float,
    temporal_scale: float,
    aspect_ratio_range: tuple[float, float],
    device: torch.device | None = None,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, grid_t, grid_h, grid_w, dtype=torch.bool, device=device)
    block_t = max(1, min(grid_t, int(round(grid_t * temporal_scale))))
    block_h, block_w = _sample_block_dims(grid_h, grid_w, spatial_scale, aspect_ratio_range)
    for idx in range(batch_size):
        start_t = 0 if block_t == grid_t else random.randint(0, grid_t - block_t)
        start_h = 0 if block_h == grid_h else random.randint(0, grid_h - block_h)
        start_w = 0 if block_w == grid_w else random.randint(0, grid_w - block_w)
        mask[idx, start_t : start_t + block_t, start_h : start_h + block_h, start_w : start_w + block_w] = True
    return mask.flatten(1)


def sample_multi_block_mask(
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    short_spatial_scale: float,
    long_spatial_scale: float,
    temporal_scale: float,
    aspect_ratio_range: tuple[float, float],
    device: torch.device | None = None,
) -> torch.Tensor:
    spatial_scale = random.choice([short_spatial_scale, long_spatial_scale])
    mask = sample_spatiotemporal_block_mask(
        batch_size=batch_size,
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        spatial_scale=spatial_scale,
        temporal_scale=temporal_scale,
        aspect_ratio_range=aspect_ratio_range,
        device=device,
    )
    if mask.all(dim=1).any():
        return sample_token_mask(batch_size, grid_t * grid_h * grid_w, 0.75, device=device)
    return mask
