from __future__ import annotations

import torch


def _make_sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 0:
        return torch.zeros((positions.numel(), 0), device=positions.device, dtype=torch.float32)
    half = dim // 2
    omega = torch.arange(half, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, half)))
    out = positions[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _split_3d_embed_dim(dim: int) -> tuple[int, int, int]:
    if dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {dim}")

    # Match the V-JEPA-style allocation: dedicate half the pairs to time and
    # split the remaining pairs equally across height and width.
    total_pairs = dim // 2
    depth_pairs = total_pairs // 2
    spatial_pairs = total_pairs - depth_pairs
    height_pairs = spatial_pairs // 2
    width_pairs = spatial_pairs - height_pairs
    return 2 * depth_pairs, 2 * height_pairs, 2 * width_pairs


def build_3d_sincos_pos_embed(
    t_size: int,
    h_size: int,
    w_size: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    tt = torch.arange(t_size, device=device, dtype=torch.float32)
    yy = torch.arange(h_size, device=device, dtype=torch.float32)
    xx = torch.arange(w_size, device=device, dtype=torch.float32)
    grid_t, grid_y, grid_x = torch.meshgrid(tt, yy, xx, indexing="ij")
    t_dim, h_dim, w_dim = _split_3d_embed_dim(dim)
    emb_t = _make_sincos_1d(grid_t.reshape(-1), t_dim)
    emb_y = _make_sincos_1d(grid_y.reshape(-1), h_dim)
    emb_x = _make_sincos_1d(grid_x.reshape(-1), w_dim)
    return torch.cat([emb_t, emb_y, emb_x], dim=1).unsqueeze(0)
