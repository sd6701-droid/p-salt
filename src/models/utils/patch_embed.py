from __future__ import annotations

import torch
from torch import nn


class VideoPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        tubelet_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(video)
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return x, grid
