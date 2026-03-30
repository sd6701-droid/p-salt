from __future__ import annotations

import torch
from torch import nn

from .utils.patch_embed import VideoPatchEmbed
from .utils.pos_embs import build_3d_sincos_pos_embed
from .utils.modules import TransformerBlock


class VideoTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        tubelet_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_embed = VideoPatchEmbed(in_channels, embed_dim, tubelet_size, patch_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

    def embed(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, grid = self.patch_embed(video)
        pos = build_3d_sincos_pos_embed(*grid, self.embed_dim, video.device).expand(
            video.size(0), -1, -1
        )
        return tokens + pos, pos

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, pos = self.embed(video)
        return self.forward_tokens(tokens), pos
