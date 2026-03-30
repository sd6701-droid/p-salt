from __future__ import annotations

import torch
from torch import nn

from .utils.modules import TransformerBlock


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        decoder_dim: int,
        depth: int,
        num_heads: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.pos_proj = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, out_dim)

    def forward(
        self,
        visible_latents: torch.Tensor,
        visible_pos: torch.Tensor,
        masked_pos: torch.Tensor,
    ) -> torch.Tensor:
        visible = self.proj(visible_latents) + self.pos_proj(visible_pos)
        masked = self.mask_token.expand(visible.size(0), masked_pos.size(1), -1) + self.pos_proj(masked_pos)
        x = torch.cat([visible, masked], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, -masked_pos.size(1) :])


class LatentPredictor(nn.Module):
    def __init__(self, embed_dim: int, predictor_dim: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, predictor_dim)
        self.pos_proj = nn.Linear(embed_dim, predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(predictor_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim)

    def forward(
        self,
        visible_latents: torch.Tensor,
        visible_pos: torch.Tensor,
        masked_pos: torch.Tensor,
    ) -> torch.Tensor:
        visible = self.in_proj(visible_latents) + self.pos_proj(visible_pos)
        masked = self.mask_token.expand(visible.size(0), masked_pos.size(1), -1) + self.pos_proj(masked_pos)
        x = torch.cat([visible, masked], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, -masked_pos.size(1) :])
        return self.out_proj(x)
