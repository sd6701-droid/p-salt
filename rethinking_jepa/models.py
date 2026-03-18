from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _make_sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    omega = torch.arange(half, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, half)))
    out = positions[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_3d_sincos_pos_embed(
    t_size: int,
    h_size: int,
    w_size: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    if dim % 6 != 0:
        raise ValueError(f"embed_dim must be divisible by 6, got {dim}")
    axis_dim = dim // 3
    if axis_dim % 2 != 0:
        raise ValueError(f"embed_dim // 3 must be even, got {axis_dim}")
    tt = torch.arange(t_size, device=device, dtype=torch.float32)
    yy = torch.arange(h_size, device=device, dtype=torch.float32)
    xx = torch.arange(w_size, device=device, dtype=torch.float32)
    grid_t, grid_y, grid_x = torch.meshgrid(tt, yy, xx, indexing="ij")
    emb_t = _make_sincos_1d(grid_t.reshape(-1), axis_dim)
    emb_y = _make_sincos_1d(grid_y.reshape(-1), axis_dim)
    emb_x = _make_sincos_1d(grid_x.reshape(-1), axis_dim)
    return torch.cat([emb_t, emb_y, emb_x], dim=1).unsqueeze(0)


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return tokens.gather(dim=1, index=gather_idx)


def _ids_from_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(mask.size(1), device=mask.device).unsqueeze(0).expand_as(mask)
    visible_counts = (~mask).sum(dim=1)
    masked_counts = mask.sum(dim=1)
    if visible_counts.min() != visible_counts.max() or masked_counts.min() != masked_counts.max():
        raise ValueError("All samples must have the same number of visible and masked tokens")
    visible_ids = ids.masked_select(~mask).view(mask.size(0), -1)
    masked_ids = ids.masked_select(mask).view(mask.size(0), -1)
    return visible_ids, masked_ids


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


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


class ReconstructionDecoder(nn.Module):
    def __init__(self, embed_dim: int, decoder_dim: int, depth: int, num_heads: int, out_dim: int) -> None:
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


def patchify_video(
    video: torch.Tensor,
    tubelet_size: int,
    patch_size: int,
) -> torch.Tensor:
    b, c, t, h, w = video.shape
    if t % tubelet_size != 0 or h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Video dimensions must be divisible by tubelet and patch sizes")
    t_grid = t // tubelet_size
    h_grid = h // patch_size
    w_grid = w // patch_size
    x = video.view(
        b,
        c,
        t_grid,
        tubelet_size,
        h_grid,
        patch_size,
        w_grid,
        patch_size,
    )
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.view(b, t_grid * h_grid * w_grid, c * tubelet_size * patch_size * patch_size)


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor


class TeacherModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        frames: int,
        image_size: int,
        embed_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        mlp_ratio: float,
        tubelet_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.encoder = VideoTransformerEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
        )
        patch_dim = in_channels * tubelet_size * patch_size * patch_size
        self.decoder = ReconstructionDecoder(
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            out_dim=patch_dim,
        )
        self.frames = frames
        self.image_size = image_size

    def forward(self, video: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        tokens, pos = self.encoder.embed(video)
        visible_ids, masked_ids = _ids_from_mask(mask)
        visible_tokens = _gather_tokens(tokens, visible_ids)
        visible_pos = _gather_tokens(pos, visible_ids)
        masked_pos = _gather_tokens(pos, masked_ids)
        visible_latents = self.encoder.forward_tokens(visible_tokens)
        prediction = self.decoder(visible_latents, visible_pos, masked_pos)
        patches = patchify_video(
            video,
            tubelet_size=self.encoder.tubelet_size,
            patch_size=self.encoder.patch_size,
        )
        target = _gather_tokens(patches, masked_ids)
        return ModelOutput(prediction=prediction, target=target, mask=mask)


class StudentModel(nn.Module):
    def __init__(
        self,
        teacher: TeacherModel,
        predictor_dim: int,
        predictor_depth: int,
        predictor_heads: int,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        encoder = teacher.encoder
        self.student = VideoTransformerEncoder(
            in_channels=encoder.patch_embed.proj.in_channels,
            embed_dim=encoder.embed_dim,
            depth=len(encoder.blocks),
            num_heads=encoder.blocks[0].attn.num_heads,
            mlp_ratio=4.0,
            tubelet_size=encoder.tubelet_size,
            patch_size=encoder.patch_size,
        )
        self.predictor = LatentPredictor(
            embed_dim=encoder.embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )

    def forward(self, video: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        with torch.no_grad():
            teacher_tokens, _ = self.teacher.encoder(video)
        student_tokens, pos = self.student.embed(video)
        visible_ids, masked_ids = _ids_from_mask(mask)
        visible_tokens = _gather_tokens(student_tokens, visible_ids)
        visible_pos = _gather_tokens(pos, visible_ids)
        masked_pos = _gather_tokens(pos, masked_ids)
        visible_latents = self.student.forward_tokens(visible_tokens)
        prediction = self.predictor(visible_latents, visible_pos, masked_pos)
        target = _gather_tokens(teacher_tokens, masked_ids)
        return ModelOutput(prediction=prediction, target=target, mask=mask)
