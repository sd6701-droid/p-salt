from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .predictor import LatentPredictor, ReconstructionDecoder
from .vision_transformer import VideoTransformerEncoder


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
