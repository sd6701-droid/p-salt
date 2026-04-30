from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.masks.types import IndexedMaskSet

from .predictor import LatentPredictor, ReconstructionDecoder
from .vision_transformer import VideoTransformerEncoder


def get_related_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
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


def _mask_pairs(mask: torch.Tensor | IndexedMaskSet) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(mask, IndexedMaskSet):
        return list(zip(mask.encoder_ids, mask.predictor_ids, strict=True))
    visible_ids, masked_ids = _ids_from_mask(mask)
    return [(visible_ids, masked_ids)]


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


def normalize_patch_targets(targets: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    mean = targets.mean(dim=-1, keepdim=True)
    var = targets.var(dim=-1, unbiased=False, keepdim=True)
    return (targets - mean) / torch.sqrt(var + eps)


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor | IndexedMaskSet


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
        norm_pix_loss: bool = False,
        norm_pix_eps: float = 1.0e-6,
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
        self.norm_pix_loss = norm_pix_loss
        self.norm_pix_eps = norm_pix_eps

    def forward(self, video: torch.Tensor, mask: torch.Tensor | IndexedMaskSet) -> ModelOutput:
        tokens, pos = self.encoder.embed(video)
        patches = patchify_video(
            video,
            tubelet_size=self.encoder.tubelet_size,
            patch_size=self.encoder.patch_size,
        )
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for visible_ids, masked_ids in _mask_pairs(mask):
            visible_tokens = get_related_tokens(tokens, visible_ids)
            visible_pos = get_related_tokens(pos, visible_ids)
            masked_pos = get_related_tokens(pos, masked_ids)
            visible_latents = self.encoder.forward_tokens(visible_tokens)
            predictions.append(self.decoder(visible_latents, visible_pos, masked_pos))
            target = get_related_tokens(patches, masked_ids)
            if self.norm_pix_loss:
                target = normalize_patch_targets(target, eps=self.norm_pix_eps)
            targets.append(target)
        prediction = torch.cat(predictions, dim=1)
        target = torch.cat(targets, dim=1)
        return ModelOutput(prediction=prediction, target=target, mask=mask)


class StudentModel(nn.Module):
    def __init__(
        self,
        teacher: TeacherModel,
        predictor_dim: int,
        predictor_depth: int,
        predictor_heads: int,
        student_in_channels: int | None = None,
        student_embed_dim: int | None = None,
        student_depth: int | None = None,
        student_heads: int | None = None,
        student_mlp_ratio: float = 4.0,
        student_tubelet_size: int | None = None,
        student_patch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        encoder = teacher.encoder
        student_in_channels = (
            encoder.patch_embed.proj.in_channels if student_in_channels is None else student_in_channels
        )
        student_embed_dim = encoder.embed_dim if student_embed_dim is None else student_embed_dim
        student_depth = len(encoder.blocks) if student_depth is None else student_depth
        student_heads = encoder.blocks[0].attn.num_heads if student_heads is None else student_heads
        student_tubelet_size = encoder.tubelet_size if student_tubelet_size is None else student_tubelet_size
        student_patch_size = encoder.patch_size if student_patch_size is None else student_patch_size

        if student_tubelet_size != encoder.tubelet_size or student_patch_size != encoder.patch_size:
            raise ValueError(
                "Student and teacher must use the same tubelet_size and patch_size so token grids match"
            )

        self.student = VideoTransformerEncoder(
            in_channels=student_in_channels,
            embed_dim=student_embed_dim,
            depth=student_depth,
            num_heads=student_heads,
            mlp_ratio=student_mlp_ratio,
            tubelet_size=student_tubelet_size,
            patch_size=student_patch_size,
        )
        self.predictor = LatentPredictor(
            embed_dim=student_embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            target_embed_dim=encoder.embed_dim,
        )

    def forward(self, video: torch.Tensor, mask: torch.Tensor | IndexedMaskSet) -> ModelOutput:
        with torch.no_grad():
            teacher_tokens, _ = self.teacher.encoder(video)
        student_tokens, pos = self.student.embed(video)
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for visible_ids, masked_ids in _mask_pairs(mask):
            visible_tokens = get_related_tokens(student_tokens, visible_ids)
            visible_pos = get_related_tokens(pos, visible_ids)
            masked_pos = get_related_tokens(pos, masked_ids)
            visible_latents = self.student.forward_tokens(visible_tokens)
            predictions.append(self.predictor(visible_latents, visible_pos, masked_pos))
            targets.append(get_related_tokens(teacher_tokens, masked_ids))
        prediction = torch.cat(predictions, dim=1)
        target = torch.cat(targets, dim=1)
        return ModelOutput(prediction=prediction, target=target, mask=mask)
