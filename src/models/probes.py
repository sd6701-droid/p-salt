from __future__ import annotations

import torch
from torch import nn

from .jepa import ModelOutput, patchify_video
from .predictor import ReconstructionDecoder
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


class FrozenStudentPixelProbe(nn.Module):
    def __init__(
        self,
        student_encoder: VideoTransformerEncoder,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
    ) -> None:
        super().__init__()
        self.encoder = student_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        in_channels = self.encoder.patch_embed.proj.in_channels
        patch_dim = in_channels * self.encoder.tubelet_size * self.encoder.patch_size * self.encoder.patch_size
        self.decoder = ReconstructionDecoder(
            embed_dim=self.encoder.embed_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            out_dim=patch_dim,
        )

    def forward(self, video: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        with torch.no_grad():
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
