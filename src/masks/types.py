from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IndexedMaskSet:
    encoder_ids: list[torch.Tensor]
    predictor_ids: list[torch.Tensor]
    num_tokens: int


def mask_batch_size(mask: torch.Tensor | IndexedMaskSet) -> int:
    if isinstance(mask, IndexedMaskSet):
        if not mask.predictor_ids:
            raise ValueError("IndexedMaskSet.predictor_ids must not be empty")
        return int(mask.predictor_ids[0].size(0))
    return int(mask.size(0))


def mask_num_views(mask: torch.Tensor | IndexedMaskSet) -> int:
    if isinstance(mask, IndexedMaskSet):
        return len(mask.predictor_ids)
    return 1


def mask_base_tokens_per_sample(mask: torch.Tensor | IndexedMaskSet) -> int:
    if isinstance(mask, IndexedMaskSet):
        return int(mask.num_tokens)
    return int(mask.size(1))


def mask_masked_tokens_per_sample(mask: torch.Tensor | IndexedMaskSet) -> int:
    if isinstance(mask, IndexedMaskSet):
        return int(sum(predictor_ids.size(1) for predictor_ids in mask.predictor_ids))
    return int(mask[0].sum().item())


def mask_visible_tokens_per_sample(mask: torch.Tensor | IndexedMaskSet) -> int:
    if isinstance(mask, IndexedMaskSet):
        return int(sum(encoder_ids.size(1) for encoder_ids in mask.encoder_ids))
    return int((~mask[0]).sum().item())


def mask_masked_tokens_total(mask: torch.Tensor | IndexedMaskSet) -> int:
    if isinstance(mask, IndexedMaskSet):
        return int(sum(predictor_ids.numel() for predictor_ids in mask.predictor_ids))
    return int(mask.sum().item())


def mask_visible_tokens_total(mask: torch.Tensor | IndexedMaskSet) -> int:
    if isinstance(mask, IndexedMaskSet):
        return int(sum(encoder_ids.numel() for encoder_ids in mask.encoder_ids))
    return int((~mask).sum().item())


def mask_total_tokens_total(mask: torch.Tensor | IndexedMaskSet) -> int:
    batch_size = mask_batch_size(mask)
    return int(mask_base_tokens_per_sample(mask) * batch_size * mask_num_views(mask))


def mask_ratio(mask: torch.Tensor | IndexedMaskSet) -> float:
    if isinstance(mask, IndexedMaskSet):
        return mask_masked_tokens_total(mask) / max(mask_total_tokens_total(mask), 1)
    return float(mask.float().mean().item())
