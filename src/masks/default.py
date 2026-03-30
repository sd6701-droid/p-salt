from __future__ import annotations

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
