from __future__ import annotations

import math
from multiprocessing import Value
from typing import Sequence

import torch

from .types import IndexedMaskSet


def _to_range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Expected length-2 range, got {value}")
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


class _MaskGenerator:
    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
    ):
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2

        self.crop_size = crop_size
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep
        self._itr_counter = Value("i", -1)
        self.inv_block = inv_block

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + rand * (max_ar - min_ar)

        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)
        return (t, h, w)

    def _sample_block_mask(self, block_size):
        t, h, w = block_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0
        return mask

    def __call__(self, batch_size):
        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)
        block_size = self._sample_block_size(
            generator=generator,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(block_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        if self.full_complement:
            collated_masks_pred = [
                torch.tensor(
                    sorted(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist())),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist())),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        if self.inv_block:
            return collated_masks_pred, collated_masks_enc
        return collated_masks_enc, collated_masks_pred


class VJEPAMultiMaskSampler:
    def __init__(
        self,
        *,
        crop_size: int | tuple[int, int],
        num_frames: int,
        patch_size: int,
        tubelet_size: int,
        mask_cfgs: list[dict],
    ) -> None:
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, crop_size)

        self.mask_generators = [
            _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=(patch_size, patch_size),
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=_to_range(cfg["spatial_scale"]),
                temporal_pred_mask_scale=_to_range(cfg["temporal_scale"]),
                aspect_ratio=tuple(cfg["aspect_ratio"]),
                npred=int(cfg["num_blocks"]),
                max_context_frames_ratio=float(cfg.get("max_temporal_keep", 1.0)),
                max_keep=cfg.get("max_keep"),
                full_complement=bool(cfg.get("full_complement", False)),
                pred_full_complement=bool(cfg.get("pred_full_complement", False)),
                inv_block=bool(cfg.get("inv_block", False)),
            )
            for cfg in mask_cfgs
        ]

        duration = num_frames // tubelet_size
        if crop_size[0] % patch_size != 0 or crop_size[1] % patch_size != 0:
            raise ValueError("crop_size must be divisible by patch_size")
        height = crop_size[0] // patch_size
        width = crop_size[1] // patch_size
        self.num_tokens = int(duration * height * width)

    def __call__(self, batch_size: int, device: torch.device) -> IndexedMaskSet:
        encoder_ids: list[torch.Tensor] = []
        predictor_ids: list[torch.Tensor] = []
        non_blocking = device.type == "cuda"
        for mask_generator in self.mask_generators:
            masks_enc, masks_pred = mask_generator(batch_size)
            encoder_ids.append(masks_enc.to(device=device, non_blocking=non_blocking))
            predictor_ids.append(masks_pred.to(device=device, non_blocking=non_blocking))
        return IndexedMaskSet(
            encoder_ids=encoder_ids,
            predictor_ids=predictor_ids,
            num_tokens=self.num_tokens,
        )
