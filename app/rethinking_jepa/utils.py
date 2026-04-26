from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset

from src.datasets.data_manager import build_video_dataset
from src.masks.default import sample_token_mask
from src.masks.multiblock3d import sample_multi_block_mask
from src.masks.types import (
    IndexedMaskSet,
    mask_base_tokens_per_sample,
    mask_batch_size,
    mask_masked_tokens_per_sample,
    mask_masked_tokens_total,
    mask_num_views,
    mask_ratio,
    mask_total_tokens_total,
    mask_visible_tokens_per_sample,
    mask_visible_tokens_total,
)
from src.masks.vjepa_exact import VJEPAMultiMaskSampler
from src.models.architectures import resolve_model_config
from src.models.dinov2_init import initialize_video_encoder_from_dinov2
from src.models.jepa import StudentModel, TeacherModel
from src.utils.schedulers import CosineScheduler


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_source_from_dataset(dataset: Any, index: int) -> str:
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, index)
        previous_size = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        return _sample_source_from_dataset(dataset.datasets[dataset_idx], index - previous_size)

    if hasattr(dataset, "video_paths"):
        return str(Path(dataset.video_paths[index]).expanduser().resolve())

    if hasattr(dataset, "archive_entries") and hasattr(dataset, "archive_path"):
        archive_path = Path(dataset.archive_path).expanduser().resolve()
        return f"{archive_path}::{dataset.archive_entries[index]}"

    if hasattr(dataset, "samples"):
        sample = dataset.samples[index]
        if isinstance(sample, tuple) and sample:
            return str(Path(sample[0]).expanduser().resolve())

    return f"{type(dataset).__name__}:{index}"


def _normalize_sample(sample: Any, source: str) -> dict[str, Any]:
    if isinstance(sample, dict):
        item = dict(sample)
        item.setdefault("source", source)
        return item
    return {"video": sample, "source": source}


class _DatasetWithMetadata(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _normalize_sample(self.dataset[index], _sample_source_from_dataset(self.dataset, index))


class _IterableDatasetWithMetadata(IterableDataset[dict[str, Any]]):
    def __init__(self, dataset: IterableDataset[Any]) -> None:
        self.dataset = dataset

    def __iter__(self):
        for index, sample in enumerate(self.dataset):
            source = sample.get("source") if isinstance(sample, dict) else None
            if source is None:
                source = f"{type(self.dataset).__name__}:{index}"
            yield _normalize_sample(sample, str(source))


def build_loader(cfg: dict, *, include_metadata: bool = False) -> DataLoader:
    dataset = build_video_dataset(cfg)
    if include_metadata:
        if isinstance(dataset, IterableDataset):
            dataset = _IterableDatasetWithMetadata(dataset)
        else:
            dataset = _DatasetWithMetadata(dataset)
    train_num_workers = int(cfg["train"].get("num_workers", 0))
    loader_kwargs = {
        "batch_size": cfg["train"]["device_batch_size"],
        "shuffle": not isinstance(dataset, IterableDataset),
        "num_workers": train_num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if train_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def resolve_batch_settings(cfg: dict) -> tuple[int, int, int]:
    train_cfg = cfg["train"]
    device_batch_size = int(train_cfg["device_batch_size"])
    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))

    if device_batch_size <= 0:
        raise ValueError(f"train.device_batch_size must be positive, got {device_batch_size}")
    if accumulation_steps <= 0:
        raise ValueError(f"train.accumulation_steps must be positive, got {accumulation_steps}")
    effective_batch_size = device_batch_size * accumulation_steps
    return device_batch_size, accumulation_steps, effective_batch_size


def resolve_dataset_size(loader: DataLoader) -> int | None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return len(dataset)
    except TypeError:
        return None


def unpack_video_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    if isinstance(batch, dict):
        video = batch["video"]
        sources = batch.get("source", batch.get("path", []))
    else:
        video = batch
        sources = []

    if isinstance(sources, str):
        source_list = [sources]
    elif isinstance(sources, (list, tuple)):
        source_list = [str(source) for source in sources]
    else:
        source_list = [] if sources is None else [str(sources)]

    return video.to(device, non_blocking=(device.type == "cuda")), source_list


def resolve_max_steps(cfg: dict) -> int:
    max_steps = int(cfg["train"]["max_steps"])
    if max_steps <= 0:
        raise ValueError(f"train.max_steps must be positive, got {max_steps}")
    return max_steps


def build_teacher_from_cfg(cfg: dict, device: torch.device) -> tuple[TeacherModel, dict]:
    model_cfg = resolve_model_config(cfg["model"])
    loss_cfg = cfg.get("loss", {})
    teacher = TeacherModel(
        **model_cfg,
        frames=cfg["data"]["frames"],
        image_size=cfg["data"]["input_size"],
        norm_pix_loss=bool(loss_cfg.get("norm_pix_loss", False)),
        norm_pix_eps=float(loss_cfg.get("norm_pix_eps", 1.0e-6)),
    ).to(device)
    return teacher, model_cfg


def build_student_from_cfg(cfg: dict, teacher: TeacherModel, device: torch.device) -> StudentModel:
    student_backbone_cfg = cfg.get("student_model")
    if student_backbone_cfg is None:
        teacher_encoder = teacher.encoder
        student_kwargs = {
            "student_in_channels": teacher_encoder.patch_embed.proj.in_channels,
            "student_embed_dim": teacher_encoder.embed_dim,
            "student_depth": len(teacher_encoder.blocks),
            "student_heads": teacher_encoder.blocks[0].attn.num_heads,
            "student_mlp_ratio": 4.0,
            "student_tubelet_size": teacher_encoder.tubelet_size,
            "student_patch_size": teacher_encoder.patch_size,
        }
    else:
        resolved_student_cfg = resolve_model_config(student_backbone_cfg)
        student_kwargs = {
            "student_in_channels": resolved_student_cfg["in_channels"],
            "student_embed_dim": resolved_student_cfg["embed_dim"],
            "student_depth": resolved_student_cfg["encoder_depth"],
            "student_heads": resolved_student_cfg["encoder_heads"],
            "student_mlp_ratio": float(resolved_student_cfg.get("mlp_ratio", 4.0)),
            "student_tubelet_size": resolved_student_cfg["tubelet_size"],
            "student_patch_size": resolved_student_cfg["patch_size"],
        }

    student = StudentModel(
        teacher=teacher,
        predictor_dim=cfg["student"]["predictor_dim"],
        predictor_depth=cfg["student"]["predictor_depth"],
        predictor_heads=cfg["student"]["predictor_heads"],
        **student_kwargs,
    ).to(device)

    init_ckpt = cfg.get("student", {}).get("init_from_dinov2_checkpoint")
    if init_ckpt:
        info = initialize_video_encoder_from_dinov2(student.student, init_ckpt)
        print(
            "initialized student encoder from DINOv2 checkpoint "
            f"path={info['checkpoint_path']} loaded_tensors={info['loaded_tensors']} "
            f"source_depth={info['source_depth']} student_depth={info['student_depth']}"
        )
    return student


def build_scheduler(
    cfg: dict,
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int | None = None,
) -> CosineScheduler:
    return CosineScheduler(
        optimizer=optimizer,
        total_steps=cfg["train"]["max_steps"] if total_steps is None else total_steps,
        warmup_steps=cfg["optimizer"]["warmup_steps"],
        start_lr=cfg["optimizer"]["start_lr"],
        peak_lr=cfg["optimizer"]["lr"],
        final_lr=cfg["optimizer"]["final_lr"],
        start_weight_decay=cfg["optimizer"]["start_weight_decay"],
        end_weight_decay=cfg["optimizer"]["end_weight_decay"],
    )


def _build_vjepa_mask_cfgs(masking_cfg: dict) -> list[dict]:
    return [
        {
            "aspect_ratio": list(masking_cfg["mask_aspect_ratio"]),
            "full_complement": False,
            "max_keep": None,
            "max_temporal_keep": float(masking_cfg.get("max_temporal_keep", 1.0)),
            "num_blocks": int(masking_cfg.get("short_num_blocks", 8)),
            "spatial_scale": [float(masking_cfg["short_spatial_mask_scale"])] * 2,
            "temporal_scale": [float(masking_cfg["temporal_mask_scale"])] * 2,
        },
        {
            "aspect_ratio": list(masking_cfg["mask_aspect_ratio"]),
            "full_complement": False,
            "max_keep": None,
            "max_temporal_keep": float(masking_cfg.get("max_temporal_keep", 1.0)),
            "num_blocks": int(masking_cfg.get("long_num_blocks", 2)),
            "spatial_scale": [float(masking_cfg["long_spatial_mask_scale"])] * 2,
            "temporal_scale": [float(masking_cfg["temporal_mask_scale"])] * 2,
        },
    ]


def _get_or_create_vjepa_sampler(patch_embed, cfg: dict) -> VJEPAMultiMaskSampler:
    sampler = getattr(patch_embed, "_vjepa_mask_sampler", None)
    if sampler is not None:
        return sampler
    sampler = VJEPAMultiMaskSampler(
        crop_size=int(cfg["data"]["input_size"]),
        num_frames=int(cfg["data"]["frames"]),
        patch_size=int(patch_embed.patch_size),
        tubelet_size=int(patch_embed.tubelet_size),
        mask_cfgs=_build_vjepa_mask_cfgs(cfg["masking"]),
    )
    setattr(patch_embed, "_vjepa_mask_sampler", sampler)
    return sampler


def sample_mask_from_model(
    patch_embed,
    video: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor | IndexedMaskSet:
    with torch.no_grad():
        _, grid = patch_embed(video)
    masking_cfg = cfg["masking"]
    strategy = masking_cfg.get("strategy", "multiblock3d").lower()

    if strategy in {"random", "random_token", "token"}:
        return sample_token_mask(
            batch_size=video.size(0),
            num_tokens=int(grid[0] * grid[1] * grid[2]),
            mask_ratio=float(masking_cfg.get("random_mask_ratio", masking_cfg.get("mask_ratio", 0.75))),
            device=device,
        )

    if strategy in {"multiblock3d", "multiblock", "block"}:
        return sample_multi_block_mask(
            batch_size=video.size(0),
            grid_t=grid[0],
            grid_h=grid[1],
            grid_w=grid[2],
            short_spatial_scale=masking_cfg["short_spatial_mask_scale"],
            long_spatial_scale=masking_cfg["long_spatial_mask_scale"],
            temporal_scale=masking_cfg["temporal_mask_scale"],
            aspect_ratio_range=tuple(masking_cfg["mask_aspect_ratio"]),
            short_num_blocks=int(masking_cfg.get("short_num_blocks", 8)),
            long_num_blocks=int(masking_cfg.get("long_num_blocks", 2)),
            profile_sampling=str(masking_cfg.get("multiblock_profile_sampling", "random")),
            target_mask_ratio=(
                float(masking_cfg["target_mask_ratio"])
                if masking_cfg.get("target_mask_ratio") is not None
                else None
            ),
            device=device,
        )

    if strategy in {"multiseq_multiblock3d", "vjepa_exact", "vjepa_multiblock3d"}:
        sampler = _get_or_create_vjepa_sampler(patch_embed, cfg)
        return sampler(batch_size=video.size(0), device=device)

    raise ValueError(f"Unknown masking.strategy '{strategy}'")


__all__ = [
    "IndexedMaskSet",
    "build_loader",
    "build_scheduler",
    "build_student_from_cfg",
    "build_teacher_from_cfg",
    "mask_base_tokens_per_sample",
    "mask_batch_size",
    "mask_masked_tokens_per_sample",
    "mask_masked_tokens_total",
    "mask_num_views",
    "mask_ratio",
    "mask_total_tokens_total",
    "mask_visible_tokens_per_sample",
    "mask_visible_tokens_total",
    "resolve_batch_settings",
    "resolve_dataset_size",
    "resolve_device",
    "resolve_max_steps",
    "sample_mask_from_model",
    "unpack_video_batch",
]
