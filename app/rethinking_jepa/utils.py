from __future__ import annotations

import torch
from torch.utils.data import DataLoader, IterableDataset

from src.datasets.data_manager import build_video_dataset
from src.masks.default import sample_token_mask
from src.masks.multiblock3d import sample_multi_block_mask
from src.models.architectures import resolve_model_config
from src.models.jepa import StudentModel, TeacherModel
from src.utils.schedulers import CosineScheduler


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(cfg: dict) -> DataLoader:
    dataset = build_video_dataset(cfg)
    train_num_workers = int(cfg["train"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["device_batch_size"],
        shuffle=not isinstance(dataset, IterableDataset),
        num_workers=train_num_workers,
    )


def build_teacher_from_cfg(cfg: dict, device: torch.device) -> tuple[TeacherModel, dict]:
    model_cfg = resolve_model_config(cfg["model"])
    teacher = TeacherModel(
        **model_cfg,
        frames=cfg["data"]["frames"],
        image_size=cfg["data"]["input_size"],
    ).to(device)
    return teacher, model_cfg


def build_student_from_cfg(cfg: dict, teacher: TeacherModel, device: torch.device) -> StudentModel:
    student = StudentModel(
        teacher=teacher,
        predictor_dim=cfg["student"]["predictor_dim"],
        predictor_depth=cfg["student"]["predictor_depth"],
        predictor_heads=cfg["student"]["predictor_heads"],
    ).to(device)
    return student


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer) -> CosineScheduler:
    return CosineScheduler(
        optimizer=optimizer,
        total_steps=cfg["train"]["max_steps"],
        warmup_steps=cfg["optimizer"]["warmup_steps"],
        start_lr=cfg["optimizer"]["start_lr"],
        peak_lr=cfg["optimizer"]["lr"],
        final_lr=cfg["optimizer"]["final_lr"],
        start_weight_decay=cfg["optimizer"]["start_weight_decay"],
        end_weight_decay=cfg["optimizer"]["end_weight_decay"],
    )


def sample_mask_from_model(
    patch_embed,
    video: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
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
            device=device,
        )

    raise ValueError(f"Unknown masking.strategy '{strategy}'")
