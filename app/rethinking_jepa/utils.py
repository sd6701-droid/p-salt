from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, IterableDataset

from src.datasets.data_manager import build_video_dataset
from src.masks.default import sample_token_mask
from src.masks.multiblock3d import sample_multi_block_mask
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


def build_loader(cfg: dict) -> DataLoader:
    dataset = build_video_dataset(cfg)
    train_num_workers = int(cfg["train"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["device_batch_size"],
        shuffle=not isinstance(dataset, IterableDataset),
        num_workers=train_num_workers,
    )


def resolve_batch_settings(cfg: dict) -> tuple[int, int, int]:
    train_cfg = cfg["train"]
    device_batch_size = int(train_cfg["device_batch_size"])
    global_batch_size = int(train_cfg.get("global_batch_size", device_batch_size))

    if device_batch_size <= 0:
        raise ValueError(f"train.device_batch_size must be positive, got {device_batch_size}")
    if global_batch_size <= 0:
        raise ValueError(f"train.global_batch_size must be positive, got {global_batch_size}")
    if global_batch_size < device_batch_size:
        raise ValueError(
            "train.global_batch_size must be >= train.device_batch_size "
            f"(got {global_batch_size} < {device_batch_size})"
        )

    accumulation_steps = math.ceil(global_batch_size / device_batch_size)
    effective_batch_size = device_batch_size * accumulation_steps
    return device_batch_size, effective_batch_size, accumulation_steps


def resolve_training_horizon(cfg: dict, accumulation_steps: int) -> tuple[int, int | None]:
    train_cfg = cfg["train"]
    max_micro_steps_cfg = train_cfg.get("max_micro_steps")
    if max_micro_steps_cfg is not None:
        max_micro_steps = int(max_micro_steps_cfg)
        if max_micro_steps <= 0:
            raise ValueError(f"train.max_micro_steps must be positive, got {max_micro_steps}")
        return math.ceil(max_micro_steps / accumulation_steps), max_micro_steps

    max_steps = int(train_cfg["max_steps"])
    if max_steps <= 0:
        raise ValueError(f"train.max_steps must be positive, got {max_steps}")
    return max_steps, None


def build_teacher_from_cfg(cfg: dict, device: torch.device) -> tuple[TeacherModel, dict]:
    model_cfg = resolve_model_config(cfg["model"])
    teacher = TeacherModel(
        **model_cfg,
        frames=cfg["data"]["frames"],
        image_size=cfg["data"]["input_size"],
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
