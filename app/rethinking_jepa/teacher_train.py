from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

if "SLURM_LOCALID" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.rethinking_jepa.utils import build_scheduler, build_teacher_from_cfg
    from src.datasets.video_data_manager import init_data
    from src.datasets.video_dataset import VideoAugmentationConfig, random_resized_crop_video
    from src.masks.types import IndexedMaskSet
    from src.masks.vjepa_style_masking import MaskCollator, RandomTokenMaskCollator
    from src.utils import (
        checkpoint_paths,
        finish_wandb_run,
        init_wandb_run,
        load_config,
        log_wandb_metrics,
        prepare_run_directory,
        redirect_run_logs,
    )
else:
    from app.rethinking_jepa.utils import build_scheduler, build_teacher_from_cfg
    from src.datasets.video_data_manager import init_data
    from src.datasets.video_dataset import VideoAugmentationConfig, random_resized_crop_video
    from src.masks.types import IndexedMaskSet
    from src.masks.vjepa_style_masking import MaskCollator, RandomTokenMaskCollator
    from src.utils import (
        checkpoint_paths,
        finish_wandb_run,
        init_wandb_run,
        load_config,
        log_wandb_metrics,
        prepare_run_directory,
        redirect_run_logs,
    )


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def log(self, values: dict[str, Any]) -> None:
        row = {key: values.get(key, "") for key in self.fieldnames}
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_multiprocessing_start_method() -> None:
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass


def init_distributed() -> tuple[torch.device, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    if torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES") and "," not in os.environ["CUDA_VISIBLE_DEVICES"]:
            device_index = 0
        else:
            device_index = local_rank % max(1, torch.cuda.device_count())
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device, rank, world_size, local_rank, rank == 0


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    precision = precision.lower()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return getattr(model, "_orig_mod", model)


def build_mask_cfgs(mask_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "configs" in mask_cfg:
        return list(mask_cfg["configs"])
    return [
        {
            "aspect_ratio": list(mask_cfg["mask_aspect_ratio"]),
            "full_complement": False,
            "max_keep": None,
            "max_temporal_keep": float(mask_cfg.get("max_temporal_keep", 1.0)),
            "num_blocks": int(mask_cfg.get("short_num_blocks", 8)),
            "spatial_scale": [float(mask_cfg["short_spatial_mask_scale"])] * 2,
            "temporal_scale": [float(mask_cfg["temporal_mask_scale"])] * 2,
        },
        {
            "aspect_ratio": list(mask_cfg["mask_aspect_ratio"]),
            "full_complement": False,
            "max_keep": None,
            "max_temporal_keep": float(mask_cfg.get("max_temporal_keep", 1.0)),
            "num_blocks": int(mask_cfg.get("long_num_blocks", 2)),
            "spatial_scale": [float(mask_cfg["long_spatial_mask_scale"])] * 2,
            "temporal_scale": [float(mask_cfg["temporal_mask_scale"])] * 2,
        },
    ]


def mask_strategy_name(mask_cfg: dict[str, Any]) -> str:
    if "strategy" in mask_cfg:
        return str(mask_cfg["strategy"])
    if "configs" in mask_cfg:
        return "custom_vjepa_configs"
    return "vjepa_multiblock3d"


def describe_mask_cfgs(mask_cfgs: list[dict[str, Any]]) -> str:
    parts = []
    for view_idx, cfg in enumerate(mask_cfgs):
        if "mask_ratio" in cfg and "num_blocks" not in cfg:
            parts.append(
                f"view={view_idx} strategy={cfg.get('strategy')} "
                f"mask_ratio={cfg.get('mask_ratio')}"
            )
        else:
            parts.append(
                f"view={view_idx} "
                f"blocks={cfg.get('num_blocks')} "
                f"spatial_scale={cfg.get('spatial_scale')} "
                f"temporal_scale={cfg.get('temporal_scale')} "
                f"aspect_ratio={cfg.get('aspect_ratio')}"
            )
    return "; ".join(parts)


def mask_view_names(mask_cfgs: list[dict[str, Any]]) -> list[str]:
    if len(mask_cfgs) == 2:
        return ["short", "long"]
    if len(mask_cfgs) == 1:
        strategy = str(mask_cfgs[0].get("strategy", "")).lower()
        if strategy in {"random", "random_token", "token"}:
            return ["random"]
        return ["view_0"]
    return [f"view_{idx}" for idx in range(len(mask_cfgs))]


class VideoTrainTransform:
    def __init__(
        self,
        *,
        input_size: int,
        channels: int,
        augmentation: VideoAugmentationConfig | None,
    ) -> None:
        self.input_size = input_size
        self.channels = channels
        self.augmentation = augmentation

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 4:
            raise ValueError(f"Expected clip with 4 dims, got shape={tuple(clip.shape)}")
        if clip.size(-1) in {1, 3, self.channels}:
            clip = clip.permute(3, 0, 1, 2).contiguous()
        if clip.size(0) == 1 and self.channels == 3:
            clip = clip.expand(3, -1, -1, -1)
        if clip.size(0) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {clip.size(0)}")
        if self.augmentation is not None:
            return random_resized_crop_video(clip, self.augmentation)
        resized = F.interpolate(
            clip.permute(1, 0, 2, 3),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(1, 0, 2, 3).contiguous()


def build_video_transform(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    aug_cfg = cfg.get("data_aug", cfg.get("augmentation", {}))
    input_size = int(data_cfg["input_size"])
    channels = int(model_cfg.get("in_channels", 3))
    augmentation = None
    if aug_cfg:
        augmentation = VideoAugmentationConfig(
            input_size=input_size,
            random_resize_aspect_ratio=tuple(aug_cfg["random_resize_aspect_ratio"]),
            random_resize_scale=tuple(aug_cfg["random_resize_scale"]),
        )
    return VideoTrainTransform(
        input_size=input_size,
        channels=channels,
        augmentation=augmentation,
    )


def maybe_compile_model(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model)
    return model


def move_masks(masks: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    non_blocking = device.type == "cuda"
    return [mask.to(device=device, dtype=torch.long, non_blocking=non_blocking) for mask in masks]


def video_from_collated_batch(collated_batch: Any) -> torch.Tensor:
    clips = collated_batch[0]
    if isinstance(clips, (list, tuple)):
        video = clips[0]
    else:
        video = clips
    if isinstance(video, (list, tuple)):
        video = video[0]
    if not torch.is_tensor(video):
        raise TypeError(f"Expected collated video tensor, got {type(video).__name__}")
    return video


def iter_collations(batch: Any):
    if isinstance(batch, list):
        yield from batch
    else:
        yield batch


def _tensor_preview(tensor: torch.Tensor, max_items: int = 8) -> list[Any]:
    preview = tensor.detach().cpu().reshape(-1)[:max_items].tolist()
    return [int(value) if isinstance(value, (int, bool)) else value for value in preview]


def _mask_preview(mask: torch.Tensor, max_items: int = 8) -> list[Any]:
    sample = mask[0] if mask.ndim > 1 else mask
    return _tensor_preview(sample, max_items=max_items)


def format_mask_collator_output(
    collated_batch: Any,
    masks_enc: list[torch.Tensor],
    masks_pred: list[torch.Tensor],
) -> str:
    video = video_from_collated_batch(collated_batch)
    lines = [
        "mask_collator output",
        f"  collated_batch_type={type(collated_batch).__name__}",
        f"  video_shape={tuple(video.shape)} video_dtype={video.dtype}",
        f"  mask_views={len(masks_pred)}",
    ]
    if isinstance(collated_batch, (list, tuple)) and len(collated_batch) > 1:
        labels = collated_batch[1]
        if torch.is_tensor(labels):
            lines.append(f"  labels_shape={tuple(labels.shape)} labels_dtype={labels.dtype}")
        else:
            lines.append(f"  labels_type={type(labels).__name__}")

    for view_idx in range(max(len(masks_enc), len(masks_pred))):
        enc = masks_enc[view_idx] if view_idx < len(masks_enc) else None
        pred = masks_pred[view_idx] if view_idx < len(masks_pred) else None
        enc_desc = "missing" if enc is None else f"shape={tuple(enc.shape)} preview={_mask_preview(enc)}"
        pred_desc = "missing" if pred is None else f"shape={tuple(pred.shape)} preview={_mask_preview(pred)}"
        lines.append(f"  view_{view_idx} encoder_visible={enc_desc}")
        lines.append(f"  view_{view_idx} predictor_masked={pred_desc}")
    return "\n".join(lines)


def build_indexed_mask(
    *,
    masks_enc: list[torch.Tensor],
    masks_pred: list[torch.Tensor],
    num_tokens: int,
) -> IndexedMaskSet:
    return IndexedMaskSet(encoder_ids=masks_enc, predictor_ids=masks_pred, num_tokens=num_tokens)


def mask_stats(video: torch.Tensor, masks_enc: list[torch.Tensor], masks_pred: list[torch.Tensor], tokens: int) -> dict[str, Any]:
    batch_size = int(video.size(0))
    visible_by_view = [int(mask.size(1)) for mask in masks_enc]
    prediction_by_view = [int(mask.size(1)) for mask in masks_pred]
    mask_ratio_by_view = [prediction_tokens / max(tokens, 1) for prediction_tokens in prediction_by_view]
    prediction_tokens_per_sample = int(sum(prediction_by_view))
    visible_tokens_per_sample = int(sum(visible_by_view))
    mask_views = len(masks_pred)
    total_tokens = int(tokens * batch_size * mask_views)
    prediction_tokens = int(prediction_tokens_per_sample * batch_size)
    visible_tokens = int(visible_tokens_per_sample * batch_size)
    return {
        "batch_size": batch_size,
        "mask_views": mask_views,
        "tokens_total_per_sample": tokens,
        "tokens_prediction_per_sample": prediction_tokens_per_sample,
        "tokens_prediction_by_view": prediction_by_view,
        "tokens_visible_per_sample": visible_tokens_per_sample,
        "tokens_visible_by_view": visible_by_view,
        "mask_ratio_by_view": mask_ratio_by_view,
        "mask_ratio": prediction_tokens / max(total_tokens, 1),
        "prediction_tokens_per_batch": prediction_tokens,
        "visible_tokens_per_batch": visible_tokens,
        "total_tokens_per_batch": total_tokens,
    }


def mean_std_from_sums(sum_value: float, sumsq_value: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    mean = sum_value / count
    variance = max(0.0, (sumsq_value / count) - mean * mean)
    return mean, math.sqrt(variance)


def save_training_state(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    epoch: int,
    loss: float,
    best_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "best_loss": best_loss,
        },
        path,
    )


def load_training_state(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler.is_enabled() and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_loss", float("inf")))


def save_teacher_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run(cfg: dict[str, Any], *, resume_preempt: bool = False) -> None:
    meta_cfg = cfg.get("meta", {})
    train_cfg = cfg.get("train", {})
    optimizer_cfg = cfg.get("optimization", cfg.get("optimizer", {}))
    mask_cfg = cfg.get("mask", cfg.get("masking", {}))
    data_cfg = cfg["data"]

    precision = str(train_cfg.get("precision", meta_cfg.get("precision", "fp32"))).lower()
    seed = int(meta_cfg.get("seed", train_cfg.get("seed", 0)))
    set_random_seed(seed)
    set_multiprocessing_start_method()
    device, rank, world_size, local_rank, is_main = init_distributed()

    model, _ = build_teacher_from_cfg(cfg, device)
    #not sure if need ?
    model = maybe_compile_model(model, bool(train_cfg.get("compile", False)))

    if world_size > 1:
        device_ids = [device.index] if device.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    tokens_per_sample = int(
        (int(data_cfg["frames"]) // int(cfg["model"]["tubelet_size"]))
        * (int(data_cfg["input_size"]) // int(cfg["model"]["patch_size"]))
        * (int(data_cfg["input_size"]) // int(cfg["model"]["patch_size"]))
    )
    strategy = str(mask_cfg.get("strategy", "multiseq_multiblock3d")).lower()
    random_strategies = {"random", "random_token", "token"}

    if strategy in random_strategies:
        mask_ratio = float(mask_cfg.get("mask_ratio", mask_cfg.get("random_mask_ratio", 0.75)))
        mask_cfgs = [{"strategy": "random", "mask_ratio": mask_ratio}]
        mask_collator = RandomTokenMaskCollator(
            num_tokens=tokens_per_sample,
            mask_ratio=mask_ratio,
            dataset_fpcs=list(data_cfg.get("dataset_fpcs", [int(data_cfg["frames"])])),
        )
    else:
        mask_cfgs = build_mask_cfgs(mask_cfg)
        mask_collator = MaskCollator(
            cfgs_mask=mask_cfgs,
            dataset_fpcs=list(data_cfg.get("dataset_fpcs", [int(data_cfg["frames"])])),
            crop_size=(int(data_cfg["input_size"]), int(data_cfg["input_size"])),
            patch_size=(int(cfg["model"]["patch_size"]), int(cfg["model"]["patch_size"])),
            tubelet_size=int(cfg["model"]["tubelet_size"]),
        )

    print('mask_collator--->', mask_collator)
    data_loader, dist_sampler = init_data(
        batch_size=int(train_cfg["device_batch_size"]),
        transform=build_video_transform(cfg),
        data="videodataset",
        collator=mask_collator,
        pin_mem=device.type == "cuda",
        num_workers=int(train_cfg.get("num_workers", 0)),
        world_size=world_size,
        rank=rank,
        root_path=data_cfg.get("data_paths") or data_cfg.get("manifest") or data_cfg.get("root"),
        drop_last=bool(train_cfg.get("drop_last", True)),
        clip_len=int(data_cfg["frames"]),
        dataset_fpcs=data_cfg.get("dataset_fpcs"),
        frame_sample_rate=int(data_cfg.get("frame_step", 1)),
        duration=data_cfg.get("duration"),
        fps=data_cfg.get("fps"),
        num_clips=int(data_cfg.get("num_clips", 1)),
        random_clip_sampling=bool(data_cfg.get("random_clip_sampling", True)),
        allow_clip_overlap=bool(data_cfg.get("allow_clip_overlap", False)),
        filter_short_videos=bool(data_cfg.get("filter_short_videos", False)),
        filter_long_videos=int(data_cfg.get("filter_long_videos", int(1e9))),
        datasets_weights=data_cfg.get("datasets_weights"),
        persistent_workers=bool(train_cfg.get("persistent_workers", train_cfg.get("num_workers", 0) > 0)),
        deterministic=bool(train_cfg.get("deterministic", True)),
        log_dir=cfg.get("runtime", {}).get("run_dir"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["start_lr"]),
        betas=tuple(optimizer_cfg["betas"]),
        weight_decay=float(optimizer_cfg["start_weight_decay"]),
    )
    max_steps = int(train_cfg["max_steps"])
    scheduler_cfg = dict(cfg)
    scheduler_cfg["optimizer"] = optimizer_cfg
    scheduler = build_scheduler(scheduler_cfg, optimizer, total_steps=max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))
    criterion = nn.MSELoss()

    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 100))
    log_interval = int(train_cfg.get("log_interval", 10))
    clip_grad = float(optimizer_cfg.get("clip_grad", 1.0))
    best_checkpoint_path, last_checkpoint_path = checkpoint_paths(cfg)
    run_dir = Path(cfg["runtime"]["run_dir"])
    latest_path = run_dir / "latest.pt"
    csv_logger = CsvLogger(
        run_dir / "train.csv",
        [
            "step",
            "epoch",
            "loss",
            "lr",
            "weight_decay",
            "mask_strategy",
            "mask_ratio",
            "mask_short_ratio",
            "mask_long_ratio",
            "mask_views",
            "tokens_total_per_sample",
            "tokens_visible_per_sample",
            "tokens_prediction_per_sample",
            "mask_short_masked_tokens_per_sample",
            "mask_long_masked_tokens_per_sample",
            "mask_short_visible_tokens_per_sample",
            "mask_long_visible_tokens_per_sample",
            "tokens_visible_by_view",
            "tokens_prediction_by_view",
            "visible_tokens_per_batch",
            "prediction_tokens_per_batch",
            "total_tokens_per_batch",
            "predictions_mean",
            "predictions_std",
            "targets_mean",
            "targets_std",
        ],
    )
    wandb_run = init_wandb_run(cfg, job_type="teacher-train") if is_main else None

    step = 0
    epoch = 0
    best_loss = float("inf")
    resume_path = train_cfg.get("resume_from")
    if resume_preempt and resume_path is None:
        resume_path = latest_path
    if resume_path and Path(resume_path).expanduser().exists():
        step, epoch, best_loss = load_training_state(
            path=Path(resume_path).expanduser(),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        for _ in range(step):
            mask_collator.step()
        scheduler.step(step)
        if is_main:
            print(f"resumed teacher_train from {resume_path} step={step} epoch={epoch}")

    if is_main:
        dataset_size = len(getattr(data_loader, "dataset", []))
        effective_batch_size = int(train_cfg["device_batch_size"]) * accumulation_steps * world_size
        print(
            "teacher_train mask setup "
            f"strategy={mask_strategy_name(mask_cfg)} "
            f"tokens_total_per_sample={tokens_per_sample} "
            f"view_names={mask_view_names(mask_cfgs)} "
            f"configs={describe_mask_cfgs(mask_cfgs)}"
        )
        print(
            "teacher_train start "
            f"device={device} rank={rank}/{world_size} local_rank={local_rank} "
            f"dataset_size={dataset_size} device_batch_size={train_cfg['device_batch_size']} "
            f"accumulation_steps={accumulation_steps} effective_batch_size={effective_batch_size} "
            f"precision={precision} max_steps={max_steps} model=encoder+decoder"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulated_weight = 0
    accumulation_count = 0
    accumulated_target_sum = 0.0
    accumulated_target_sumsq = 0.0
    accumulated_target_numel = 0
    accumulated_prediction_sum = 0.0
    accumulated_prediction_sumsq = 0.0
    accumulated_prediction_numel = 0
    last_loss = float("nan")
    printed_mask_collator_output = False

    while step < max_steps:
        epoch += 1
        if hasattr(dist_sampler, "set_epoch"):
            dist_sampler.set_epoch(epoch)

        for batch in data_loader:
            if step >= max_steps:
                break
            for collated_batch, masks_enc_cpu, masks_pred_cpu in iter_collations(batch):
                if is_main and not printed_mask_collator_output:
                    print(
                        format_mask_collator_output(collated_batch, masks_enc_cpu, masks_pred_cpu),
                        flush=True,
                    )
                    printed_mask_collator_output = True
                video = video_from_collated_batch(collated_batch).to(device, non_blocking=(device.type == "cuda"))
                masks_enc = move_masks(masks_enc_cpu, device)
                masks_pred = move_masks(masks_pred_cpu, device)
                mask = build_indexed_mask(masks_enc=masks_enc, masks_pred=masks_pred, num_tokens=tokens_per_sample)

                if accumulation_count == 0:
                    lr, wd = scheduler.step(step)

                with autocast_context(device, precision):
                    out = model(video, mask)
                    prediction = out.prediction.float()
                    target = out.target.float()
                    loss = criterion(prediction, target)

                scaled_loss = loss / accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                loss_weight = int(target.numel())
                accumulated_loss += float(loss.detach().item()) * loss_weight
                accumulated_weight += loss_weight
                accumulation_count += 1
                accumulated_target_sum += float(target.sum().item())
                accumulated_target_sumsq += float(target.square().sum().item())
                accumulated_target_numel += int(target.numel())
                accumulated_prediction_sum += float(prediction.sum().item())
                accumulated_prediction_sumsq += float(prediction.square().sum().item())
                accumulated_prediction_numel += int(prediction.numel())

                if accumulation_count < accumulation_steps:
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                last_loss = accumulated_loss / max(accumulated_weight, 1)
                target_mean, target_std = mean_std_from_sums(
                    accumulated_target_sum,
                    accumulated_target_sumsq,
                    accumulated_target_numel,
                )
                prediction_mean, prediction_std = mean_std_from_sums(
                    accumulated_prediction_sum,
                    accumulated_prediction_sumsq,
                    accumulated_prediction_numel,
                )
                stats = mask_stats(video, masks_enc, masks_pred, tokens=tokens_per_sample)
                strategy_name = mask_strategy_name(mask_cfg)

                if is_main and (step == 1 or step % log_interval == 0 or step >= max_steps):
                    view_names = mask_view_names(mask_cfgs)
                    mask_view_metrics = {}
                    named_mask_metrics = {}
                    for view_idx, visible_tokens in enumerate(stats["tokens_visible_by_view"]):
                        view_name = view_names[view_idx] if view_idx < len(view_names) else f"view_{view_idx}"
                        mask_view_metrics[f"train/mask_view_{view_idx}_visible_tokens_per_sample"] = visible_tokens
                        named_mask_metrics[f"train/mask_{view_name}_visible_tokens_per_sample"] = visible_tokens
                    for view_idx, prediction_tokens in enumerate(stats["tokens_prediction_by_view"]):
                        view_name = view_names[view_idx] if view_idx < len(view_names) else f"view_{view_idx}"
                        mask_ratio = float(stats["mask_ratio_by_view"][view_idx])
                        mask_view_metrics[f"train/mask_view_{view_idx}_prediction_tokens_per_sample"] = prediction_tokens
                        mask_view_metrics[f"train/mask_view_{view_idx}_ratio"] = mask_ratio
                        named_mask_metrics[f"train/mask_{view_name}_masked_tokens_per_sample"] = prediction_tokens
                        named_mask_metrics[f"train/mask_{view_name}_prediction_tokens_per_sample"] = prediction_tokens
                        named_mask_metrics[f"train/mask_{view_name}_ratio"] = mask_ratio
                    metrics = {
                        "step": step,
                        "epoch": epoch,
                        "loss": last_loss,
                        "lr": float(lr),
                        "weight_decay": float(wd),
                        "mask_strategy": strategy_name,
                        "mask_ratio": float(stats["mask_ratio"]),
                        "mask_short_ratio": named_mask_metrics.get("train/mask_short_ratio", ""),
                        "mask_long_ratio": named_mask_metrics.get("train/mask_long_ratio", ""),
                        "mask_views": int(stats["mask_views"]),
                        "tokens_total_per_sample": int(stats["tokens_total_per_sample"]),
                        "tokens_visible_per_sample": int(stats["tokens_visible_per_sample"]),
                        "tokens_prediction_per_sample": int(stats["tokens_prediction_per_sample"]),
                        "mask_short_masked_tokens_per_sample": named_mask_metrics.get(
                            "train/mask_short_masked_tokens_per_sample", ""
                        ),
                        "mask_long_masked_tokens_per_sample": named_mask_metrics.get(
                            "train/mask_long_masked_tokens_per_sample", ""
                        ),
                        "mask_short_visible_tokens_per_sample": named_mask_metrics.get(
                            "train/mask_short_visible_tokens_per_sample", ""
                        ),
                        "mask_long_visible_tokens_per_sample": named_mask_metrics.get(
                            "train/mask_long_visible_tokens_per_sample", ""
                        ),
                        "tokens_visible_by_view": str(stats["tokens_visible_by_view"]),
                        "tokens_prediction_by_view": str(stats["tokens_prediction_by_view"]),
                        "visible_tokens_per_batch": int(stats["visible_tokens_per_batch"]),
                        "prediction_tokens_per_batch": int(stats["prediction_tokens_per_batch"]),
                        "total_tokens_per_batch": int(stats["total_tokens_per_batch"]),
                        "predictions_mean": prediction_mean,
                        "predictions_std": prediction_std,
                        "targets_mean": target_mean,
                        "targets_std": target_std,
                    }
                    print(
                        "teacher_train "
                        f"step={step}/{max_steps} epoch={epoch} loss={last_loss:.6f} "
                        f"lr={lr:.7f} wd={wd:.4f} "
                        f"mask_strategy={strategy_name} "
                        f"mask_ratio={stats['mask_ratio']:.6f} "
                        f"short_ratio={named_mask_metrics.get('train/mask_short_ratio', 'n/a')} "
                        f"long_ratio={named_mask_metrics.get('train/mask_long_ratio', 'n/a')} "
                        f"visible_tokens_per_sample={stats['tokens_visible_per_sample']} "
                        f"prediction_tokens_per_sample={stats['tokens_prediction_per_sample']} "
                        f"short_masked={named_mask_metrics.get('train/mask_short_masked_tokens_per_sample', 'n/a')} "
                        f"long_masked={named_mask_metrics.get('train/mask_long_masked_tokens_per_sample', 'n/a')} "
                        f"visible_by_view={stats['tokens_visible_by_view']} "
                        f"prediction_by_view={stats['tokens_prediction_by_view']}"
                    )
                    csv_logger.log(metrics)
                    log_wandb_metrics(
                        wandb_run,
                        {
                            "train/step": step,
                            "train/epoch": epoch,
                            "train/loss": last_loss,
                            "train/lr": float(lr),
                            "train/weight_decay": float(wd),
                            "train/masking_strategy": strategy_name,
                            "train/mask_ratio": float(stats["mask_ratio"]),
                            "train/mask_views": int(stats["mask_views"]),
                            "train/tokens_total_per_sample": int(stats["tokens_total_per_sample"]),
                            "train/tokens_masked_per_sample": int(stats["tokens_prediction_per_sample"]),
                            "train/tokens_prediction_per_sample": int(stats["tokens_prediction_per_sample"]),
                            "train/tokens_visible_per_sample": int(stats["tokens_visible_per_sample"]),
                            "train/visible_tokens_per_batch": int(stats["visible_tokens_per_batch"]),
                            "train/prediction_tokens_per_batch": int(stats["prediction_tokens_per_batch"]),
                            "train/total_tokens_per_batch": int(stats["total_tokens_per_batch"]),
                            "train/predictions_mean": prediction_mean,
                            "train/predictions_std": prediction_std,
                            "train/targets_mean": target_mean,
                            "train/targets_std": target_std,
                            **mask_view_metrics,
                            **named_mask_metrics,
                        },
                    )

                if is_main and (step % checkpoint_interval == 0 or step >= max_steps):
                    save_training_state(
                        path=latest_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        epoch=epoch,
                        loss=last_loss,
                        best_loss=best_loss,
                    )
                    save_teacher_checkpoint(model, last_checkpoint_path)
                    print(f"teacher_train checkpoint saved step={step} path={latest_path}")

                if is_main and last_loss < best_loss:
                    best_loss = last_loss
                    save_teacher_checkpoint(model, best_checkpoint_path)

                accumulated_loss = 0.0
                accumulated_weight = 0
                accumulation_count = 0
                accumulated_target_sum = 0.0
                accumulated_target_sumsq = 0.0
                accumulated_target_numel = 0
                accumulated_prediction_sum = 0.0
                accumulated_prediction_sumsq = 0.0
                accumulated_prediction_numel = 0

    if is_main:
        save_training_state(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            epoch=epoch,
            loss=last_loss,
            best_loss=best_loss,
        )
        save_teacher_checkpoint(model, last_checkpoint_path)
        finish_wandb_run(
            wandb_run,
            summary={
                "train/final_step": step,
                "train/final_epoch": epoch,
                "train/best_loss": best_loss,
                "train/latest_checkpoint_path": str(latest_path),
                "train/best_checkpoint_path": str(best_checkpoint_path),
                "train/last_checkpoint_path": str(last_checkpoint_path),
            },
        )
    cleanup_distributed()


def main(cfg: dict | None = None, *, resume_preempt: bool = False) -> None:
    config_path: str | None = None
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--resume-preempt", action="store_true")
        args = parser.parse_args()
        config_path = args.config
        resume_preempt = bool(args.resume_preempt)
        cfg = load_config(args.config)
    prepare_run_directory(cfg, config_path=config_path, app_name="rethinking_jepa.teacher_train")
    with redirect_run_logs(cfg):
        try:
            run(cfg, resume_preempt=resume_preempt)
        except Exception:
            traceback.print_exc()
            cleanup_distributed()
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
