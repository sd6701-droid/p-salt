from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.rethinking_jepa.utils import (
        build_loader,
        build_scheduler,
        build_student_from_cfg,
        build_teacher_from_cfg,
        mask_base_tokens_per_sample,
        mask_batch_size,
        mask_masked_tokens_per_sample,
        mask_masked_tokens_total,
        mask_num_views,
        mask_ratio,
        mask_total_tokens_total,
        mask_visible_tokens_per_sample,
        mask_visible_tokens_total,
        resolve_device,
        resolve_batch_settings,
        resolve_dataset_size,
        resolve_max_steps,
        sample_mask_from_model,
        unpack_video_batch,
    )
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
    from app.rethinking_jepa.utils import (
        build_loader,
        build_scheduler,
        build_student_from_cfg,
        build_teacher_from_cfg,
        mask_base_tokens_per_sample,
        mask_batch_size,
        mask_masked_tokens_per_sample,
        mask_masked_tokens_total,
        mask_num_views,
        mask_ratio,
        mask_total_tokens_total,
        mask_visible_tokens_per_sample,
        mask_visible_tokens_total,
        resolve_device,
        resolve_batch_settings,
        resolve_dataset_size,
        resolve_max_steps,
        sample_mask_from_model,
        unpack_video_batch,
    )
    from src.utils import (
        checkpoint_paths,
        finish_wandb_run,
        init_wandb_run,
        load_config,
        log_wandb_metrics,
        prepare_run_directory,
        redirect_run_logs,
    )


def _save_student_checkpoint(
    student: nn.Module,
    checkpoint_path: Path,
    *,
    label: str,
    step: int,
    loss: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), checkpoint_path)
    print(f"{label} student checkpoint saved step={step} loss={loss:.6f} path={checkpoint_path}")


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    precision = precision.lower()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _log_student_debug_batch(
    *,
    step: int,
    max_steps: int,
    epoch: int,
    epoch_step: int,
    video: torch.Tensor,
    mask: torch.Tensor,
    dataset_size: int | None,
) -> None:
    batch_size = mask_batch_size(mask)
    total_tokens_per_sample = mask_base_tokens_per_sample(mask)
    masked_tokens_per_sample = mask_masked_tokens_per_sample(mask)
    visible_tokens_per_sample = mask_visible_tokens_per_sample(mask)
    mask_ratio_value = mask_ratio(mask)
    mask_views = mask_num_views(mask)

    print(
        "student debug "
        f"pid={os.getpid()} step={step}/{max_steps} epoch={epoch} epoch_step={epoch_step} "
        f"dataset_size={dataset_size if dataset_size is not None else 'unknown'} "
        f"batch_size={batch_size} "
        f"teacher_tokens_per_sample={total_tokens_per_sample} "
        f"mask_views={mask_views} "
        f"masked_tokens_per_sample={masked_tokens_per_sample} "
        f"student_visible_tokens_per_sample={visible_tokens_per_sample} "
        f"mask_ratio={mask_ratio_value:.6f} "
        f"teacher_tokens_per_batch={total_tokens_per_sample * batch_size} "
        f"student_visible_tokens_per_batch={visible_tokens_per_sample * batch_size} "
        f"predictor_query_tokens_per_batch={masked_tokens_per_sample * batch_size}"
    )


def _student_prediction_diagnostics(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    target_mean = float(target.mean().item())
    target_std = float(target.std(unbiased=False).item())
    prediction_mean = float(prediction.mean().item())
    prediction_std = float(prediction.std(unbiased=False).item())
    cosine_similarity = float(F.cosine_similarity(prediction, target, dim=-1).mean().item())
    target_token_norm_mean = float(target.norm(dim=-1).mean().item())
    prediction_token_norm_mean = float(prediction.norm(dim=-1).mean().item())
    return {
        "target_mean": target_mean,
        "target_std": target_std,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "cosine_similarity": cosine_similarity,
        "target_token_norm_mean": target_token_norm_mean,
        "prediction_token_norm_mean": prediction_token_norm_mean,
    }


def _log_student_prediction_diagnostics(
    *,
    step: int,
    diagnostics: dict[str, float],
) -> None:
    print(
        "student prediction stats "
        f"step={step} "
        f"targets.mean={diagnostics['target_mean']:.6f} "
        f"targets.std={diagnostics['target_std']:.6f} "
        f"targets.token_norm_mean={diagnostics['target_token_norm_mean']:.6f} "
        f"predictions.mean={diagnostics['prediction_mean']:.6f} "
        f"predictions.std={diagnostics['prediction_std']:.6f} "
        f"predictions.token_norm_mean={diagnostics['prediction_token_norm_mean']:.6f} "
        f"predictions_vs_targets.cos_sim={diagnostics['cosine_similarity']:.6f}"
    )


def _mean_std_from_sums(sum_value: float, sumsq_value: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    mean = sum_value / count
    variance = max(0.0, (sumsq_value / count) - (mean * mean))
    return mean, math.sqrt(variance)


def run(cfg: dict) -> None:
    device = resolve_device()
    teacher, _ = build_teacher_from_cfg(cfg, device)
    teacher.load_state_dict(torch.load(cfg["train"]["teacher_checkpoint"], map_location=device))

    student = build_student_from_cfg(cfg, teacher, device)
    loader = build_loader(cfg)
    device_batch_size, accumulation_steps, effective_batch_size = resolve_batch_settings(cfg)
    max_steps = resolve_max_steps(cfg)
    best_checkpoint_path, last_checkpoint_path = checkpoint_paths(cfg)
    wandb_run = init_wandb_run(cfg, job_type="student-train")
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = build_scheduler(cfg, optimizer, total_steps=max_steps)
    criterion = nn.SmoothL1Loss()
    step = 0
    init_ckpt = cfg.get("student", {}).get("init_from_dinov2_checkpoint")
    checkpoint_interval = int(cfg["train"].get("checkpoint_interval", 100))
    log_interval = int(cfg["train"].get("log_interval", 10))
    precision = str(cfg["train"].get("precision", "fp32"))
    debug_steps = int(cfg["train"].get("debug_steps", 0))

    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = None
    dataset_size = resolve_dataset_size(loader)

    print(
        "student training start "
        f"device={device} device_batch_size={device_batch_size} "
        f"accumulation_steps={accumulation_steps} "
        f"effective_batch_size={effective_batch_size} "
        f"steps_per_epoch={steps_per_epoch if steps_per_epoch is not None else 'unknown'} "
        f"max_steps={max_steps} precision={precision} "
        f"debug_steps={debug_steps}"
    )
    if dataset_size is not None:
        print(f"student dataset_size={dataset_size}")

    student.train()
    best_loss = float("inf")
    epoch = 0
    loss_value = float("nan")
    optimizer.zero_grad(set_to_none=True)
    accumulation_count = 0
    accumulated_sample_count = 0
    accumulated_total_tokens = 0
    accumulated_masked_tokens = 0
    accumulated_visible_tokens = 0
    accumulated_loss_sum = 0.0
    accumulated_loss_weight = 0
    accumulated_target_sum = 0.0
    accumulated_target_sumsq = 0.0
    accumulated_target_numel = 0
    accumulated_prediction_sum = 0.0
    accumulated_prediction_sumsq = 0.0
    accumulated_prediction_numel = 0
    accumulated_target_token_norm_sum = 0.0
    accumulated_prediction_token_norm_sum = 0.0
    accumulated_cosine_similarity_sum = 0.0
    accumulated_token_count = 0

    while step < max_steps:
        epoch += 1
        saw_batch = False
        last_epoch_step = 0
        for epoch_step, batch in enumerate(loader, start=1):
            saw_batch = True
            last_epoch_step = epoch_step
            video, _ = unpack_video_batch(batch, device)
            mask = sample_mask_from_model(student.student.patch_embed, video, cfg, device)
            if step < debug_steps:
                _log_student_debug_batch(
                    step=step + 1,
                    max_steps=max_steps,
                    epoch=epoch,
                    epoch_step=epoch_step,
                    video=video,
                    mask=mask,
                    dataset_size=dataset_size,
                )
            with _autocast_context(device, precision):
                out = student(video, mask)
            prediction_for_loss = out.prediction.float()
            target_for_loss = out.target.float()
            loss = criterion(prediction_for_loss, target_for_loss)
            (loss / accumulation_steps).backward()

            batch_size = mask_batch_size(mask)
            total_tokens = mask_total_tokens_total(mask)
            masked_tokens = mask_masked_tokens_total(mask)
            visible_token_count = mask_visible_tokens_total(mask)
            token_count = int(prediction_for_loss.shape[0] * prediction_for_loss.shape[1])
            loss_weight = int(target_for_loss.numel())

            accumulation_count += 1
            accumulated_sample_count += batch_size
            accumulated_total_tokens += total_tokens
            accumulated_masked_tokens += masked_tokens
            accumulated_visible_tokens += visible_token_count
            accumulated_loss_sum += float(loss.item()) * loss_weight
            accumulated_loss_weight += loss_weight
            accumulated_target_sum += float(target_for_loss.sum().item())
            accumulated_target_sumsq += float(target_for_loss.square().sum().item())
            accumulated_target_numel += int(target_for_loss.numel())
            accumulated_prediction_sum += float(prediction_for_loss.sum().item())
            accumulated_prediction_sumsq += float(prediction_for_loss.square().sum().item())
            accumulated_prediction_numel += int(prediction_for_loss.numel())
            accumulated_target_token_norm_sum += float(target_for_loss.norm(dim=-1).sum().item())
            accumulated_prediction_token_norm_sum += float(prediction_for_loss.norm(dim=-1).sum().item())
            accumulated_cosine_similarity_sum += float(
                F.cosine_similarity(prediction_for_loss, target_for_loss, dim=-1).sum().item()
            )
            accumulated_token_count += token_count

            if accumulation_count < accumulation_steps:
                continue

            loss_value = accumulated_loss_sum / max(accumulated_loss_weight, 1)
            target_mean, target_std = _mean_std_from_sums(
                accumulated_target_sum,
                accumulated_target_sumsq,
                accumulated_target_numel,
            )
            prediction_mean, prediction_std = _mean_std_from_sums(
                accumulated_prediction_sum,
                accumulated_prediction_sumsq,
                accumulated_prediction_numel,
            )
            diagnostics = {
                "target_mean": target_mean,
                "target_std": target_std,
                "prediction_mean": prediction_mean,
                "prediction_std": prediction_std,
                "cosine_similarity": accumulated_cosine_similarity_sum / max(accumulated_token_count, 1),
                "target_token_norm_mean": accumulated_target_token_norm_sum / max(accumulated_token_count, 1),
                "prediction_token_norm_mean": accumulated_prediction_token_norm_sum / max(accumulated_token_count, 1),
            }
            mask_ratio_value = accumulated_masked_tokens / max(accumulated_total_tokens, 1)
            visible_tokens = accumulated_visible_tokens / max(accumulated_sample_count, 1)
            tokens_total_per_sample = mask_base_tokens_per_sample(mask)
            tokens_masked_per_sample = accumulated_masked_tokens / max(accumulated_sample_count, 1)
            tokens_visible_per_sample = accumulated_visible_tokens / max(accumulated_sample_count, 1)
            mask_views = mask_num_views(mask)

            lr, wd = scheduler.step(step)
            nn.utils.clip_grad_norm_(student.parameters(), cfg["optimizer"]["clip_grad"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            accumulation_count = 0

            should_checkpoint = step % checkpoint_interval == 0 or step >= max_steps
            if should_checkpoint:
                _save_student_checkpoint(
                    student,
                    last_checkpoint_path,
                    label="last",
                    step=step,
                    loss=loss_value,
                )
            if loss_value < best_loss:
                best_loss = loss_value
                if should_checkpoint:
                    _save_student_checkpoint(
                        student,
                        best_checkpoint_path,
                        label="best",
                        step=step,
                        loss=loss_value,
                    )

            should_log_prediction_diagnostics = step <= debug_steps or step % log_interval == 0 or step >= max_steps
            if should_log_prediction_diagnostics:
                _log_student_prediction_diagnostics(
                    step=step,
                    diagnostics=diagnostics,
                )

            if step % log_interval == 0 or step == 1 or step >= max_steps:
                if steps_per_epoch is not None:
                    print(
                        f"student step={step}/{max_steps} epoch={epoch} "
                        f"epoch_step={epoch_step}/{steps_per_epoch} loss={loss_value:.6f} "
                        f"lr={lr:.7f} wd={wd:.4f}"
                    )
                else:
                    print(
                        f"student step={step}/{max_steps} epoch={epoch} "
                        f"loss={loss_value:.6f} lr={lr:.7f} wd={wd:.4f}"
                    )

            log_wandb_metrics(
                wandb_run,
                {
                    "train/step": step,
                    "train/max_steps": max_steps,
                    "train/epoch": epoch,
                    "train/epoch_step": epoch_step,
                    "train/loss": loss_value,
                    "train/lr": float(lr),
                    "train/weight_decay": float(wd),
                    "train/accumulation_steps": accumulation_steps,
                    "train/device_batch_size": device_batch_size,
                    "train/effective_batch_size": effective_batch_size,
                    "train/mask_views": mask_views,
                    "train/batch_size": accumulated_sample_count,
                    "train/dataset_size": int(dataset_size) if dataset_size is not None else 0,
                    "train/tokens_total_per_sample": tokens_total_per_sample,
                    "train/tokens_masked_per_sample": tokens_masked_per_sample,
                    "train/tokens_visible_per_sample": tokens_visible_per_sample,
                    "train/mask_ratio": mask_ratio_value,
                    "train/mask_ratio_mean": mask_ratio_value,
                    "train/visible_tokens": visible_tokens,
                    "train/frozen_teacher_tokens_per_batch": accumulated_total_tokens,
                    "train/student_visible_tokens_per_batch": accumulated_visible_tokens,
                    "train/predictor_query_tokens_per_batch": accumulated_masked_tokens,
                    "train/targets_mean": diagnostics["target_mean"],
                    "train/targets_std": diagnostics["target_std"],
                    "train/targets_token_norm_mean": diagnostics["target_token_norm_mean"],
                    "train/predictions_mean": diagnostics["prediction_mean"],
                    "train/predictions_std": diagnostics["prediction_std"],
                    "train/predictions_token_norm_mean": diagnostics["prediction_token_norm_mean"],
                    "train/predictions_targets_cosine_similarity": diagnostics["cosine_similarity"],
                },
            )

            accumulated_sample_count = 0
            accumulated_total_tokens = 0
            accumulated_masked_tokens = 0
            accumulated_visible_tokens = 0
            accumulated_loss_sum = 0.0
            accumulated_loss_weight = 0
            accumulated_target_sum = 0.0
            accumulated_target_sumsq = 0.0
            accumulated_target_numel = 0
            accumulated_prediction_sum = 0.0
            accumulated_prediction_sumsq = 0.0
            accumulated_prediction_numel = 0
            accumulated_target_token_norm_sum = 0.0
            accumulated_prediction_token_norm_sum = 0.0
            accumulated_cosine_similarity_sum = 0.0
            accumulated_token_count = 0
            if step >= max_steps:
                break

        if not saw_batch:
            raise RuntimeError("Training loader yielded no batches.")
        if steps_per_epoch is not None and last_epoch_step == steps_per_epoch:
            print(f"student epoch={epoch} complete step={step}")
        else:
            print(f"student epoch={epoch} stopped step={step}")

    _save_student_checkpoint(
        student,
        last_checkpoint_path,
        label="last-final",
        step=step,
        loss=loss_value,
    )
    _save_student_checkpoint(
        student,
        best_checkpoint_path,
        label="best-final",
        step=step,
        loss=best_loss,
    )

    finish_wandb_run(
        wandb_run,
        summary={
            "train/final_step": step,
            "train/max_steps": max_steps,
            "train/final_epoch": epoch,
            "train/teacher_checkpoint": str(cfg["train"]["teacher_checkpoint"]),
            "train/accumulation_steps": accumulation_steps,
            "train/device_batch_size": device_batch_size,
            "train/effective_batch_size": effective_batch_size,
            "train/best_loss": best_loss,
            "train/run_id": str(cfg["runtime"]["run_id"]),
            "train/run_dir": str(cfg["runtime"]["run_dir"]),
            "train/best_checkpoint_path": str(best_checkpoint_path),
            "train/last_checkpoint_path": str(last_checkpoint_path),
            "train/student_init_checkpoint": str(init_ckpt) if init_ckpt else "",
            "model/teacher_architecture": str(cfg["model"]["architecture"]),
            "model/student_architecture": str(
                cfg.get("student_model", {}).get("architecture", cfg["model"]["architecture"])
            ),
        },
    )


def main(cfg: dict | None = None) -> None:
    config_path: str | None = None
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        config_path = args.config
        cfg = load_config(args.config)
    prepare_run_directory(cfg, config_path=config_path, app_name="rethinking_jepa.student")
    with redirect_run_logs(cfg):
        try:
            run(cfg)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
