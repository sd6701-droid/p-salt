from __future__ import annotations

import argparse
import os
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.rethinking_jepa.utils import (
        build_loader,
        build_scheduler,
        build_student_from_cfg,
        build_teacher_from_cfg,
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
    batch_size = int(video.size(0))
    total_tokens_per_sample = int(mask.size(1))
    masked_tokens_per_sample = int(mask[0].sum().item())
    visible_tokens_per_sample = total_tokens_per_sample - masked_tokens_per_sample

    print(
        "student debug "
        f"pid={os.getpid()} step={step}/{max_steps} epoch={epoch} epoch_step={epoch_step} "
        f"dataset_size={dataset_size if dataset_size is not None else 'unknown'} "
        f"batch_size={batch_size} "
        f"teacher_tokens_per_sample={total_tokens_per_sample} "
        f"masked_tokens_per_sample={masked_tokens_per_sample} "
        f"student_visible_tokens_per_sample={visible_tokens_per_sample} "
        f"teacher_tokens_per_batch={total_tokens_per_sample * batch_size} "
        f"student_visible_tokens_per_batch={visible_tokens_per_sample * batch_size} "
        f"predictor_query_tokens_per_batch={masked_tokens_per_sample * batch_size}"
    )


def run(cfg: dict) -> None:
    device = resolve_device()
    teacher, _ = build_teacher_from_cfg(cfg, device)
    teacher.load_state_dict(torch.load(cfg["train"]["teacher_checkpoint"], map_location=device))

    student = build_student_from_cfg(cfg, teacher, device)
    loader = build_loader(cfg)
    device_batch_size, effective_batch_size = resolve_batch_settings(cfg)
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
                loss = criterion(out.prediction, out.target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lr, wd = scheduler.step(step)
            nn.utils.clip_grad_norm_(student.parameters(), cfg["optimizer"]["clip_grad"])
            optimizer.step()
            step += 1

            loss_value = float(loss.item())
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
                    "train/device_batch_size": device_batch_size,
                    "train/effective_batch_size": effective_batch_size,
                    "train/batch_size": int(video.size(0)),
                    "train/dataset_size": int(dataset_size) if dataset_size is not None else 0,
                    "train/tokens_total_per_sample": int(mask.size(1)),
                    "train/tokens_masked_per_sample": int(mask[0].sum().item()),
                    "train/tokens_visible_per_sample": int((~mask[0]).sum().item()),
                    "train/frozen_teacher_tokens_per_batch": int(mask.numel()),
                    "train/student_visible_tokens_per_batch": int((~mask).sum().item()),
                    "train/predictor_query_tokens_per_batch": int(mask.sum().item()),
                },
            )
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
