from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.rethinking_jepa.utils import (
        build_loader,
        resolve_batch_settings,
        resolve_training_horizon,
        build_scheduler,
        build_student_from_cfg,
        build_teacher_from_cfg,
        resolve_device,
        sample_mask_from_model,
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
        resolve_batch_settings,
        resolve_training_horizon,
        build_scheduler,
        build_student_from_cfg,
        build_teacher_from_cfg,
        resolve_device,
        sample_mask_from_model,
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
    print(
        f"{label} student checkpoint saved step={step} loss={loss:.6f} path={checkpoint_path}"
    )


def run(cfg: dict) -> None:
    device = resolve_device()
    teacher, _ = build_teacher_from_cfg(cfg, device)
    teacher.load_state_dict(torch.load(cfg["train"]["teacher_checkpoint"], map_location=device))

    student = build_student_from_cfg(cfg, teacher, device)
    loader = build_loader(cfg)
    device_batch_size, effective_batch_size, accumulation_steps = resolve_batch_settings(cfg)
    max_optimizer_steps, max_micro_steps = resolve_training_horizon(cfg, accumulation_steps)
    best_checkpoint_path, last_checkpoint_path = checkpoint_paths(cfg)
    wandb_run = init_wandb_run(cfg, job_type="student-train")
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = build_scheduler(cfg, optimizer, total_steps=max_optimizer_steps)
    criterion = nn.SmoothL1Loss()
    step = 0
    init_ckpt = cfg.get("student", {}).get("init_from_dinov2_checkpoint")

    student.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    running_loss = 0.0
    running_micro_batches = 0
    best_loss = float("inf")
    current_window_size = accumulation_steps

    while step < max_optimizer_steps:
        saw_batch = False
        for video in loader:
            saw_batch = True
            if running_micro_batches == 0:
                remaining_micro_steps = None
                if max_micro_steps is not None:
                    remaining_micro_steps = max_micro_steps - micro_step
                    if remaining_micro_steps <= 0:
                        break
                current_window_size = accumulation_steps
                if remaining_micro_steps is not None:
                    current_window_size = min(accumulation_steps, remaining_micro_steps)

            micro_step += 1
            video = video.to(device)
            mask = sample_mask_from_model(student.student.patch_embed, video, cfg, device)
            out = student(video, mask)
            loss = criterion(out.prediction, out.target)
            (loss / current_window_size).backward()
            running_loss += float(loss.item())
            running_micro_batches += 1

            if running_micro_batches < current_window_size:
                continue

            step += 1
            lr, wd = scheduler.step(step - 1)
            nn.utils.clip_grad_norm_(student.parameters(), cfg["optimizer"]["clip_grad"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = running_loss / max(1, running_micro_batches)
            _save_student_checkpoint(
                student,
                last_checkpoint_path,
                label="last",
                step=step,
                loss=mean_loss,
            )
            if mean_loss < best_loss:
                best_loss = mean_loss
                _save_student_checkpoint(
                    student,
                    best_checkpoint_path,
                    label="best",
                    step=step,
                    loss=mean_loss,
                )
            print(
                f"student step={step} micro_step={micro_step} loss={mean_loss:.6f} "
                f"lr={lr:.7f} wd={wd:.4f} effective_batch={effective_batch_size}"
            )
            log_wandb_metrics(
                wandb_run,
                {
                    "train/step": step,
                    "train/micro_step": micro_step,
                    "train/max_micro_steps": max_micro_steps if max_micro_steps is not None else 0,
                    "train/loss": mean_loss,
                    "train/lr": float(lr),
                    "train/weight_decay": float(wd),
                    "train/device_batch_size": device_batch_size,
                    "train/effective_batch_size": effective_batch_size,
                    "train/accumulation_steps": accumulation_steps,
                },
            )
            running_loss = 0.0
            running_micro_batches = 0
            if step >= max_optimizer_steps or (
                max_micro_steps is not None and micro_step >= max_micro_steps
            ):
                break

        if not saw_batch:
            raise RuntimeError("Training loader yielded no batches.")
        if max_micro_steps is not None and micro_step >= max_micro_steps:
            break

    finish_wandb_run(
        wandb_run,
        summary={
            "train/final_step": step,
            "train/final_micro_step": micro_step,
            "train/max_optimizer_steps": max_optimizer_steps,
            "train/max_micro_steps": max_micro_steps if max_micro_steps is not None else 0,
            "train/teacher_checkpoint": str(cfg["train"]["teacher_checkpoint"]),
            "train/device_batch_size": device_batch_size,
            "train/effective_batch_size": effective_batch_size,
            "train/accumulation_steps": accumulation_steps,
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
