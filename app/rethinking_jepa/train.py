from __future__ import annotations

import argparse
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
        build_teacher_from_cfg,
        resolve_device,
        resolve_batch_settings,
        resolve_max_steps,
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
        build_scheduler,
        build_teacher_from_cfg,
        resolve_device,
        resolve_batch_settings,
        resolve_max_steps,
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


def _save_checkpoint(model: nn.Module, checkpoint_path: Path, *, label: str, step: int, loss: float) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"{label} checkpoint saved step={step} loss={loss:.6f} path={checkpoint_path}")


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    precision = precision.lower()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run(cfg: dict) -> None:
    device = resolve_device()
    model, _ = build_teacher_from_cfg(cfg, device)
    loader = build_loader(cfg)
    device_batch_size, effective_batch_size = resolve_batch_settings(cfg)
    max_steps = resolve_max_steps(cfg)
    best_checkpoint_path, last_checkpoint_path = checkpoint_paths(cfg)
    wandb_run = init_wandb_run(cfg, job_type="teacher-train")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = build_scheduler(cfg, optimizer, total_steps=max_steps)
    criterion = nn.MSELoss()
    step = 0
    checkpoint_interval = int(cfg["train"].get("checkpoint_interval", 100))
    log_interval = int(cfg["train"].get("log_interval", 10))
    precision = str(cfg["train"].get("precision", "fp32"))

    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = None
    dataset_size = steps_per_epoch * device_batch_size if steps_per_epoch is not None else None
    target_epochs = math.ceil(max_steps / steps_per_epoch) if steps_per_epoch else None

    print(
        "teacher training start "
        f"device={device} device_batch_size={device_batch_size} "
        f"effective_batch_size={effective_batch_size} "
        f"steps_per_epoch={steps_per_epoch if steps_per_epoch is not None else 'unknown'} "
        f"max_steps={max_steps} "
        f"target_epochs={target_epochs if target_epochs is not None else 'unknown'} "
        f"precision={precision}"
    )
    if dataset_size is not None:
        print(f"teacher dataset_size={dataset_size}")

    model.train()
    best_loss = float("inf")
    epoch = 0

    while step < max_steps:
        epoch += 1
        saw_batch = False
        last_epoch_step = 0
        for epoch_step, video in enumerate(loader, start=1):
            saw_batch = True
            last_epoch_step = epoch_step
            video = video.to(device, non_blocking=(device.type == "cuda"))
            mask = sample_mask_from_model(model.encoder.patch_embed, video, cfg, device)
            with _autocast_context(device, precision):
                out = model(video, mask)
                loss = criterion(out.prediction, out.target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lr, wd = scheduler.step(step)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["clip_grad"])
            optimizer.step()
            step += 1

            loss_value = float(loss.item())
            should_checkpoint = step % checkpoint_interval == 0 or step >= max_steps
            if should_checkpoint:
                _save_checkpoint(model, last_checkpoint_path, label="last", step=step, loss=loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                if should_checkpoint:
                    _save_checkpoint(model, best_checkpoint_path, label="best", step=step, loss=loss_value)

            if step % log_interval == 0 or step == 1 or step >= max_steps:
                if steps_per_epoch is not None:
                    print(
                        f"teacher step={step}/{max_steps} epoch={epoch} "
                        f"epoch_step={epoch_step}/{steps_per_epoch} loss={loss_value:.6f} "
                        f"lr={lr:.7f} wd={wd:.4f}"
                    )
                else:
                    print(
                        f"teacher step={step}/{max_steps} epoch={epoch} "
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
                },
            )
            if step >= max_steps:
                break

        if not saw_batch:
            raise RuntimeError("Training loader yielded no batches.")
        if steps_per_epoch is not None and last_epoch_step == steps_per_epoch:
            print(f"teacher epoch={epoch} complete step={step}")
        else:
            print(f"teacher epoch={epoch} stopped step={step}")

    _save_checkpoint(model, last_checkpoint_path, label="last-final", step=step, loss=loss_value)
    _save_checkpoint(model, best_checkpoint_path, label="best-final", step=step, loss=best_loss)

    finish_wandb_run(
        wandb_run,
        summary={
            "train/final_step": step,
            "train/max_steps": max_steps,
            "train/final_epoch": epoch,
            "train/device_batch_size": device_batch_size,
            "train/effective_batch_size": effective_batch_size,
            "train/best_loss": best_loss,
            "train/run_id": str(cfg["runtime"]["run_id"]),
            "train/run_dir": str(cfg["runtime"]["run_dir"]),
            "train/best_checkpoint_path": str(best_checkpoint_path),
            "train/last_checkpoint_path": str(last_checkpoint_path),
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
    prepare_run_directory(cfg, config_path=config_path, app_name="rethinking_jepa.train")
    with redirect_run_logs(cfg):
        try:
            run(cfg)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
