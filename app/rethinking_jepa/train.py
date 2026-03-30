from __future__ import annotations

import argparse
import sys
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
        sample_mask_from_model,
    )
    from src.utils import finish_wandb_run, init_wandb_run, load_config, log_wandb_metrics
else:
    from app.rethinking_jepa.utils import (
        build_loader,
        build_scheduler,
        build_teacher_from_cfg,
        resolve_device,
        sample_mask_from_model,
    )
    from src.utils import finish_wandb_run, init_wandb_run, load_config, log_wandb_metrics


def run(cfg: dict) -> None:
    device = resolve_device()
    model, _ = build_teacher_from_cfg(cfg, device)
    loader = build_loader(cfg)
    wandb_run = init_wandb_run(cfg, job_type="teacher-train")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = build_scheduler(cfg, optimizer)
    criterion = nn.MSELoss()
    step = 0

    model.train()
    for step, video in enumerate(loader, start=1):
        video = video.to(device)
        lr, wd = scheduler.step(step - 1)
        mask = sample_mask_from_model(model.encoder.patch_embed, video, cfg, device)
        out = model(video, mask)
        loss = criterion(out.prediction, out.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["clip_grad"])
        optimizer.step()
        print(
            f"teacher step={step} loss={loss.item():.6f} "
            f"lr={lr:.7f} wd={wd:.4f}"
        )
        log_wandb_metrics(
            wandb_run,
            {
                "train/step": step,
                "train/loss": float(loss.item()),
                "train/lr": float(lr),
                "train/weight_decay": float(wd),
            },
        )
        if step >= cfg["train"]["max_steps"]:
            break

    checkpoint_path = Path(cfg["train"]["checkpoint_path"]).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    finish_wandb_run(
        wandb_run,
        summary={
            "train/checkpoint_path": str(checkpoint_path),
            "train/final_step": step,
        },
    )


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
