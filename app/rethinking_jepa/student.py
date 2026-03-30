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
        build_student_from_cfg,
        build_teacher_from_cfg,
        resolve_device,
        sample_mask_from_model,
    )
    from src.utils.config import load_config
else:
    from app.rethinking_jepa.utils import (
        build_loader,
        build_scheduler,
        build_student_from_cfg,
        build_teacher_from_cfg,
        resolve_device,
        sample_mask_from_model,
    )
    from src.utils.config import load_config


def run(cfg: dict) -> None:
    device = resolve_device()
    teacher, _ = build_teacher_from_cfg(cfg, device)
    teacher.load_state_dict(torch.load(cfg["train"]["teacher_checkpoint"], map_location=device))

    student = build_student_from_cfg(cfg, teacher, device)
    loader = build_loader(cfg)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = build_scheduler(cfg, optimizer)
    criterion = nn.SmoothL1Loss()

    student.train()
    for step, video in enumerate(loader, start=1):
        video = video.to(device)
        lr, wd = scheduler.step(step - 1)
        mask = sample_mask_from_model(student.student.patch_embed, video, cfg, device)
        out = student(video, mask)
        loss = criterion(out.prediction, out.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), cfg["optimizer"]["clip_grad"])
        optimizer.step()
        print(
            f"student step={step} loss={loss.item():.6f} "
            f"lr={lr:.7f} wd={wd:.4f}"
        )
        if step >= cfg["train"]["max_steps"]:
            break

    checkpoint_path = Path(cfg["train"]["checkpoint_path"]).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), checkpoint_path)


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
