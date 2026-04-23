from __future__ import annotations

import argparse
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rethinking_jepa.utils import (
    build_loader,
    build_teacher_from_cfg,
    resolve_device,
    sample_mask_from_model,
    unpack_video_batch,
)
from src.utils import load_config


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    precision = precision.lower()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _resolve_requested_device(name: str) -> torch.device:
    if name == "auto":
        return resolve_device()
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false.")
    return device


def _apply_synthetic_smoke_overrides(
    cfg: dict,
    *,
    batch_size: int,
    architecture: str,
) -> None:
    cfg["data"] = {
        "source": "synthetic",
        "num_samples": max(batch_size, 4),
        "frames": 4,
        "frame_step": 1,
        "image_size": 32,
        "input_size": 16,
    }
    cfg["augmentation"] = {
        "random_resize_aspect_ratio": [1.0, 1.0],
        "random_resize_scale": [1.0, 1.0],
    }
    cfg["model"] = {
        "architecture": architecture,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 8,
    }
    cfg.setdefault("loss", {})["norm_pix_loss"] = True
    cfg.setdefault("loss", {})["norm_pix_eps"] = 1.0e-6


def _prepare_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    cfg.setdefault("wandb", {})["enabled"] = False
    cfg.setdefault("train", {})
    cfg["train"]["device_batch_size"] = args.batch_size
    cfg["train"]["num_workers"] = args.num_workers
    cfg["train"].pop("prefetch_factor", None)
    if args.synthetic_smoke:
        _apply_synthetic_smoke_overrides(
            cfg,
            batch_size=args.batch_size,
            architecture=args.synthetic_smoke_architecture,
        )
    return cfg


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = _prepare_cfg(load_config(args.config), args)
    device = _resolve_requested_device(args.device)
    precision = cfg["train"].get("precision", "fp32") if args.precision == "config" else args.precision

    model, _ = build_teacher_from_cfg(cfg, device)
    loader = build_loader(cfg)
    batch = next(iter(loader))
    video, _ = unpack_video_batch(batch, device)
    mask = sample_mask_from_model(model.encoder.patch_embed, video, cfg, device)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    print(
        "single-batch overfit start "
        f"device={device} precision={precision} steps={args.steps} "
        f"batch_shape={tuple(video.shape)} total_tokens={mask.size(1)} "
        f"masked_tokens={int(mask[0].sum().item())} visible_tokens={int((~mask[0]).sum().item())}"
    )

    final_loss = None
    for step in range(args.steps + 1):
        with _autocast_context(device, str(precision)):
            out = model(video, mask)

        prediction = out.prediction.float()
        target = out.target.float()
        loss = F.mse_loss(prediction, target)
        final_loss = float(loss.item())

        if step % args.log_interval == 0 or step == args.steps:
            print(
                f"step={step:4d} loss={final_loss:.6f} "
                f"pred_mean={prediction.mean().item():+.4f} pred_std={prediction.std(unbiased=False).item():.4f} "
                f"target_mean={target.mean().item():+.4f} target_std={target.std(unbiased=False).item():.4f}"
            )

        if step == args.steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

    print(f"single-batch overfit complete final_loss={final_loss:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher_kinetics400_local_extracted.yaml")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16", "config"), default="fp32")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Use a tiny synthetic config for local CPU smoke tests.",
    )
    parser.add_argument("--synthetic-smoke-architecture", default="vit_tiny")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
