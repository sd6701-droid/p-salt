# Step 1: Load the trained teacher model, freeze it, and use only its encoder for feature extraction.
# Step 2: Build a labeled video dataset/DataLoader for the probe task.
# Step 3: Convert each video batch into teacher encoder features and pool token features into one vector per video.
# Step 4: Train a small linear classifier on top of those frozen teacher features.
# Step 5: Evaluate the classifier with accuracy metrics and save the best/latest probe checkpoints.
# Step 6: Once the flow is verified, wire this script into the app entrypoint and Slurm config.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rethinking_jepa.utils import build_teacher_from_cfg, resolve_device
from src.utils import load_config


def checkpoint_path_from_args(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser()
    if args.config:
        cfg = load_config(args.config)
        return Path(cfg["train"]["teacher_checkpoint"]).expanduser()
    return Path("checkpoint.pth")


def print_keys(name: str, value: object) -> None:
    print(name, type(value))
    if isinstance(value, dict):
        keys = list(value.keys())
        print(f"{name}.keys()")
        for key in keys:
            print(f"  {key}")


def extract_model_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)!r}")


def load_frozen_teacher_encoder(
    cfg: dict,
    device: torch.device,
    *,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    # this build a teacher skeleton after which we load wts to this 
    teacher, _ = build_teacher_from_cfg(cfg, device)
    if checkpoint_path is None:
        checkpoint_path = Path(cfg["train"]["teacher_checkpoint"]).expanduser()
    ckpt = torch.load(checkpoint_path, map_location=device)
    teacher.load_state_dict(extract_model_state_dict(ckpt))
    encoder = teacher.encoder
    encoder.eval()
    # for every params in the encoder we stop the flow of the gradients 
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def main(cfg: dict | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args() if cfg is None else argparse.Namespace(
        checkpoint=None,
        config=None,
    )
    if cfg is None and args.config:
        cfg = load_config(args.config)
    checkpoint_path = Path(cfg["train"]["teacher_checkpoint"]).expanduser() if cfg is not None else checkpoint_path_from_args(args)

    print(f"checkpoint_path={checkpoint_path}")
    # ckpt = torch.load(checkpoint_path, map_location="cpu")
    #print_keys("ckpt", ckpt)

    if cfg is None:
        return

    #check which device we are using //cpu, apple gpu, gpu
    device = resolve_device()
    # load the frozen encoder 
    encoder = load_frozen_teacher_encoder(cfg, device, checkpoint_path=checkpoint_path)
    num_params = sum(p.numel() for p in encoder.parameters())
    print(
        "linear-probe teacher: encoder loaded "
        f"device={device} embed_dim={encoder.embed_dim} "
        f"patch_size={encoder.patch_size} tubelet_size={encoder.tubelet_size} "
        f"params={num_params}"
    )


if __name__ == "__main__":
    main()
