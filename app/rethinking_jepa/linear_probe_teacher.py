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


def load_frozen_teacher_encoder(cfg: dict, device: torch.device) -> nn.Module:
    teacher, _ = build_teacher_from_cfg(cfg, device)
    teacher.load_state_dict(
        torch.load(cfg["train"]["teacher_checkpoint"], map_location=device)
    )
    encoder = teacher.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def run(cfg: dict) -> None:
    device = resolve_device()
    encoder = load_frozen_teacher_encoder(cfg, device)
    num_params = sum(p.numel() for p in encoder.parameters())
    print(
        "linear-probe teacher: encoder loaded "
        f"device={device} embed_dim={encoder.embed_dim} "
        f"patch_size={encoder.patch_size} tubelet_size={encoder.tubelet_size} "
        f"params={num_params}"
    )
    # Step 2-6: TODO


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, type=str)
        args = parser.parse_args()
        cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
