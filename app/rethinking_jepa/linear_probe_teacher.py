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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def main(cfg: dict | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args() if cfg is None else argparse.Namespace(
        checkpoint=None,
        config=None,
    )
    checkpoint_path = Path(cfg["train"]["teacher_checkpoint"]).expanduser() if cfg is not None else checkpoint_path_from_args(args)

    print(f"checkpoint_path={checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print_keys("ckpt", ckpt)

    if isinstance(ckpt, dict):
        for nested_key in ("model", "state_dict", "teacher", "teacher_state_dict"):
            if nested_key in ckpt:
                print_keys(f"ckpt['{nested_key}']", ckpt[nested_key])


if __name__ == "__main__":
    main()
