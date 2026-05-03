# Step 1: Load the trained teacher model, freeze it, and use only its encoder for feature extraction.
# Step 2: Build a labeled video dataset/DataLoader for the probe task.
# Step 3: Convert each video batch into teacher encoder features and pool token features into one vector per video.
# Step 4: Train a small linear classifier on top of those frozen teacher features.
# Step 5: Evaluate the classifier with accuracy metrics and save the best/latest probe checkpoints.
# Step 6: Once the flow is verified, wire this script into the app entrypoint and Slurm config.

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rethinking_jepa.utils import build_teacher_from_cfg, resolve_device
from src.utils import load_config


def _strip_module_prefix(key: str) -> str:
    return key.removeprefix("module.")


def _looks_like_state_dict(value: object) -> bool:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def state_dict_from_checkpoint(checkpoint: object) -> Mapping[str, object]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected checkpoint mapping, got {type(checkpoint).__name__}")
    for key in ("model", "state_dict", "teacher", "teacher_state_dict"):
        value = checkpoint.get(key)
        if _looks_like_state_dict(value):
            return value
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    raise TypeError("Could not find a state_dict-like mapping in checkpoint")


def encoder_state_from_state_dict(state_dict: Mapping[str, object]) -> dict[str, object]:
    encoder_state: dict[str, object] = {}
    direct_encoder_prefixes = ("patch_embed.", "blocks.", "norm.")
    direct_encoder_keys = {"pos_embed"}
    for key, value in state_dict.items():
        normalized_key = _strip_module_prefix(key)
        if normalized_key.startswith("encoder."):
            encoder_state[normalized_key.removeprefix("encoder.")] = value
        elif normalized_key.startswith(direct_encoder_prefixes) or normalized_key in direct_encoder_keys:
            encoder_state[normalized_key] = value
    return encoder_state


def decoder_state_from_state_dict(state_dict: Mapping[str, object]) -> dict[str, object]:
    decoder_state: dict[str, object] = {}
    direct_decoder_prefixes = ("patch_embed.", "blocks.", "norm.")
    direct_decoder_keys = {"pos_embed"}

    for key, value in state_dict.items():
        normalized_key = _strip_module_prefix(key)
        if normalized_key.startswith("decoder."):
            decoder_state[normalized_key.removeprefix("decoder.")] = value
        elif normalized_key.startswith(direct_decoder_prefixes) or normalized_key in direct_decoder_keys:
            decoder_state[normalized_key] = value
    return decoder_state


def _value_summary(value: object) -> str:
    if torch.is_tensor(value):
        return f"shape={tuple(value.shape)} dtype={value.dtype}"
    return type(value).__name__


def print_encoder_checkpoint_summary(
    checkpoint_path: Path,
    state_dict: Mapping[str, object],
    encoder_state: Mapping[str, object],
    *,
    max_keys: int = 12,
) -> None:
    state_keys = list(state_dict.keys())
    encoder_keys = list(encoder_state.keys())
    print("teacher checkpoint summary")
    print(f"  path={checkpoint_path}")
    print(f"  total_state_keys={len(state_keys)}")
    print(f"  encoder_keys={len(encoder_keys)} encoder_weights_present={bool(encoder_keys)}")
    if encoder_keys:
        print("  encoder_key_preview:")
        for key in encoder_keys[:max_keys]:
            print(f"    {key}: {_value_summary(encoder_state[key])}")
    else:
        print("  no encoder keys found; checkpoint key preview:")
        for key in state_keys[:max_keys]:
            print(f"    {key}: {_value_summary(state_dict[key])}")

def print_decoder_checkpoint_summary(
        checkpoint_path: Path,
        state_dict: Mapping[str, object],
        decoder_state: Mapping[str, object],
        *,
        max_keys: int = 12,
) -> None:
    state_keys = list(state_dict.keys())
    decoder_keys = list(decoder_keys.keys())

    print("teacher checkpoint summary")
    print(f"  path={checkpoint_path}")
    print(f"  total_state_keys={len(state_keys)}")
    print(f"  decoder_keys={len(decoder_keys)} decoder_weights_present={bool(decoder_keys)}")

    if decoder_keys: 
        print("decoder_key_perview: ")
        for key in decoder_keys[:max_keys]:
            print(f"    {key}: {_value_summary(decoder_state[key])}")


def print_load_result_summary(load_result: torch.nn.modules.module._IncompatibleKeys) -> None:
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    print("teacher encoder load summary")
    print(f"  missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print(f"  missing_key_preview={missing[:12]}")
    if unexpected:
        print(f"  unexpected_key_preview={unexpected[:12]}")


def load_frozen_teacher_encoder(cfg: dict, device: torch.device) -> nn.Module:
    teacher, _ = build_teacher_from_cfg(cfg, device)
    checkpoint_path = Path(cfg["train"]["teacher_checkpoint"]).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = state_dict_from_checkpoint(checkpoint)
    encoder_state = encoder_state_from_state_dict(state_dict)
    decoder_state = decoder_state_from_state_dict(state_dict)
    print_encoder_checkpoint_summary(checkpoint_path, state_dict, encoder_state)
    print_decoder_checkpoint_summary(checkpoint_path, state_dict, decoder_state)
    encoder = teacher.encoder
    if not encoder_state:
        raise RuntimeError("Teacher checkpoint does not contain encoder weights")
    load_result = encoder.load_state_dict(encoder_state, strict=False)
    print_load_result_summary(load_result)
    if load_result.missing_keys:
        raise RuntimeError("Teacher encoder checkpoint is missing expected encoder weights")
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
