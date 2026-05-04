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
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rethinking_jepa.utils import build_teacher_from_cfg, resolve_device
from src.datasets.data_manager import _load_manifest_paths
from src.datasets.video_dataset import VideoAugmentationConfig, VideoFileDataset
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
    # this stops Batch and normalization updates right ? 
    encoder.eval()
    # for every params in the encoder we stop the flow of the gradients 
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


class LabeledVideoDataset(Dataset):
    """Wraps VideoFileDataset to also return the integer class label per sample.

    Owns its own decode-error fallback so that when a clip fails and we walk
    forward to a different index, the returned label still matches the clip
    that was actually loaded (preventing video/label mismatch).
    """

    def __init__(
        self,
        video_paths: list[Path],
        labels: list[int],
        *,
        channels: int,
        frames: int,
        frame_step: int,
        image_size: int,
        augmentation: VideoAugmentationConfig | None = None,
        skip_decode_errors: bool = False,
        max_decode_attempts: int = 16,
        log_decode_warnings: bool = True,
    ) -> None:
        if len(video_paths) != len(labels):
            raise ValueError(
                f"video_paths ({len(video_paths)}) and labels ({len(labels)}) length mismatch"
            )
        # the inner dataset is built with skip_decode_errors=False so we own the
        # fallback and can return the matching label for whichever index loaded
        self._inner = VideoFileDataset(
            video_paths=video_paths,
            channels=channels,
            frames=frames,
            frame_step=frame_step,
            image_size=image_size,
            augmentation=augmentation,
            skip_decode_errors=False,
            max_decode_attempts=max_decode_attempts,
            log_decode_warnings=log_decode_warnings,
        )
        self._labels = list(labels)
        self._skip_decode_errors = bool(skip_decode_errors)
        self._max_decode_attempts = max(1, int(max_decode_attempts))

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not self._skip_decode_errors:
            return self._inner[index], self._labels[index]
        attempts = min(self._max_decode_attempts, len(self._labels))
        last_error: Exception | None = None
        for attempt in range(attempts):
            candidate = (index + attempt) % len(self._labels)
            try:
                return self._inner[candidate], self._labels[candidate]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"Failed to decode any usable clip in {attempts} attempts starting at index {index}"
        ) from last_error


def resolve_manifest_paths(manifest_path: Path) -> list[Path]:
    base_dir = manifest_path.parent
    raw = _load_manifest_paths(manifest_path)
    return [p if p.is_absolute() else (base_dir / p).resolve() for p in raw]


def load_labeled_paths(
    manifest_path: str | Path,
    *,
    class_to_idx: dict[str, int] | None = None,
) -> tuple[list[Path], list[int], dict[str, int]]:
    # derive labels from each path's parent directory; train builds the mapping,
    # val reuses it so indices stay consistent across splits
    manifest_path = Path(manifest_path).expanduser().resolve()
    paths = resolve_manifest_paths(manifest_path)
    class_names = [p.parent.name for p in paths]
    if class_to_idx is None:
        unique = sorted(set(class_names))
        class_to_idx = {name: idx for idx, name in enumerate(unique)}
    missing = sorted({n for n in class_names if n not in class_to_idx})
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} contains classes not in class_to_idx: {missing}"
        )
    labels = [class_to_idx[n] for n in class_names]
    return paths, labels, class_to_idx


def build_probe_loader(
    cfg: dict,
    manifest_key: str,
    *,
    shuffle: bool,
    class_to_idx: dict[str, int] | None = None,
) -> tuple[DataLoader, dict[str, int]]:
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})

    manifest_path = data_cfg[manifest_key]
    paths, labels, class_to_idx = load_labeled_paths(manifest_path, class_to_idx=class_to_idx)

    # train uses the configured augmentation; val gets deterministic resize-only
    augmentation = None
    if shuffle and "augmentation" in cfg:
        aug_cfg = cfg["augmentation"]
        augmentation = VideoAugmentationConfig(
            input_size=int(data_cfg["input_size"]),
            random_resize_aspect_ratio=tuple(aug_cfg["random_resize_aspect_ratio"]),
            random_resize_scale=tuple(aug_cfg["random_resize_scale"]),
        )

    dataset = LabeledVideoDataset(
        video_paths=paths,
        labels=labels,
        channels=int(cfg["model"]["in_channels"]),
        frames=int(data_cfg["frames"]),
        frame_step=int(data_cfg["frame_step"]),
        image_size=int(data_cfg["image_size"]),
        augmentation=augmentation,
        skip_decode_errors=bool(data_cfg.get("skip_decode_errors", False)),
        max_decode_attempts=int(data_cfg.get("max_decode_attempts", 16)),
        log_decode_warnings=bool(data_cfg.get("log_decode_warnings", True)),
    )

    num_workers = int(probe_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(probe_cfg.get("batch_size", 32)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(probe_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **loader_kwargs), class_to_idx


def extract_features(
    encoder: nn.Module,
    video: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # encoder is already eval() + requires_grad_(False); inference_mode skips
    # autograd bookkeeping entirely. Mean over the token dim collapses
    # (B, N, D) -> (B, D), one vector per video.
    video = video.to(device, non_blocking=(device.type == "cuda"))
    with torch.inference_mode():
        tokens, _ = encoder(video)
    return tokens.mean(dim=1)


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

    # Step 2 sanity check: build train + val loaders, peek at one batch.
    train_loader, class_to_idx = build_probe_loader(cfg, "train_manifest", shuffle=True)
    val_loader, _ = build_probe_loader(
        cfg, "val_manifest", shuffle=False, class_to_idx=class_to_idx
    )
    print(
        "linear-probe teacher: loaders built "
        f"num_classes={len(class_to_idx)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )
    video, labels = next(iter(train_loader))
    print(
        "linear-probe teacher: first batch "
        f"video.shape={tuple(video.shape)} labels.shape={tuple(labels.shape)} "
        f"labels[:8]={labels[:8].tolist()}"
    )

    # Step 3 sanity check: encode the batch and pool tokens to one vector per video.
    features = extract_features(encoder, video, device)
    print(
        "linear-probe teacher: pooled features "
        f"features.shape={tuple(features.shape)} dtype={features.dtype} "
        f"mean={features.mean().item():.4f} std={features.std().item():.4f}"
    )


if __name__ == "__main__":
    main()
