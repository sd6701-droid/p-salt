from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset

from .video_dataset import (
    HuggingFaceVideoDataset,
    SQUASHFS_FILE_EXTENSIONS,
    SquashFSVideoDataset,
    SyntheticVideoDataset,
    VideoAugmentationConfig,
    VideoFileDataset,
)


def _load_manifest_paths(manifest_path: Path) -> list[Path]:
    if manifest_path.suffix.lower() == ".txt":
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        return [Path(line.strip()) for line in lines if line.strip()]

    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            for candidate in ("path", "video_path", "filepath", "file"):
                if candidate in fieldnames:
                    return [Path(row[candidate].strip()) for row in reader if row.get(candidate)]
        raise ValueError(
            f"CSV manifest {manifest_path} must include one of: path, video_path, filepath, file"
        )

    raise ValueError(f"Unsupported manifest format for {manifest_path}; use .txt or .csv")


def _collect_video_paths(dataset_cfg: dict[str, Any]) -> list[Path]:
    root = dataset_cfg.get("root")
    manifest = dataset_cfg.get("manifest")

    if manifest:
        base_dir = Path(manifest).expanduser().resolve().parent
        raw_paths = _load_manifest_paths(Path(manifest).expanduser().resolve())
        paths = [path if path.is_absolute() else (base_dir / path).resolve() for path in raw_paths]
    elif root:
        root_path = Path(root).expanduser().resolve()
        exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        paths = sorted(path for path in root_path.rglob("*") if path.suffix.lower() in exts)
    else:
        raise ValueError("Each real dataset entry must define either 'root' or 'manifest'")

    if not paths:
        raise ValueError(f"No videos found for dataset config: {dataset_cfg}")

    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None:
        sample_seed = int(dataset_cfg.get("sample_seed", 0))
        rng = random.Random(sample_seed)
        paths = list(paths)
        rng.shuffle(paths)
        paths = paths[: int(max_samples)]

    return paths


def _dataset_common_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "channels": cfg["model"]["in_channels"],
        "frames": cfg["data"]["frames"],
        "frame_step": cfg["data"]["frame_step"],
        "augmentation": VideoAugmentationConfig(
            input_size=cfg["data"]["input_size"],
            random_resize_aspect_ratio=tuple(cfg["augmentation"]["random_resize_aspect_ratio"]),
            random_resize_scale=tuple(cfg["augmentation"]["random_resize_scale"]),
        ),
    }


def build_video_dataset(cfg: dict[str, Any]) -> Dataset[torch.Tensor]:
    data_cfg = cfg["data"]
    common_kwargs = _dataset_common_kwargs(cfg)
    source = data_cfg.get("source", "synthetic")

    if source == "synthetic":
        return SyntheticVideoDataset(
            num_samples=data_cfg["num_samples"],
            height=data_cfg["image_size"],
            width=data_cfg["image_size"],
            **common_kwargs,
        )

    if source == "real":
        root = data_cfg.get("root")
        if root and Path(root).expanduser().suffix.lower() in SQUASHFS_FILE_EXTENSIONS:
            return SquashFSVideoDataset(
                archive_path=root,
                image_size=data_cfg["image_size"],
                max_samples=data_cfg.get("max_samples"),
                sample_seed=int(data_cfg.get("sample_seed", 0)),
                **common_kwargs,
            )
        video_paths = _collect_video_paths(data_cfg)
        return VideoFileDataset(
            video_paths=video_paths,
            image_size=data_cfg["image_size"],
            **common_kwargs,
        )

    if source == "squashfs":
        return SquashFSVideoDataset(
            archive_path=data_cfg["archive_path"],
            image_size=data_cfg["image_size"],
            max_samples=data_cfg.get("max_samples"),
            sample_seed=int(data_cfg.get("sample_seed", 0)),
            **common_kwargs,
        )

    if source == "huggingface":
        return HuggingFaceVideoDataset(
            repo_id=data_cfg["repo_id"],
            config_name=data_cfg.get("config_name"),
            split=data_cfg.get("split", "train"),
            video_column=data_cfg.get("video_column", "video"),
            max_samples=data_cfg.get("max_samples"),
            sample_seed=int(data_cfg.get("sample_seed", 0)),
            shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 256)),
            decode_threads=int(data_cfg.get("decode_threads", 0)),
            class_names=data_cfg.get("class_names"),
            class_fraction=data_cfg.get("class_fraction"),
            annotation_csv_url=data_cfg.get("annotation_csv_url"),
            annotation_csv_path=data_cfg.get("annotation_csv_path"),
            skip_decode_errors=bool(data_cfg.get("skip_decode_errors", True)),
            image_size=data_cfg["image_size"],
            **common_kwargs,
        )

    if source == "mixture":
        datasets = []
        for _, dataset_cfg in data_cfg.get("datasets", {}).items():
            video_paths = _collect_video_paths(dataset_cfg)
            repeat = max(1, int(dataset_cfg.get("repeat", 1)))
            for _ in range(repeat):
                datasets.append(
                    VideoFileDataset(
                        video_paths=video_paths,
                        image_size=data_cfg["image_size"],
                        **common_kwargs,
                    )
                )
        if not datasets:
            raise ValueError("Mixture dataset source requires at least one entry in data.datasets")
        return ConcatDataset(datasets)

    raise ValueError(
        f"Unknown data.source '{source}'. Expected synthetic, real, squashfs, huggingface, or mixture"
    )
