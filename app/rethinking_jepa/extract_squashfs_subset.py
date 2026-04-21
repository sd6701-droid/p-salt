from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.video_dataset import SquashFSVideoDataset

KNOWN_DATASET_SPLITS = ("train_256", "val_256", "test_256", "train", "val", "test")


class ExtractionProgress:
    def __init__(self, total: int, progress_interval: int) -> None:
        self.total = max(1, total)
        self.progress_interval = max(1, progress_interval)
        self.start_time = time.monotonic()
        self.use_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._last_render_len = 0

    def update(self, current: int, entry: str) -> None:
        elapsed = max(1e-6, time.monotonic() - self.start_time)
        rate = current / elapsed
        percent = (100.0 * current) / self.total
        message = (
            f"extract progress {current}/{self.total} "
            f"({percent:5.1f}%) rate={rate:5.2f} files/s current={entry}"
        )

        if self.use_tty:
            padded = message.ljust(self._last_render_len)
            print(f"\r{padded}", end="", flush=True)
            self._last_render_len = max(self._last_render_len, len(message))
            if current >= self.total:
                print()
            return

        if current == 1 or current % self.progress_interval == 0 or current >= self.total:
            print(message, flush=True)


def _write_manifest(video_paths: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(str(path.resolve()) for path in video_paths) + "\n",
        encoding="utf-8",
    )


def _infer_split_name(path: Path) -> str:
    for part in path.parts:
        if part in KNOWN_DATASET_SPLITS:
            return part
    return "unknown"


def _class_name_from_entry(entry: str) -> str:
    parent = Path(entry).parent.name
    if not parent:
        raise ValueError(f"Could not infer class name from archive entry '{entry}'")
    return parent


def _sample_class_names(entries: list[str], num_classes: int, class_seed: int) -> list[str]:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    available_classes = sorted({_class_name_from_entry(entry) for entry in entries})
    if num_classes > len(available_classes):
        raise ValueError(
            f"Requested num_classes={num_classes}, but archive only has {len(available_classes)} classes"
        )

    rng = random.Random(class_seed)
    return sorted(rng.sample(available_classes, num_classes))


def _write_split_manifests(output_dir: Path, video_paths: list[Path]) -> dict[str, str]:
    by_split: dict[str, list[Path]] = {}
    for path in video_paths:
        split_name = _infer_split_name(path)
        by_split.setdefault(split_name, []).append(path)

    manifest_paths: dict[str, str] = {}
    for split_name, split_paths in sorted(by_split.items()):
        manifest_path = output_dir / f"{split_name}_manifest.txt"
        _write_manifest(sorted(split_paths), manifest_path)
        manifest_paths[split_name] = str(manifest_path)
        print(f"{split_name} manifest written to {manifest_path}")

    return manifest_paths


def _write_summary(
    *,
    archive_path: str,
    output_dir: Path,
    manifest_path: Path,
    extracted_paths: list[Path],
    split_manifest_paths: dict[str, str],
    class_names: list[str] | None,
    max_samples: int | None,
    max_samples_per_class: int | None,
    sample_seed: int,
    elapsed_seconds: float,
) -> None:
    by_class = Counter(path.parent.name for path in extracted_paths)
    by_split = Counter(_infer_split_name(path) for path in extracted_paths)
    by_split_and_class: dict[str, dict[str, int]] = {}
    for split_name in sorted(by_split):
        split_paths = [path for path in extracted_paths if _infer_split_name(path) == split_name]
        by_split_and_class[split_name] = dict(
            sorted(Counter(path.parent.name for path in split_paths).items())
        )

    summary = {
        "archive_path": archive_path,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "split_manifest_paths": split_manifest_paths,
        "num_videos": len(extracted_paths),
        "class_names": class_names,
        "max_samples": max_samples,
        "max_samples_per_class": max_samples_per_class,
        "sample_seed": sample_seed,
        "elapsed_seconds": elapsed_seconds,
        "class_counts": dict(sorted(by_class.items())),
        "split_counts": dict(sorted(by_split.items())),
        "split_class_counts": by_split_and_class,
    }
    summary_path = output_dir / "subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary written to {summary_path}")


def run(cfg: dict) -> None:
    data_cfg = cfg["data"]
    archive_path = str(Path(data_cfg["archive_path"]).expanduser().resolve())
    output_dir = Path(data_cfg["output_dir"]).expanduser().resolve()
    videos_dir = output_dir / "videos"
    manifest_path = Path(
        data_cfg.get("manifest_path", output_dir / "manifest.txt")
    ).expanduser().resolve()
    sample_seed = int(data_cfg.get("sample_seed", 0))
    class_names = data_cfg.get("class_names")
    num_classes = data_cfg.get("num_classes")
    class_seed = int(data_cfg.get("class_seed", sample_seed))
    max_samples = data_cfg.get("max_samples")
    max_samples_per_class = data_cfg.get("max_samples_per_class")
    progress_interval = int(data_cfg.get("progress_interval", 25))

    if class_names is not None and num_classes is not None:
        raise ValueError("Specify either data.class_names or data.num_classes, not both")

    dataset_kwargs = {
        "archive_path": archive_path,
        "channels": int(data_cfg.get("channels", 3)),
        "frames": int(data_cfg.get("frames", 16)),
        "frame_step": int(data_cfg.get("frame_step", 4)),
        "image_size": int(data_cfg.get("image_size", 224)),
        "augmentation": None,
        "sample_seed": sample_seed,
        "class_fraction": data_cfg.get("class_fraction"),
        "cache_dir": None,
        "unsquashfs_path": data_cfg.get("unsquashfs_path"),
        "sqfscat_path": data_cfg.get("sqfscat_path"),
    }

    if num_classes is not None:
        unfiltered_dataset = SquashFSVideoDataset(
            **dataset_kwargs,
            max_samples=None,
            class_names=None,
            max_samples_per_class=None,
        )
        class_names = _sample_class_names(
            unfiltered_dataset.archive_entries,
            num_classes=int(num_classes),
            class_seed=class_seed,
        )
        print(
            f"selected {len(class_names)} classes using class_seed={class_seed}: "
            + ", ".join(class_names)
        )

    dataset = SquashFSVideoDataset(
        **dataset_kwargs,
        max_samples=max_samples,
        class_names=class_names,
        max_samples_per_class=max_samples_per_class,
    )

    videos_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset.archive_entries)
    selected_class_count = len(class_names) if class_names is not None else "all"
    print(
        f"extracting {total} videos from {archive_path} to {videos_dir} "
        f"class_count={selected_class_count}"
    )

    progress = ExtractionProgress(total=total, progress_interval=progress_interval)
    extract_start = time.monotonic()
    extracted_paths: list[Path] = []
    for idx, entry in enumerate(dataset.archive_entries, start=1):
        path = dataset.extract_entry(entry, videos_dir)
        extracted_paths.append(path)
        progress.update(idx, entry)

    elapsed = time.monotonic() - extract_start
    extracted_paths = sorted(path.resolve() for path in extracted_paths if path.exists())
    _write_manifest(extracted_paths, manifest_path)
    print(f"manifest written to {manifest_path}")
    split_manifest_paths = _write_split_manifests(output_dir, extracted_paths)
    _write_summary(
        archive_path=archive_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        extracted_paths=extracted_paths,
        split_manifest_paths=split_manifest_paths,
        class_names=class_names,
        max_samples=max_samples,
        max_samples_per_class=max_samples_per_class,
        sample_seed=sample_seed,
        elapsed_seconds=elapsed,
    )
    print(f"extraction complete videos={len(extracted_paths)} elapsed_seconds={elapsed:.2f}")


def main(cfg: dict | None = None) -> None:
    if cfg is not None:
        run(cfg)
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--class-seed", type=int, default=None)
    parser.add_argument("--class-names", nargs="*", default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    run(
        {
            "data": {
                "archive_path": args.archive_path,
                "output_dir": args.output_dir,
                "manifest_path": args.manifest_path,
                "max_samples": args.max_samples,
                "max_samples_per_class": args.max_samples_per_class,
                "sample_seed": args.sample_seed,
                "num_classes": args.num_classes,
                "class_seed": args.class_seed,
                "class_names": args.class_names,
                "progress_interval": args.progress_interval,
            }
        }
    )


if __name__ == "__main__":
    main()
