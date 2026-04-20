from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.video_dataset import SquashFSVideoDataset


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


def _write_summary(
    *,
    archive_path: str,
    output_dir: Path,
    manifest_path: Path,
    extracted_paths: list[Path],
    class_names: list[str] | None,
    max_samples: int | None,
    max_samples_per_class: int | None,
    sample_seed: int,
    elapsed_seconds: float,
) -> None:
    by_class = Counter(path.parent.name for path in extracted_paths)
    summary = {
        "archive_path": archive_path,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "num_videos": len(extracted_paths),
        "class_names": class_names,
        "max_samples": max_samples,
        "max_samples_per_class": max_samples_per_class,
        "sample_seed": sample_seed,
        "elapsed_seconds": elapsed_seconds,
        "class_counts": dict(sorted(by_class.items())),
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
    max_samples = data_cfg.get("max_samples")
    max_samples_per_class = data_cfg.get("max_samples_per_class")
    progress_interval = int(data_cfg.get("progress_interval", 25))

    dataset = SquashFSVideoDataset(
        archive_path=archive_path,
        channels=int(data_cfg.get("channels", 3)),
        frames=int(data_cfg.get("frames", 16)),
        frame_step=int(data_cfg.get("frame_step", 4)),
        image_size=int(data_cfg.get("image_size", 224)),
        augmentation=None,
        max_samples=max_samples,
        sample_seed=sample_seed,
        class_names=class_names,
        class_fraction=data_cfg.get("class_fraction"),
        max_samples_per_class=max_samples_per_class,
        cache_dir=None,
        unsquashfs_path=data_cfg.get("unsquashfs_path"),
        sqfscat_path=data_cfg.get("sqfscat_path"),
    )

    videos_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset.archive_entries)
    print(f"extracting {total} videos from {archive_path} to {videos_dir}")

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
    _write_summary(
        archive_path=archive_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        extracted_paths=extracted_paths,
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
                "class_names": args.class_names,
                "progress_interval": args.progress_interval,
            }
        }
    )


if __name__ == "__main__":
    main()
