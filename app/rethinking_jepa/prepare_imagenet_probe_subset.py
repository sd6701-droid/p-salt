from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from torchvision.datasets import ImageNet

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.datasets import related_imagenet_labels_for_actions
    from src.utils import load_config
else:
    from src.datasets import related_imagenet_labels_for_actions
    from src.utils import load_config


def _sanitize_label(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _matches_related_label(class_names: tuple[str, ...], related_labels: set[str]) -> bool:
    lower_names = {name.lower() for name in class_names}
    return any(label.lower() in lower_names for label in related_labels)


def _class_names_from_hf_label(label_name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in label_name.split(","))


def _compute_target_counts(class_names: list[str], max_total_images: int) -> dict[str, int]:
    if max_total_images <= 0:
        raise ValueError(f"max_total_images must be positive, got {max_total_images}")
    ordered = sorted(class_names)
    base = max_total_images // len(ordered)
    remainder = max_total_images % len(ordered)
    return {
        class_name: base + (1 if idx < remainder else 0)
        for idx, class_name in enumerate(ordered)
    }


def _prepare_staging_dir(output_root: Path) -> Path:
    staging_root = output_root / "_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root


def _split_staged_subset(
    staging_root: Path,
    output_root: Path,
    *,
    val_fraction: float,
    sample_seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    train_root = output_root / "train"
    val_root = output_root / "val"
    for path in (train_root, val_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(sample_seed)
    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}

    for class_dir in sorted(path for path in staging_root.iterdir() if path.is_dir()):
        files = sorted(path for path in class_dir.iterdir() if path.is_file())
        if not files:
            continue
        rng.shuffle(files)
        val_count = int(round(len(files) * val_fraction))
        if len(files) > 1:
            val_count = max(1, min(len(files) - 1, val_count))
        else:
            val_count = 0

        val_files = files[:val_count]
        train_files = files[val_count:]
        train_class_dir = train_root / class_dir.name
        val_class_dir = val_root / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for src_path in train_files:
            shutil.move(str(src_path), train_class_dir / src_path.name)
        for src_path in val_files:
            shutil.move(str(src_path), val_class_dir / src_path.name)

        train_counts[class_dir.name] = len(train_files)
        val_counts[class_dir.name] = len(val_files)

    shutil.rmtree(staging_root)
    return train_counts, val_counts


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    os.symlink(src, dst)


def _build_subset_for_split(
    imagenet_root: Path,
    split: str,
    output_root: Path,
    related_labels: list[str],
    max_per_class: int | None,
    sample_seed: int,
    mode: str,
) -> dict[str, int]:
    dataset = ImageNet(str(imagenet_root), split=split)
    related_label_set = set(related_labels)
    samples_by_class: dict[str, list[Path]] = {}

    for path_str, target_idx in dataset.samples:
        class_names = dataset.classes[target_idx]
        if not _matches_related_label(class_names, related_label_set):
            continue
        primary_name = class_names[0]
        samples_by_class.setdefault(primary_name, []).append(Path(path_str))

    if not samples_by_class:
        raise RuntimeError(
            f"No ImageNet classes matched the related label set for split='{split}'. "
            "Check your ImageNet root and related label definitions."
        )

    rng = random.Random(sample_seed)
    split_output = output_root / split
    if split_output.exists():
        shutil.rmtree(split_output)
    split_output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for class_name, paths in sorted(samples_by_class.items()):
        paths = list(paths)
        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        class_dir = split_output / _sanitize_label(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx, src_path in enumerate(paths):
            dst_name = f"{idx:05d}_{src_path.name}"
            _link_or_copy(src_path, class_dir / dst_name, mode=mode)
        counts[class_name] = len(paths)
    return counts


def _build_subset_for_split_from_hf(
    repo_id: str,
    split: str,
    output_root: Path,
    related_labels: list[str],
    max_total_images: int,
    val_fraction: float,
    sample_seed: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face ImageNet subset preparation requires the 'datasets' package."
        ) from exc

    dataset = load_dataset(repo_id, split=split, streaming=True)
    if "label" not in dataset.features or "image" not in dataset.features:
        raise ValueError(
            f"Hugging Face dataset '{repo_id}' split '{split}' must expose 'image' and 'label' features."
        )

    label_feature = dataset.features["label"]
    if not hasattr(label_feature, "names"):
        raise ValueError(
            f"Hugging Face dataset '{repo_id}' split '{split}' must use a ClassLabel 'label' feature."
        )

    related_label_set = set(related_labels)
    target_label_ids: dict[int, str] = {}
    for label_idx, label_name in enumerate(label_feature.names):
        class_names = _class_names_from_hf_label(label_name)
        if not _matches_related_label(class_names, related_label_set):
            continue
        target_label_ids[label_idx] = class_names[0]

    if not target_label_ids:
        raise RuntimeError(
            f"No Hugging Face ImageNet classes matched the related label set for split='{split}'."
        )

    target_counts = _compute_target_counts(list(set(target_label_ids.values())), max_total_images)
    staging_root = _prepare_staging_dir(output_root)
    requested_total = sum(target_counts.values())

    print(
        "starting Hugging Face ImageNet subset scan "
        f"repo_id={repo_id} split={split} requested_total={requested_total}"
    )
    print("target_counts_per_class:")
    for class_name, target_count in sorted(target_counts.items()):
        print(f"  {class_name}: {target_count}")

    counts = {class_name: 0 for class_name in sorted(set(target_label_ids.values()))}
    for example_idx, example in enumerate(dataset):
        label_idx = int(example["label"])
        if label_idx not in target_label_ids:
            if example_idx > 0 and example_idx % 5000 == 0:
                print(
                    f"hf scan progress scanned={example_idx} "
                    f"saved={sum(counts.values())}/{requested_total}"
                )
            continue

        class_name = target_label_ids[label_idx]
        if counts[class_name] >= target_counts[class_name]:
            if all(counts[name] >= target_counts[name] for name in counts):
                break
            if example_idx > 0 and example_idx % 5000 == 0:
                print(
                    f"hf scan progress scanned={example_idx} "
                    f"saved={sum(counts.values())}/{requested_total}"
                )
            continue

        image = example["image"]
        if image is None:
            continue
        image = image.convert("RGB")

        class_dir = staging_root / _sanitize_label(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        dst_path = class_dir / f"{counts[class_name]:05d}_{example_idx:08d}.png"
        image.save(dst_path)
        counts[class_name] += 1
        saved_total = sum(counts.values())
        if saved_total % 250 == 0 or saved_total == requested_total:
            print(
                f"hf save progress scanned={example_idx + 1} "
                f"saved={saved_total}/{requested_total}"
            )

        if all(counts[name] >= target_counts[name] for name in counts):
            break

    counts = {key: value for key, value in counts.items() if value > 0}
    if not counts:
        raise RuntimeError(
            f"No Hugging Face examples were written for split='{split}'. "
            "Check the dataset repo, access permissions, or related class mapping."
        )
    saved_total = sum(counts.values())
    if saved_total < requested_total:
        print(
            f"warning: collected {saved_total} images out of requested {requested_total}. "
            "Some related classes may be sparse in the source split."
        )
    train_counts, val_counts = _split_staged_subset(
        staging_root,
        output_root,
        val_fraction=val_fraction,
        sample_seed=sample_seed,
    )
    return train_counts, val_counts, target_counts


def run(cfg: dict) -> None:
    subset_cfg = cfg["subset"]
    output_root = Path(subset_cfg["output_root"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    action_classes = list(subset_cfg["action_classes"])
    related_labels = related_imagenet_labels_for_actions(action_classes)
    source = str(subset_cfg.get("source", "local")).lower()
    mode = str(subset_cfg.get("copy_mode", "symlink")).lower()
    if mode not in {"symlink", "copy"}:
        raise ValueError(f"Unsupported copy_mode '{mode}'. Use 'symlink' or 'copy'.")

    if source == "local":
        imagenet_root = Path(subset_cfg["imagenet_root"]).expanduser()
        train_counts = _build_subset_for_split(
            imagenet_root=imagenet_root,
            split="train",
            output_root=output_root,
            related_labels=related_labels,
            max_per_class=subset_cfg.get("max_train_per_class"),
            sample_seed=int(subset_cfg.get("sample_seed", 0)),
            mode=mode,
        )
        val_counts = _build_subset_for_split(
            imagenet_root=imagenet_root,
            split="val",
            output_root=output_root,
            related_labels=related_labels,
            max_per_class=subset_cfg.get("max_val_per_class"),
            sample_seed=int(subset_cfg.get("sample_seed", 0)),
            mode=mode,
        )
        source_summary: dict[str, str] = {
            "source": source,
            "imagenet_root": str(imagenet_root),
            "copy_mode": mode,
        }
    elif source == "huggingface":
        repo_id = str(subset_cfg["hf_repo_id"])
        source_split = str(subset_cfg.get("hf_source_split", "train"))
        max_total_images = int(subset_cfg.get("max_total_images", 10000))
        val_fraction = float(subset_cfg.get("val_fraction", 0.2))
        sample_seed = int(subset_cfg.get("sample_seed", 0))
        train_counts, val_counts, target_counts = _build_subset_for_split_from_hf(
            repo_id=repo_id,
            split=source_split,
            output_root=output_root,
            related_labels=related_labels,
            max_total_images=max_total_images,
            val_fraction=val_fraction,
            sample_seed=sample_seed,
        )
        source_summary = {
            "source": source,
            "hf_repo_id": repo_id,
            "hf_source_split": source_split,
            "max_total_images": max_total_images,
            "val_fraction": val_fraction,
            "sample_seed": sample_seed,
            "target_counts_per_class": target_counts,
            "copy_mode": "copy",
        }
    else:
        raise ValueError(f"Unsupported subset source '{source}'. Use 'local' or 'huggingface'.")

    summary = {
        "output_root": str(output_root),
        "action_classes": action_classes,
        "related_imagenet_labels": related_labels,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "train_total": int(sum(train_counts.values())),
        "val_total": int(sum(val_counts.values())),
        "total_images": int(sum(train_counts.values()) + sum(val_counts.values())),
    }
    summary.update(source_summary)
    (output_root / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("imagenet probe subset prepared")
    print(f"output_root: {output_root}")
    for key, value in source_summary.items():
        print(f"{key}: {value}")
    print(f"related_imagenet_labels: {related_labels}")
    print("train_counts:")
    for key, value in train_counts.items():
        print(f"  {key}: {value}")
    print("val_counts:")
    for key, value in val_counts.items():
        print(f"  {key}: {value}")
    print(
        f"totals: train={sum(train_counts.values())} "
        f"val={sum(val_counts.values())} total={sum(train_counts.values()) + sum(val_counts.values())}"
    )


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
