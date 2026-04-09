from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.rethinking_jepa.utils import build_student_from_cfg, build_teacher_from_cfg, resolve_device
    from src.datasets import ImageFolderRepeatedFrameDataset
    from src.models import FrozenStudentLinearProbe
    from src.utils import (
        CosineScheduler,
        finish_wandb_run,
        init_wandb_run,
        load_config,
        log_wandb_artifact,
        log_wandb_metrics,
    )
else:
    from app.rethinking_jepa.utils import build_student_from_cfg, build_teacher_from_cfg, resolve_device
    from src.datasets import ImageFolderRepeatedFrameDataset
    from src.models import FrozenStudentLinearProbe
    from src.utils import (
        CosineScheduler,
        finish_wandb_run,
        init_wandb_run,
        load_config,
        log_wandb_artifact,
        log_wandb_metrics,
    )


def _extract_student_encoder_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    state = checkpoint.get("state_dict")
    if isinstance(state, dict):
        checkpoint = state
    encoder_state = {
        key[len("student.") :]: value
        for key, value in checkpoint.items()
        if key.startswith("student.") and torch.is_tensor(value)
    }
    if not encoder_state:
        raise ValueError("No 'student.' encoder weights found in the provided student checkpoint")
    return encoder_state


def _build_loader(
    root: str | Path,
    cfg: dict,
    *,
    class_names: list[str] | None,
    max_samples: int | None,
    shuffle: bool,
) -> tuple[DataLoader, ImageFolderRepeatedFrameDataset]:
    data_cfg = cfg["data"]
    probe_cfg = cfg["linear_probe"]
    dataset = ImageFolderRepeatedFrameDataset(
        root=root,
        input_size=int(data_cfg["input_size"]),
        frames=int(data_cfg["frames"]),
        resize_size=int(data_cfg.get("resize_size", data_cfg["input_size"])),
        max_samples=max_samples,
        sample_seed=int(probe_cfg.get("sample_seed", 0)),
        class_names=class_names if class_names is not None else data_cfg.get("class_names"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(probe_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(probe_cfg.get("num_workers", 0)),
    )
    return loader, dataset


def _iter_forever(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _discover_shared_classes(train_root: str | Path, val_root: str | Path) -> list[str]:
    train_root = Path(train_root).expanduser()
    val_root = Path(val_root).expanduser()
    train_classes = {path.name for path in train_root.iterdir() if path.is_dir()}
    val_classes = {path.name for path in val_root.iterdir() if path.is_dir()}
    shared = sorted(train_classes & val_classes)
    if not shared:
        raise ValueError(
            "No shared class folders were found between the linear probe train and validation roots"
        )
    return shared


def _format_prediction_table(sample_predictions: list[dict[str, object]]) -> str:
    headers = ["image name", "label", "pred", "confidence", "correct"]
    rows = [
        [
            str(item["image_name"]),
            str(item["label"]),
            str(item["prediction"]),
            f"{float(item['confidence']):.6f}",
            str(bool(item["correct"])),
        ]
        for item in sample_predictions
    ]
    table = [headers] + rows
    widths = [max(len(row[col]) for row in table) for col in range(len(headers))]

    def _fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    lines = [_fmt(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def _evaluate_linear_probe(
    probe: FrozenStudentLinearProbe,
    loader: DataLoader,
    dataset: ImageFolderRepeatedFrameDataset,
    device: torch.device,
    *,
    max_prediction_rows: int = 20,
) -> tuple[dict[str, float | int], list[dict[str, object]], dict[str, torch.Tensor]]:
    probe.eval()
    criterion = nn.CrossEntropyLoss()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    sample_predictions: list[dict[str, object]] = []
    pooled_features: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            labels = batch["label_index"].to(device)
            logits, pooled = probe(video)
            loss = criterion(logits, labels)
            probs = logits.softmax(dim=-1)
            pred = probs.argmax(dim=-1)
            topk = min(5, logits.size(-1))
            top5 = probs.topk(topk, dim=-1).indices
            total_examples += video.size(0)
            total_loss += float(loss.item()) * video.size(0)
            total_correct += int((pred == labels).sum().item())
            total_top5 += int((top5 == labels.unsqueeze(1)).any(dim=1).sum().item())

            labels_cpu = labels.cpu()
            pred_cpu = pred.cpu()
            probs_cpu = probs.cpu()
            pooled_cpu = pooled.cpu()
            paths = batch["path"]
            label_names = batch["label"]
            if not isinstance(paths, list):
                paths = [paths]
            if not isinstance(label_names, list):
                label_names = [label_names]

            for idx in range(video.size(0)):
                sample_id = f"{total_examples - video.size(0) + idx + 1:03d}_{Path(str(paths[idx])).stem}"
                pooled_features[sample_id] = pooled_cpu[idx]
                if len(sample_predictions) < max_prediction_rows:
                    pred_name = dataset.classes[int(pred_cpu[idx].item())]
                    conf = float(probs_cpu[idx, pred_cpu[idx]].item())
                    sample_predictions.append(
                        {
                            "sample_id": sample_id,
                            "image_name": Path(str(paths[idx])).name,
                            "label": str(label_names[idx]),
                            "prediction": pred_name,
                            "confidence": conf,
                            "correct": bool(pred_cpu[idx].item() == labels_cpu[idx].item()),
                            "path": str(paths[idx]),
                        }
                    )

    if total_examples == 0:
        raise RuntimeError("No validation examples were processed during linear probe evaluation.")

    metrics = {
        "num_examples": total_examples,
        "loss": total_loss / total_examples,
        "accuracy_top1": total_correct / total_examples,
        "accuracy_top5": total_top5 / total_examples,
    }
    return metrics, sample_predictions, pooled_features


def _save_linear_probe_checkpoint(probe: FrozenStudentLinearProbe, checkpoint_path: Path, step: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), checkpoint_path)
    print(f"linear probe checkpoint saved step={step} path={checkpoint_path}")


def run(cfg: dict) -> None:
    device = resolve_device()
    probe_cfg = cfg["linear_probe"]
    output_dir = Path(probe_cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher, _ = build_teacher_from_cfg(cfg, device)
    student = build_student_from_cfg(cfg, teacher, device)

    student_ckpt = Path(probe_cfg["student_checkpoint"]).expanduser()
    if not student_ckpt.exists():
        raise FileNotFoundError(f"Student checkpoint not found at '{student_ckpt}'")
    checkpoint = torch.load(student_ckpt, map_location=device)
    encoder_state = _extract_student_encoder_state_dict(checkpoint)
    student.student.load_state_dict(encoder_state, strict=True)

    configured_class_names = cfg["data"].get("class_names") or None
    shared_class_names = (
        list(configured_class_names)
        if configured_class_names is not None
        else _discover_shared_classes(probe_cfg["train_root"], probe_cfg["val_root"])
    )

    train_loader, train_dataset = _build_loader(
        root=probe_cfg["train_root"],
        cfg=cfg,
        class_names=shared_class_names,
        max_samples=probe_cfg.get("max_train_samples"),
        shuffle=True,
    )
    val_loader, val_dataset = _build_loader(
        root=probe_cfg["val_root"],
        cfg=cfg,
        class_names=shared_class_names,
        max_samples=probe_cfg.get("max_val_samples"),
        shuffle=False,
    )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and validation class sets differ for the linear probe dataset")

    num_classes = len(train_dataset.classes)
    probe = FrozenStudentLinearProbe(
        student_encoder=student.student,
        num_classes=num_classes,
        pool=str(probe_cfg.get("pool", "mean")),
        dropout=float(probe_cfg.get("dropout", 0.0)),
    ).to(device)

    batch_size = int(probe_cfg.get("batch_size", 1))
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / batch_size))
    epochs = int(probe_cfg.get("epochs", 1))
    max_steps_cfg = probe_cfg.get("max_steps")
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else steps_per_epoch * epochs
    print(
        "linear probe dataset "
        f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
        f"num_classes={num_classes} batch_size={batch_size} "
        f"steps_per_epoch={steps_per_epoch} epochs={epochs} max_steps={max_steps}"
    )

    wandb_run = init_wandb_run(cfg, job_type="student-linear-probe")
    optimizer = torch.optim.AdamW(
        probe.classifier.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = CosineScheduler(
        optimizer=optimizer,
        total_steps=max_steps,
        warmup_steps=cfg["optimizer"]["warmup_steps"],
        start_lr=cfg["optimizer"]["start_lr"],
        peak_lr=cfg["optimizer"]["lr"],
        final_lr=cfg["optimizer"]["final_lr"],
        start_weight_decay=cfg["optimizer"]["start_weight_decay"],
        end_weight_decay=cfg["optimizer"]["end_weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = Path(probe_cfg["checkpoint_path"]).expanduser()
    checkpoint_every_steps = int(probe_cfg.get("checkpoint_every_steps", 0))
    eval_every_steps = int(probe_cfg.get("eval_every_steps", 0))
    student_arch = str(cfg.get("student_model", {}).get("architecture", cfg["model"]["architecture"]))
    teacher_arch = str(cfg["model"]["architecture"])
    probe_metadata = {
        "model/teacher_architecture": teacher_arch,
        "model/student_architecture": student_arch,
        "linear_probe/pool": str(probe_cfg.get("pool", "mean")),
        "linear_probe/dropout": float(probe_cfg.get("dropout", 0.0)),
        "linear_probe/batch_size": batch_size,
        "linear_probe/num_classes": num_classes,
        "linear_probe/student_checkpoint": str(student_ckpt),
        "linear_probe/checkpoint_path": str(checkpoint_path),
        "linear_probe/output_dir": str(output_dir),
        "linear_probe/train_samples": len(train_dataset),
        "linear_probe/val_samples": len(val_dataset),
        "linear_probe/steps_per_epoch": steps_per_epoch,
        "linear_probe/epochs": epochs,
        "linear_probe/max_steps": max_steps,
    }
    log_wandb_metrics(wandb_run, probe_metadata)

    step = 0
    train_iter = _iter_forever(train_loader)
    probe.train()
    for step in range(1, max_steps + 1):
        batch = next(train_iter)
        video = batch["video"].to(device)
        labels = batch["label_index"].to(device)
        lr, wd = scheduler.step(step - 1)
        logits, _ = probe(video)
        loss = criterion(logits, labels)
        acc = float((logits.argmax(dim=-1) == labels).float().mean().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.classifier.parameters(), cfg["optimizer"]["clip_grad"])
        optimizer.step()
        print(
            f"linear probe step={step} loss={loss.item():.6f} "
            f"acc={acc:.4f} lr={lr:.7f} wd={wd:.4f}"
        )
        log_wandb_metrics(
            wandb_run,
            {
                "train/step": step,
                "train/loss": float(loss.item()),
                "train/accuracy_top1": acc,
                "train/lr": float(lr),
                "train/weight_decay": float(wd),
            },
        )
        if checkpoint_every_steps > 0 and step % checkpoint_every_steps == 0:
            _save_linear_probe_checkpoint(probe, checkpoint_path, step)
        if eval_every_steps > 0 and step % eval_every_steps == 0:
            val_metrics, _, _ = _evaluate_linear_probe(
                probe,
                val_loader,
                val_dataset,
                device,
                max_prediction_rows=int(probe_cfg.get("num_prediction_rows", 20)),
            )
            print(
                "linear probe eval "
                f"step={step} val_loss={val_metrics['loss']:.6f} "
                f"val_acc={val_metrics['accuracy_top1']:.4f}"
            )
            log_wandb_metrics(
                wandb_run,
                {
                    "eval/step": step,
                    "eval/loss": float(val_metrics["loss"]),
                    "eval/accuracy_top1": float(val_metrics["accuracy_top1"]),
                    "eval/accuracy_top5": float(val_metrics["accuracy_top5"]),
                },
            )

    _save_linear_probe_checkpoint(probe, checkpoint_path, max_steps)
    final_metrics, sample_predictions, pooled_features = _evaluate_linear_probe(
        probe,
        val_loader,
        val_dataset,
        device,
        max_prediction_rows=int(probe_cfg.get("num_prediction_rows", 20)),
    )
    summary = {
        "model/teacher_architecture": teacher_arch,
        "model/student_architecture": student_arch,
        "linear_probe/student_checkpoint": str(student_ckpt),
        "linear_probe/checkpoint_path": str(checkpoint_path),
        "linear_probe/output_dir": str(output_dir),
        "linear_probe/final_step": max_steps,
        "linear_probe/num_classes": num_classes,
        "linear_probe/num_examples": int(final_metrics["num_examples"]),
        "linear_probe/val_loss": float(final_metrics["loss"]),
        "linear_probe/val_accuracy_top1": float(final_metrics["accuracy_top1"]),
        "linear_probe/val_accuracy_top5": float(final_metrics["accuracy_top5"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "sample_predictions.json").write_text(
        json.dumps(sample_predictions, indent=2), encoding="utf-8"
    )
    with (output_dir / "sample_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_name", "label", "prediction", "confidence", "correct", "path"],
        )
        writer.writeheader()
        writer.writerows(sample_predictions)
    table_text = _format_prediction_table(sample_predictions)
    (output_dir / "sample_predictions_table.txt").write_text(table_text + "\n", encoding="utf-8")
    torch.save(pooled_features, output_dir / "val_pooled_features.pt")

    log_wandb_metrics(wandb_run, summary)
    artifact_paths = [
        str(output_dir / "summary.json"),
        str(output_dir / "sample_predictions.json"),
        str(output_dir / "sample_predictions.csv"),
        str(output_dir / "sample_predictions_table.txt"),
        str(output_dir / "val_pooled_features.pt"),
        str(checkpoint_path),
    ]
    log_wandb_artifact(
        wandb_run,
        name=f"linear-probe-{student_arch}-analysis",
        artifact_type="evaluation",
        paths=artifact_paths,
        metadata=summary,
    )
    finish_wandb_run(wandb_run, summary=summary)

    print("student linear probe complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"sample_predictions_path: {output_dir / 'sample_predictions.json'}")
    print("")
    print(table_text)
    print("")
    print(f"sample_predictions_table_path: {output_dir / 'sample_predictions_table.txt'}")
    print(f"pooled_features_path: {output_dir / 'val_pooled_features.pt'}")


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
