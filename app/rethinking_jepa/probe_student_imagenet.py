from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.rethinking_jepa.utils import (
        build_student_from_cfg,
        build_teacher_from_cfg,
        resolve_device,
        sample_mask_from_model,
    )
    from src.datasets import ImageFolderRepeatedFrameDataset
    from src.models import FrozenStudentPixelProbe
    from src.models.jepa import patchify_video
    from src.utils import (
        CosineScheduler,
        finish_wandb_run,
        init_wandb_run,
        load_config,
        log_wandb_artifact,
        log_wandb_metrics,
    )
else:
    from app.rethinking_jepa.utils import (
        build_student_from_cfg,
        build_teacher_from_cfg,
        resolve_device,
        sample_mask_from_model,
    )
    from src.datasets import ImageFolderRepeatedFrameDataset
    from src.models import FrozenStudentPixelProbe
    from src.models.jepa import patchify_video
    from src.utils import (
        CosineScheduler,
        finish_wandb_run,
        init_wandb_run,
        load_config,
        log_wandb_artifact,
        log_wandb_metrics,
    )


def _ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    ids = torch.arange(mask.size(1), device=mask.device).unsqueeze(0).expand_as(mask)
    return ids.masked_select(mask).view(mask.size(0), -1)


def _scatter_masked_values(base_tokens: torch.Tensor, mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    masked_ids = _ids_from_mask(mask)
    scatter_idx = masked_ids.unsqueeze(-1).expand(-1, -1, base_tokens.size(-1))
    output = base_tokens.clone()
    output.scatter_(1, scatter_idx, values)
    return output


def _mask_tokens_with_value(base_tokens: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    output = base_tokens.clone()
    output[mask] = value
    return output


def _unpatchify_video(
    patches: torch.Tensor,
    channels: int,
    frames: int,
    height: int,
    width: int,
    tubelet_size: int,
    patch_size: int,
) -> torch.Tensor:
    batch_size = patches.size(0)
    t_grid = frames // tubelet_size
    h_grid = height // patch_size
    w_grid = width // patch_size
    x = patches.view(
        batch_size,
        t_grid,
        h_grid,
        w_grid,
        channels,
        tubelet_size,
        patch_size,
        patch_size,
    )
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.view(batch_size, channels, frames, height, width)


def _build_loader(
    root: str | Path,
    cfg: dict,
    *,
    max_samples: int | None,
    shuffle: bool,
) -> DataLoader:
    data_cfg = cfg["data"]
    probe_cfg = cfg["probe"]
    dataset = ImageFolderRepeatedFrameDataset(
        root=root,
        input_size=int(data_cfg["input_size"]),
        frames=int(data_cfg["frames"]),
        resize_size=int(data_cfg.get("resize_size", data_cfg["input_size"])),
        max_samples=max_samples,
        sample_seed=int(probe_cfg.get("sample_seed", 0)),
        class_names=data_cfg.get("class_names"),
    )
    return DataLoader(
        dataset,
        batch_size=int(probe_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(probe_cfg.get("num_workers", 0)),
    )


def _iter_forever(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _save_probe_visualizations(
    probe: FrozenStudentPixelProbe,
    batch: dict[str, object],
    cfg: dict,
    output_dir: Path,
    max_visualizations: int,
    saved_count: int,
) -> int:
    video = batch["video"]
    image = batch["image"]
    labels = batch["label"]
    paths = batch["path"]
    if not isinstance(labels, list):
        labels = [labels]
    if not isinstance(paths, list):
        paths = [paths]

    mask = sample_mask_from_model(probe.encoder.patch_embed, video, cfg, video.device)
    out = probe(video, mask)
    patches = patchify_video(
        video,
        tubelet_size=probe.encoder.tubelet_size,
        patch_size=probe.encoder.patch_size,
    )
    masked_video = _unpatchify_video(
        _mask_tokens_with_value(patches, mask, 0.0),
        channels=video.size(1),
        frames=video.size(2),
        height=video.size(3),
        width=video.size(4),
        tubelet_size=probe.encoder.tubelet_size,
        patch_size=probe.encoder.patch_size,
    )
    reconstructed_video = _unpatchify_video(
        _scatter_masked_values(patches, mask, out.prediction),
        channels=video.size(1),
        frames=video.size(2),
        height=video.size(3),
        width=video.size(4),
        tubelet_size=probe.encoder.tubelet_size,
        patch_size=probe.encoder.patch_size,
    )

    for batch_idx in range(video.size(0)):
        if saved_count >= max_visualizations:
            break
        original = image[batch_idx]
        masked = masked_video[batch_idx, :, 0].cpu()
        reconstructed = reconstructed_video[batch_idx, :, 0].cpu()
        panel = torch.cat([original[batch_idx].cpu() if original.ndim == 4 else original.cpu(), masked, reconstructed], dim=2)
        label = str(labels[batch_idx]).replace("/", "_")
        stem = Path(str(paths[batch_idx])).stem
        save_image(panel, output_dir / f"{saved_count:03d}_{label}_{stem}.png")
        saved_count += 1
    return saved_count


def _evaluate_probe(
    probe: FrozenStudentPixelProbe,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, float | int | str]:
    probe.eval()
    total_examples = 0
    masked_patch_mse = 0.0
    masked_patch_l1 = 0.0
    saved_visualizations = 0
    max_visualizations = int(cfg["probe"].get("num_visualizations", 8))

    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            mask = sample_mask_from_model(probe.encoder.patch_embed, video, cfg, device)
            out = probe(video, mask)
            batch_size = video.size(0)
            total_examples += batch_size
            masked_patch_mse += F.mse_loss(out.prediction, out.target).item() * batch_size
            masked_patch_l1 += F.l1_loss(out.prediction, out.target).item() * batch_size

            if output_dir is not None and saved_visualizations < max_visualizations:
                saved_visualizations = _save_probe_visualizations(
                    probe=probe,
                    batch={k: (v.to(device) if isinstance(v, torch.Tensor) and k == "video" else v) for k, v in batch.items()},
                    cfg=cfg,
                    output_dir=output_dir,
                    max_visualizations=max_visualizations,
                    saved_count=saved_visualizations,
                )

    if total_examples == 0:
        raise RuntimeError("No probe evaluation samples were processed.")

    metrics = {
        "num_examples": total_examples,
        "masked_patch_mse": masked_patch_mse / total_examples,
        "masked_patch_l1": masked_patch_l1 / total_examples,
    }
    if output_dir is not None:
        metrics["output_dir"] = str(output_dir)
    return metrics


def _save_probe_checkpoint(probe: FrozenStudentPixelProbe, checkpoint_path: Path, step: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), checkpoint_path)
    print(f"probe checkpoint saved step={step} path={checkpoint_path}")


def run(cfg: dict) -> None:
    device = resolve_device()
    probe_cfg = cfg["probe"]
    output_dir = Path(probe_cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher, _ = build_teacher_from_cfg(cfg, device)
    student = build_student_from_cfg(cfg, teacher, device)
    student.load_state_dict(torch.load(probe_cfg["student_checkpoint"], map_location=device))

    probe = FrozenStudentPixelProbe(
        student_encoder=student.student,
        decoder_dim=int(probe_cfg["decoder_dim"]),
        decoder_depth=int(probe_cfg["decoder_depth"]),
        decoder_heads=int(probe_cfg["decoder_heads"]),
    ).to(device)

    train_loader = _build_loader(
        root=probe_cfg["train_root"],
        cfg=cfg,
        max_samples=probe_cfg.get("max_train_samples"),
        shuffle=True,
    )
    val_loader = _build_loader(
        root=probe_cfg["val_root"],
        cfg=cfg,
        max_samples=probe_cfg.get("max_val_samples"),
        shuffle=False,
    )
    batch_size = int(probe_cfg.get("batch_size", 1))
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    steps_per_epoch = math.ceil(train_dataset_size / batch_size)
    epochs = int(probe_cfg.get("epochs", 1))
    max_steps_cfg = probe_cfg.get("max_steps")
    if max_steps_cfg is None:
        max_steps = steps_per_epoch * epochs
    else:
        max_steps = int(max_steps_cfg)

    print(
        "probe dataset "
        f"train_samples={train_dataset_size} "
        f"val_samples={val_dataset_size} "
        f"batch_size={batch_size} "
        f"steps_per_epoch={steps_per_epoch} "
        f"epochs={epochs} "
        f"max_steps={max_steps}"
    )

    wandb_run = init_wandb_run(cfg, job_type="student-pixel-probe")
    optimizer = torch.optim.AdamW(
        probe.decoder.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = CosineScheduler(
        optimizer=optimizer,
        total_steps=max_steps,
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
        start_lr=float(cfg["optimizer"]["start_lr"]),
        peak_lr=float(cfg["optimizer"]["lr"]),
        final_lr=float(cfg["optimizer"]["final_lr"]),
        start_weight_decay=float(cfg["optimizer"]["start_weight_decay"]),
        end_weight_decay=float(cfg["optimizer"]["end_weight_decay"]),
    )
    train_iter = _iter_forever(train_loader)
    checkpoint_path = Path(probe_cfg["checkpoint_path"]).expanduser()
    checkpoint_every_steps = int(probe_cfg.get("checkpoint_every_steps", 50))
    eval_every_steps = int(probe_cfg.get("eval_every_steps", 50))
    teacher_arch = str(cfg["model"]["architecture"])
    student_arch = str(cfg.get("student_model", {}).get("architecture", cfg["model"]["architecture"]))
    probe_metadata = {
        "model/teacher_architecture": teacher_arch,
        "model/student_architecture": student_arch,
        "probe/batch_size": batch_size,
        "probe/decoder_dim": int(probe_cfg["decoder_dim"]),
        "probe/decoder_depth": int(probe_cfg["decoder_depth"]),
        "probe/decoder_heads": int(probe_cfg["decoder_heads"]),
        "probe/student_checkpoint": str(probe_cfg["student_checkpoint"]),
        "probe/checkpoint_path": str(checkpoint_path),
        "probe/output_dir": str(output_dir),
        "probe/train_samples": train_dataset_size,
        "probe/val_samples": val_dataset_size,
        "probe/steps_per_epoch": steps_per_epoch,
        "probe/epochs": epochs,
        "probe/max_steps": max_steps,
    }
    log_wandb_metrics(wandb_run, probe_metadata)

    probe.train()
    for step in range(1, max_steps + 1):
        batch = next(train_iter)
        video = batch["video"].to(device)
        lr, wd = scheduler.step(step - 1)
        mask = sample_mask_from_model(probe.encoder.patch_embed, video, cfg, device)
        out = probe(video, mask)
        loss = F.mse_loss(out.prediction, out.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.decoder.parameters(), cfg["optimizer"]["clip_grad"])
        optimizer.step()

        print(f"probe step={step} loss={loss.item():.6f} lr={lr:.7f} wd={wd:.4f}")
        log_wandb_metrics(
            wandb_run,
            {
                "train/step": step,
                "train/loss": float(loss.item()),
                "train/lr": float(lr),
                "train/weight_decay": float(wd),
            },
        )

        if checkpoint_every_steps > 0 and step % checkpoint_every_steps == 0:
            _save_probe_checkpoint(probe, checkpoint_path, step)

        if eval_every_steps > 0 and step % eval_every_steps == 0:
            val_metrics = _evaluate_probe(probe, val_loader, cfg, device)
            print(
                "probe eval "
                f"step={step} masked_patch_mse={val_metrics['masked_patch_mse']:.6f} "
                f"masked_patch_l1={val_metrics['masked_patch_l1']:.6f}"
            )
            log_wandb_metrics(
                wandb_run,
                {
                    "train/step": step,
                    "eval/masked_patch_mse": float(val_metrics["masked_patch_mse"]),
                    "eval/masked_patch_l1": float(val_metrics["masked_patch_l1"]),
                },
            )

    _save_probe_checkpoint(probe, checkpoint_path, max_steps)
    final_metrics = _evaluate_probe(probe, val_loader, cfg, device, output_dir=output_dir)
    summary = {
        "train/checkpoint_path": str(checkpoint_path),
        "train/student_checkpoint": str(probe_cfg["student_checkpoint"]),
        "train/final_step": max_steps,
        "model/teacher_architecture": teacher_arch,
        "model/student_architecture": student_arch,
        "probe/batch_size": batch_size,
        "probe/decoder_dim": int(probe_cfg["decoder_dim"]),
        "probe/decoder_depth": int(probe_cfg["decoder_depth"]),
        "probe/decoder_heads": int(probe_cfg["decoder_heads"]),
        "probe/train_root": str(probe_cfg["train_root"]),
        "probe/val_root": str(probe_cfg["val_root"]),
        "probe/output_dir": str(output_dir),
        "probe/train_samples": train_dataset_size,
        "probe/val_samples": val_dataset_size,
        "probe/steps_per_epoch": steps_per_epoch,
        "probe/epochs": epochs,
        "probe/max_steps": max_steps,
    }
    summary.update({f"eval/{key}": value for key, value in final_metrics.items()})
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_wandb_artifact(
        wandb_run,
        name=f"probe-{student_arch}-checkpoint",
        artifact_type="model",
        paths=[str(checkpoint_path)],
        metadata={
            "teacher_architecture": teacher_arch,
            "student_architecture": student_arch,
            "decoder_dim": int(probe_cfg["decoder_dim"]),
            "decoder_depth": int(probe_cfg["decoder_depth"]),
            "decoder_heads": int(probe_cfg["decoder_heads"]),
        },
    )
    metrics_path = output_dir / "metrics.json"
    artifact_paths = [str(metrics_path)]
    artifact_paths.extend(str(path) for path in sorted(output_dir.glob("*.png")))
    log_wandb_artifact(
        wandb_run,
        name=f"probe-{student_arch}-eval",
        artifact_type="evaluation",
        paths=artifact_paths,
        metadata={
            "teacher_architecture": teacher_arch,
            "student_architecture": student_arch,
            "num_visualizations": len(list(output_dir.glob('*.png'))),
        },
    )
    finish_wandb_run(wandb_run, summary=summary)

    print("student pixel probe complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
