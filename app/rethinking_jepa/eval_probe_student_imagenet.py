from __future__ import annotations

import argparse
import csv
import json
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


def _build_loader(root: str | Path, cfg: dict) -> DataLoader:
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    dataset = ImageFolderRepeatedFrameDataset(
        root=root,
        input_size=int(data_cfg["input_size"]),
        frames=int(data_cfg["frames"]),
        resize_size=int(data_cfg.get("resize_size", data_cfg["input_size"])),
        max_samples=eval_cfg.get("max_val_samples"),
        sample_seed=int(eval_cfg.get("sample_seed", 0)),
        class_names=data_cfg.get("class_names"),
    )
    return DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
    )


def _normalize_per_sample(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    denom = (maxs - mins).clamp_min(1e-6)
    return (x - mins) / denom


def _latent_heatmaps(probe: FrozenStudentPixelProbe, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latents, _ = probe.encoder(video)
        _, grid = probe.encoder.patch_embed(video)
    token_norms = latents.norm(dim=-1).view(video.size(0), grid[0], grid[1], grid[2])
    heat = token_norms.mean(dim=1, keepdim=True)
    heat = F.interpolate(
        heat,
        size=(video.size(3), video.size(4)),
        mode="bilinear",
        align_corners=False,
    )
    heat = _normalize_per_sample(heat).repeat(1, 3, 1, 1)
    return heat, token_norms.cpu()


def _format_sample_metrics_table(sample_metrics: list[dict[str, object]]) -> str:
    headers = ["image name", "label", "mse", "l1", "latent mean norm"]
    rows = [
        [
            str(metric["image_name"]),
            str(metric["label"]),
            f"{float(metric['masked_patch_mse']):.6f}",
            f"{float(metric['masked_patch_l1']):.6f}",
            f"{float(metric['latent_mean_norm']):.6f}",
        ]
        for metric in sample_metrics
    ]
    table = [headers] + rows
    widths = [max(len(row[col]) for row in table) for col in range(len(headers))]

    def _fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    lines = [_fmt(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def run(cfg: dict) -> None:
    device = resolve_device()
    eval_cfg = cfg["eval"]
    output_dir = Path(eval_cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher, _ = build_teacher_from_cfg(cfg, device)
    student = build_student_from_cfg(cfg, teacher, device)
    student_ckpt = Path(eval_cfg.get("student_checkpoint", cfg.get("probe", {}).get("student_checkpoint", "")))
    if not student_ckpt.exists():
        raise FileNotFoundError(
            f"Student checkpoint not found at '{student_ckpt}'. "
            "Set eval.student_checkpoint to an existing file."
        )
    student.load_state_dict(torch.load(student_ckpt, map_location=device))

    probe = FrozenStudentPixelProbe(
        student_encoder=student.student,
        decoder_dim=int(eval_cfg["decoder_dim"]),
        decoder_depth=int(eval_cfg["decoder_depth"]),
        decoder_heads=int(eval_cfg["decoder_heads"]),
    ).to(device)
    probe_ckpt = Path(eval_cfg["probe_checkpoint"]).expanduser()
    if not probe_ckpt.exists():
        raise FileNotFoundError(
            f"Probe checkpoint not found at '{probe_ckpt}'. "
            "Set eval.probe_checkpoint to your saved probe weights."
        )
    probe.load_state_dict(torch.load(probe_ckpt, map_location=device), strict=True)
    probe.eval()

    val_loader = _build_loader(eval_cfg["val_root"], cfg)
    wandb_run = init_wandb_run(cfg, job_type="student-pixel-probe-eval")

    teacher_arch = str(cfg["model"]["architecture"])
    student_arch = str(cfg.get("student_model", {}).get("architecture", cfg["model"]["architecture"]))
    log_wandb_metrics(
        wandb_run,
        {
            "eval_probe/teacher_architecture": teacher_arch,
            "eval_probe/student_architecture": student_arch,
            "eval_probe/probe_checkpoint": str(probe_ckpt),
            "eval_probe/student_checkpoint": str(student_ckpt),
            "eval_probe/num_samples_requested": int(eval_cfg.get("max_val_samples", 0) or 0),
        },
    )

    sample_metrics: list[dict[str, object]] = []
    latent_norm_grids: dict[str, torch.Tensor] = {}
    total_examples = 0
    total_mse = 0.0
    total_l1 = 0.0
    saved_visualizations = 0
    max_visualizations = int(eval_cfg.get("num_visualizations", 20))

    with torch.no_grad():
        for batch in val_loader:
            video = batch["video"].to(device)
            image = batch["image"]
            labels = batch["label"]
            paths = batch["path"]
            if not isinstance(labels, list):
                labels = [labels]
            if not isinstance(paths, list):
                paths = [paths]

            mask = sample_mask_from_model(probe.encoder.patch_embed, video, cfg, device)
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
            heatmaps, token_norms = _latent_heatmaps(probe, video)

            for idx in range(video.size(0)):
                per_sample_mse = F.mse_loss(out.prediction[idx], out.target[idx]).item()
                per_sample_l1 = F.l1_loss(out.prediction[idx], out.target[idx]).item()
                total_examples += 1
                total_mse += per_sample_mse
                total_l1 += per_sample_l1

                stem = Path(str(paths[idx])).stem
                label = str(labels[idx]).replace("/", "_")
                sample_id = f"{total_examples:03d}_{label}_{stem}"
                grid = token_norms[idx]
                latent_norm_grids[sample_id] = grid
                sample_metrics.append(
                    {
                        "sample_id": sample_id,
                        "image_name": Path(str(paths[idx])).name,
                        "label": str(labels[idx]),
                        "path": str(paths[idx]),
                        "masked_patch_mse": per_sample_mse,
                        "masked_patch_l1": per_sample_l1,
                        "latent_mean_norm": float(grid.mean().item()),
                        "latent_std_norm": float(grid.std().item()),
                        "latent_max_norm": float(grid.max().item()),
                        "latent_min_norm": float(grid.min().item()),
                    }
                )

                if saved_visualizations < max_visualizations:
                    original = image[idx].cpu()
                    masked = masked_video[idx, :, 0].cpu()
                    reconstructed = reconstructed_video[idx, :, 0].cpu()
                    heatmap = heatmaps[idx].cpu()
                    panel = torch.cat([original, masked, reconstructed, heatmap], dim=2)
                    save_image(panel, output_dir / f"{sample_id}.png")
                    saved_visualizations += 1

    if total_examples == 0:
        raise RuntimeError("No validation examples were processed during probe evaluation.")

    summary = {
        "model/teacher_architecture": teacher_arch,
        "model/student_architecture": student_arch,
        "eval_probe/probe_checkpoint": str(probe_ckpt),
        "eval_probe/student_checkpoint": str(student_ckpt),
        "eval_probe/num_examples": total_examples,
        "eval_probe/avg_masked_patch_mse": total_mse / total_examples,
        "eval_probe/avg_masked_patch_l1": total_l1 / total_examples,
        "eval_probe/output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "sample_metrics.json").write_text(json.dumps(sample_metrics, indent=2), encoding="utf-8")
    with (output_dir / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "image_name",
                "label",
                "path",
                "masked_patch_mse",
                "masked_patch_l1",
                "latent_mean_norm",
                "latent_std_norm",
                "latent_max_norm",
                "latent_min_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_metrics)
    table_text = _format_sample_metrics_table(sample_metrics)
    (output_dir / "sample_metrics_table.txt").write_text(table_text + "\n", encoding="utf-8")
    torch.save(latent_norm_grids, output_dir / "latent_token_norms.pt")

    log_wandb_metrics(wandb_run, summary)
    artifact_paths = [
        str(output_dir / "summary.json"),
        str(output_dir / "sample_metrics.json"),
        str(output_dir / "sample_metrics.csv"),
        str(output_dir / "sample_metrics_table.txt"),
        str(output_dir / "latent_token_norms.pt"),
    ]
    artifact_paths.extend(str(path) for path in sorted(output_dir.glob("*.png")))
    log_wandb_artifact(
        wandb_run,
        name=f"probe-{student_arch}-eval-analysis",
        artifact_type="evaluation",
        paths=artifact_paths,
        metadata=summary,
    )
    finish_wandb_run(wandb_run, summary=summary)

    print("saved probe evaluation complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"sample_metrics_path: {output_dir / 'sample_metrics.json'}")
    print("")
    print(_format_sample_metrics_table(sample_metrics))
    print("")
    print(f"sample_metrics_table_path: {output_dir / 'sample_metrics_table.txt'}")
    print(f"latent_token_norms_path: {output_dir / 'latent_token_norms.pt'}")


def main(cfg: dict | None = None) -> None:
    if cfg is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        args = parser.parse_args()
        cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
