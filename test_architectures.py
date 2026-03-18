from __future__ import annotations

import argparse

import torch

from rethinking_jepa import (
    StudentModel,
    TeacherModel,
    VIT_ARCHITECTURES,
    load_config,
    resolve_model_config,
    sample_multi_block_mask,
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/teacher.yaml")
    parser.add_argument(
        "--architectures",
        nargs="*",
        default=sorted(VIT_ARCHITECTURES),
        choices=sorted(VIT_ARCHITECTURES),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    frames = cfg["data"]["frames"]
    size = cfg["data"]["input_size"]
    in_channels = cfg["model"].get("in_channels", 3)

    for architecture in args.architectures:
        model_cfg = resolve_model_config(
            {
                "architecture": architecture,
                "in_channels": in_channels,
                "tubelet_size": cfg["model"]["tubelet_size"],
                "patch_size": cfg["model"]["patch_size"],
            }
        )
        student_cfg = cfg.get("student", {})
        predictor_dim = student_cfg.get("predictor_dim", model_cfg["embed_dim"])
        predictor_depth = student_cfg.get("predictor_depth", 4)
        predictor_heads = student_cfg.get("predictor_heads", model_cfg["encoder_heads"])

        teacher = TeacherModel(**model_cfg, frames=frames, image_size=size).to(device)
        student = StudentModel(
            teacher=teacher,
            predictor_dim=predictor_dim,
            predictor_depth=predictor_depth,
            predictor_heads=predictor_heads,
        ).to(device)

        video = torch.randn(1, in_channels, frames, size, size, device=device)
        with torch.no_grad():
            _, grid = teacher.encoder.patch_embed(video)
            mask = sample_multi_block_mask(
                batch_size=1,
                grid_t=grid[0],
                grid_h=grid[1],
                grid_w=grid[2],
                short_spatial_scale=cfg["masking"]["short_spatial_mask_scale"],
                long_spatial_scale=cfg["masking"]["long_spatial_mask_scale"],
                temporal_scale=cfg["masking"]["temporal_mask_scale"],
                aspect_ratio_range=tuple(cfg["masking"]["mask_aspect_ratio"]),
                device=device,
            )
            teacher_out = teacher(video, mask)
            student_out = student(video, mask)

        print(
            f"{architecture}: "
            f"teacher_params={count_parameters(teacher):,} "
            f"student_params={count_parameters(student):,} "
            f"tokens={grid[0] * grid[1] * grid[2]} "
            f"teacher_pred={tuple(teacher_out.prediction.shape)} "
            f"student_pred={tuple(student_out.prediction.shape)}"
        )


if __name__ == "__main__":
    main()
