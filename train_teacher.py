from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from rethinking_jepa import (
    CosineScheduler,
    SyntheticVideoDataset,
    TeacherModel,
    VideoAugmentationConfig,
    load_config,
    resolve_model_config,
    sample_multi_block_mask,
)


def build_dataset(cfg: dict) -> SyntheticVideoDataset:
    return SyntheticVideoDataset(
        num_samples=cfg["data"]["num_samples"],
        channels=cfg["model"]["in_channels"],
        frames=cfg["data"]["frames"],
        frame_step=cfg["data"]["frame_step"],
        height=cfg["data"]["image_size"],
        width=cfg["data"]["image_size"],
        augmentation=VideoAugmentationConfig(
            input_size=cfg["data"]["input_size"],
            random_resize_aspect_ratio=tuple(cfg["augmentation"]["random_resize_aspect_ratio"]),
            random_resize_scale=tuple(cfg["augmentation"]["random_resize_scale"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = resolve_model_config(cfg["model"])

    dataset = build_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg["train"]["device_batch_size"], shuffle=True)
    model = TeacherModel(
        **model_cfg,
        frames=cfg["data"]["frames"],
        image_size=cfg["data"]["input_size"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["start_lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = CosineScheduler(
        optimizer=optimizer,
        total_steps=cfg["train"]["max_steps"],
        warmup_steps=cfg["optimizer"]["warmup_steps"],
        start_lr=cfg["optimizer"]["start_lr"],
        peak_lr=cfg["optimizer"]["lr"],
        final_lr=cfg["optimizer"]["final_lr"],
        start_weight_decay=cfg["optimizer"]["start_weight_decay"],
        end_weight_decay=cfg["optimizer"]["end_weight_decay"],
    )
    criterion = nn.MSELoss()

    model.train()
    for step, video in enumerate(loader, start=1):
        video = video.to(device)
        with torch.no_grad():
            _, grid = model.encoder.patch_embed(video)
        lr, wd = scheduler.step(step - 1)
        mask = sample_multi_block_mask(
            batch_size=video.size(0),
            grid_t=grid[0],
            grid_h=grid[1],
            grid_w=grid[2],
            short_spatial_scale=cfg["masking"]["short_spatial_mask_scale"],
            long_spatial_scale=cfg["masking"]["long_spatial_mask_scale"],
            temporal_scale=cfg["masking"]["temporal_mask_scale"],
            aspect_ratio_range=tuple(cfg["masking"]["mask_aspect_ratio"]),
            device=device,
        )
        out = model(video, mask)
        loss = criterion(out.prediction, out.target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["clip_grad"])
        optimizer.step()
        print(
            f"teacher step={step} loss={loss.item():.6f} "
            f"lr={lr:.7f} wd={wd:.4f}"
        )
        if step >= cfg["train"]["max_steps"]:
            break

    torch.save(model.state_dict(), cfg["train"]["checkpoint_path"])


if __name__ == "__main__":
    main()
