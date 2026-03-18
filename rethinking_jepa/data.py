from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class VideoAugmentationConfig:
    input_size: int
    random_resize_aspect_ratio: tuple[float, float]
    random_resize_scale: tuple[float, float]


def random_resized_crop_video(
    video: torch.Tensor,
    config: VideoAugmentationConfig,
) -> torch.Tensor:
    _, _, height, width = video.shape
    area = height * width
    scale = random.uniform(*config.random_resize_scale)
    aspect_ratio = random.uniform(*config.random_resize_aspect_ratio)
    crop_h = int(round(math.sqrt(area * scale / aspect_ratio)))
    crop_w = int(round(math.sqrt(area * scale * aspect_ratio)))
    crop_h = max(1, min(height, crop_h))
    crop_w = max(1, min(width, crop_w))
    top = 0 if crop_h == height else random.randint(0, height - crop_h)
    left = 0 if crop_w == width else random.randint(0, width - crop_w)
    cropped = video[:, :, top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(
        cropped.permute(1, 0, 2, 3),
        size=(config.input_size, config.input_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(1, 0, 2, 3).contiguous()


class SyntheticVideoDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        num_samples: int,
        channels: int,
        frames: int,
        frame_step: int,
        height: int,
        width: int,
        augmentation: VideoAugmentationConfig | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.channels = channels
        self.frames = frames
        self.frame_step = frame_step
        self.height = height
        self.width = width
        self.augmentation = augmentation

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        full_frames = self.frames * self.frame_step
        t = torch.linspace(0, 1, full_frames)
        grid_y = torch.linspace(-1, 1, self.height)
        grid_x = torch.linspace(-1, 1, self.width)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        base = torch.stack(
            [
                torch.sin(2 * math.pi * (xx + t_i)) + torch.cos(2 * math.pi * yy)
                for t_i in t
            ],
            dim=0,
        )
        video = base.unsqueeze(0).repeat(self.channels, 1, 1, 1)
        noise = 0.05 * torch.randn_like(video)
        video = (video + noise)[:, :: self.frame_step].float()
        if self.augmentation is not None:
            video = random_resized_crop_video(video, self.augmentation)
        return video
