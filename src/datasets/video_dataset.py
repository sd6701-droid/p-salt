from __future__ import annotations

import csv
import math
import os
import random
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

DEFAULT_KINETICS700_TRAIN_ANNOTATION_URL = "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv"
VIDEO_FILE_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
SQUASHFS_FILE_EXTENSIONS = (".sqf", ".sqfs", ".squashfs")
logger = getLogger(__name__)


def _resolve_squashfs_tool(tool_name: str, explicit_path: str | Path | None = None) -> str | None:
    candidates: list[Path | str] = []
    env_name = f"{tool_name.upper()}_BIN"

    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get(env_name)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    tools_dir = os.environ.get("SQUASHFS_TOOLS_DIR")
    if tools_dir:
        candidates.append(Path(tools_dir).expanduser() / tool_name)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())

    resolved = shutil.which(tool_name)
    if resolved:
        return resolved

    for candidate in (
        Path("/usr/bin") / tool_name,
        Path("/usr/sbin") / tool_name,
        Path("/usr/local/bin") / tool_name,
        Path("/opt/homebrew/bin") / tool_name,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    return None


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


def _read_video_frames(path: str | Path) -> torch.Tensor:
    path = str(path)
    try:
        from torchvision.io import read_video
    except ImportError:
        read_video = None
    except Exception:
        read_video = None

    if read_video is not None:
        try:
            video, _, _ = read_video(path, pts_unit="sec")
            if video.ndim == 4 and video.size(0) > 0:
                return video
        except Exception:
            pass

    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "Local video loading requires either torchvision.io.read_video or the 'av' package. "
            "Install PyAV with `pip install av` if your torchvision build lacks video decoding."
        ) from exc

    frames: list[torch.Tensor] = []
    with av.open(path) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in '{path}'")
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))

    if not frames:
        raise ValueError(f"No decoded frames found in '{path}'")
    return torch.stack(frames, dim=0)


def _sample_frame_indices(total_frames: int, frames: int, frame_step: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Encountered a video with zero frames")

    total_needed = frames * frame_step
    if total_frames >= total_needed:
        max_start = total_frames - total_needed
        start = 0 if max_start == 0 else random.randint(0, max_start)
        return list(range(start, start + total_needed, frame_step))

    return torch.linspace(0, total_frames - 1, frames).round().long().tolist()


def _estimate_video_frame_count_from_av(container: Any, stream: Any) -> int | None:
    try:
        import av
    except ImportError:
        return None

    try:
        if getattr(stream, "frames", 0):
            count = int(stream.frames)
            if count > 0:
                return count

        fps = float(stream.average_rate) if stream.average_rate is not None else None
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
        else:
            duration_seconds = None

        if fps is not None and duration_seconds is not None:
            estimated = int(round(fps * duration_seconds))
            if estimated > 0:
                return estimated
    except Exception:
        return None

    return None


def _read_video_clip_with_av(
    path: str | Path,
    *,
    frames: int,
    frame_step: int,
) -> torch.Tensor | None:
    try:
        import av
    except ImportError:
        return None

    selected: list[torch.Tensor] = []
    try:
        with av.open(str(path)) as container:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if video_stream is None:
                raise ValueError(f"No video stream found in '{path}'")
            total_frames = _estimate_video_frame_count_from_av(container, video_stream)
            if total_frames is None:
                return None

            target_indices = _sample_frame_indices(total_frames, frames, frame_step)
            remaining = dict.fromkeys(target_indices, 0)
            for index in target_indices:
                remaining[index] = remaining.get(index, 0) + 1

            max_target_index = max(target_indices)
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index > max_target_index and not remaining:
                    break

                repeat_count = remaining.pop(frame_index, 0)
                if repeat_count <= 0:
                    continue

                frame_tensor = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                selected.append(frame_tensor)
                for _ in range(repeat_count - 1):
                    selected.append(frame_tensor.clone())

                if len(selected) >= len(target_indices):
                    break
    except Exception:
        return None

    if len(selected) != len(target_indices):
        return None

    return torch.stack(selected, dim=0)


def _read_video_clip(
    path: str | Path,
    *,
    frames: int,
    frame_step: int,
) -> torch.Tensor:
    clip = _read_video_clip_with_av(path, frames=frames, frame_step=frame_step)
    if clip is not None:
        return clip

    video = _read_video_frames(path)
    indices = _sample_frame_indices(int(video.size(0)), frames, frame_step)
    return video.index_select(0, torch.tensor(indices, dtype=torch.long))


def _read_video_clip_with_decord(
    path: str | Path,
    *,
    frames: int,
    frame_step: int,
) -> torch.Tensor | None:
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return None

    try:
        vr = VideoReader(str(path), num_threads=-1, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            return None
        indices = _sample_frame_indices(total_frames, frames, frame_step)
        batch = vr.get_batch(indices).asnumpy()
        return torch.from_numpy(batch)
    except Exception:
        return None


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


class VideoFileDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        video_paths: list[Path],
        channels: int,
        frames: int,
        frame_step: int,
        image_size: int,
        augmentation: VideoAugmentationConfig | None = None,
        skip_decode_errors: bool = False,
        max_decode_attempts: int = 16,
        log_decode_warnings: bool = True,
    ) -> None:
        if not video_paths:
            raise ValueError("VideoFileDataset requires at least one video path")
        self.video_paths = video_paths
        self.channels = channels
        self.frames = frames
        self.frame_step = frame_step
        self.image_size = image_size
        self.augmentation = augmentation
        self.skip_decode_errors = skip_decode_errors
        self.max_decode_attempts = max(1, int(max_decode_attempts))
        self.log_decode_warnings = log_decode_warnings
        self._decode_warning_count = 0

    def __len__(self) -> int:
        return len(self.video_paths)

    def _sample_clip(self, video: torch.Tensor) -> torch.Tensor:
        video = video.float() / 255.0
        return video.permute(3, 0, 1, 2).contiguous()

    def _load_clip(self, index: int) -> torch.Tensor:
        path = self.video_paths[index]
        sampled_frames = _read_video_clip(path, frames=self.frames, frame_step=self.frame_step)
        clip = self._sample_clip(sampled_frames)
        if self.augmentation is not None:
            clip = random_resized_crop_video(clip, self.augmentation)
        else:
            clip = F.interpolate(
                clip.permute(1, 0, 2, 3),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3).contiguous()
        if clip.size(0) != self.channels:
            if clip.size(0) == 1 and self.channels == 3:
                clip = clip.expand(3, -1, -1, -1)
            else:
                raise ValueError(
                    f"Expected {self.channels} channels but decoded {clip.size(0)} from {path}"
                )
        return clip

    def _warn_decode_error(self, *, requested_index: int, fallback_index: int, exc: Exception) -> None:
        if not self.log_decode_warnings:
            return
        self._decode_warning_count += 1
        if self._decode_warning_count > 5 and self._decode_warning_count % 25 != 0:
            return
        print(
            "warning: skipping undecodable local video "
            f"index={requested_index} fallback_index={fallback_index} "
            f"error={type(exc).__name__}"
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        if not self.skip_decode_errors:
            return self._load_clip(index)

        attempts = min(self.max_decode_attempts, len(self.video_paths))
        last_error: Exception | None = None
        for attempt in range(attempts):
            candidate_index = (index + attempt) % len(self.video_paths)
            try:
                return self._load_clip(candidate_index)
            except Exception as exc:
                last_error = exc
                self._warn_decode_error(
                    requested_index=index,
                    fallback_index=candidate_index,
                    exc=exc,
                )

        raise RuntimeError(
            "Failed to decode a usable local video clip after "
            f"{attempts} attempts starting at dataset index {index}."
        ) from last_error


class SquashFSVideoDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        archive_path: str | Path,
        channels: int,
        frames: int,
        frame_step: int,
        image_size: int,
        augmentation: VideoAugmentationConfig | None = None,
        max_samples: int | None = None,
        sample_seed: int = 0,
        class_names: list[str] | None = None,
        class_fraction: float | None = None,
        max_samples_per_class: int | None = None,
        cache_dir: str | Path | None = None,
        unsquashfs_path: str | Path | None = None,
        sqfscat_path: str | Path | None = None,
    ) -> None:
        archive = Path(archive_path).expanduser()
        if archive.suffix.lower() not in SQUASHFS_FILE_EXTENSIONS:
            raise ValueError(
                f"Expected a SquashFS archive with one of {SQUASHFS_FILE_EXTENSIONS}, got '{archive}'"
            )
        if not archive.is_file():
            raise FileNotFoundError(f"SquashFS archive not found at '{archive}'")

        self.archive_path = archive
        self.channels = channels
        self.frames = frames
        self.frame_step = frame_step
        self.image_size = image_size
        # do we need any augmentations 
        self.augmentation = augmentation
        self.class_names = list(class_names) if class_names is not None else None
        self.class_fraction = class_fraction
        self.max_samples_per_class = (
            int(max_samples_per_class) if max_samples_per_class is not None else None
        )
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self.unsquashfs_path = _resolve_squashfs_tool("unsquashfs", unsquashfs_path)   #used tool to get the encoded videos (not need but we may need since scratch has limited size ?)
        self.sqfscat_path = _resolve_squashfs_tool("sqfscat", sqfscat_path)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.archive_entries = self._list_video_entries()
        if not self.archive_entries:
            raise ValueError(
                f"No video files with extensions {VIDEO_FILE_EXTENSIONS} were found inside '{archive}'"
            )

        if self.class_names is not None:
            self.archive_entries = self._filter_entries_by_class(self.archive_entries, sample_seed)
            if not self.archive_entries:
                raise ValueError(
                    f"No videos found for class_names={self.class_names} inside '{archive}'"
                )

        if max_samples is not None and len(self.archive_entries) > max_samples:
            rng = random.Random(sample_seed)
            archive_entries = list(self.archive_entries)
            rng.shuffle(archive_entries)
            self.archive_entries = archive_entries[: int(max_samples)]

 # it lists all video files inside a SquashFS archive (and sorted them for further decod)
    def _list_video_entries(self) -> list[str]:
        if self.unsquashfs_path is None:
            raise RuntimeError(
                "Listing videos inside a SquashFS archive requires 'unsquashfs'. "
                "Install or load squashfs-tools, set UNSQUASHFS_BIN=/full/path/to/unsquashfs, "
                "or set data.unsquashfs_path in the config."
            )

        try:
            result = subprocess.run(
                [self.unsquashfs_path, "-no-progress", "-d", "", "-lc", str(self.archive_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            raise RuntimeError(
                f"Failed to list SquashFS archive '{self.archive_path}': {stderr or exc}"
            ) from exc

        entries: list[str] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = line.removeprefix("squashfs-root/").lstrip("/")
            if Path(normalized).suffix.lower() not in VIDEO_FILE_EXTENSIONS:
                continue
            entries.append(normalized)
        return sorted(set(entries))

    @staticmethod
    def _class_name_from_entry(entry: str) -> str:
        parent = Path(entry).parent.name
        if not parent:
            raise ValueError(f"Could not infer class name from SquashFS entry '{entry}'")
        return parent

    def _filter_entries_by_class(self, entries: list[str], sample_seed: int) -> list[str]:
        requested_classes = list(dict.fromkeys(self.class_names or []))
        allowed = set(requested_classes)
        by_class: dict[str, list[str]] = {class_name: [] for class_name in requested_classes}

        for entry in entries:
            class_name = self._class_name_from_entry(entry)
            if class_name in allowed:
                by_class[class_name].append(entry)

        missing = [class_name for class_name, class_entries in by_class.items() if not class_entries]
        if missing:
            raise ValueError(
                "The following classes were not found inside the SquashFS archive: "
                + ", ".join(missing)
            )

        base_rng = random.Random(sample_seed)
        selected: list[str] = []
        for class_name in requested_classes:
            class_entries = sorted(by_class[class_name])
            rng = random.Random(base_rng.randint(0, 10**9))
            rng.shuffle(class_entries)

            if self.class_fraction is not None:
                fraction = float(self.class_fraction)
                if not 0.0 < fraction <= 1.0:
                    raise ValueError(f"class_fraction must be in (0, 1], got {fraction}")
                keep = max(1, math.ceil(len(class_entries) * fraction))
                class_entries = class_entries[:keep]

            if self.max_samples_per_class is not None:
                class_entries = class_entries[: self.max_samples_per_class]

            selected.extend(class_entries)

        return selected

    def __len__(self) -> int:
        return len(self.archive_entries)

    def _sample_clip(self, video: torch.Tensor) -> torch.Tensor:
        video = video.float() / 255.0
        return video.permute(3, 0, 1, 2).contiguous()

    def _extract_archive_entry(self, entry: str, output_dir: Path) -> Path:
        cached_path: Path | None = None
        if self.cache_dir is not None:
            cached_path = self.cache_dir / entry
            if cached_path.exists():
                return cached_path
            cached_path.parent.mkdir(parents=True, exist_ok=True)

        if self.sqfscat_path is not None:
            # Preserve the archive's relative folder structure so downstream code can
            # still infer class names from parent directories after extraction.
            target_path = cached_path if cached_path is not None else output_dir / entry
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            try:
                with temp_path.open("wb") as handle:
                    subprocess.run(
                        [self.sqfscat_path, str(self.archive_path), entry],
                        check=True,
                        stdout=handle,
                        stderr=subprocess.PIPE,
                    )
            except subprocess.CalledProcessError as exc:
                temp_path.unlink(missing_ok=True)
                stderr = exc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Failed to extract '{entry}' from SquashFS archive '{self.archive_path}': "
                    f"{stderr or exc}"
                ) from exc
            temp_path.replace(target_path)
            return target_path

        if self.unsquashfs_path is None:
            raise RuntimeError(
                "Reading videos from a SquashFS archive requires either 'sqfscat' or "
                "'unsquashfs'. Install or load squashfs-tools, set SQFSCAT_BIN/UNSQUASHFS_BIN, "
                "or set data.sqfscat_path/data.unsquashfs_path in the config."
            )

        try:
            subprocess.run(
                [
                    self.unsquashfs_path,
                    "-f",
                    "-no-progress",
                    "-no-xattrs",
                    "-d",
                    str(output_dir),
                    str(self.archive_path),
                    entry,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Failed to extract '{entry}' from SquashFS archive '{self.archive_path}': "
                f"{stderr or exc}"
            ) from exc

        extracted_path = output_dir / entry
        if extracted_path.exists():
            if cached_path is None:
                return extracted_path
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted_path), str(cached_path))
            return cached_path

        fallback = next(output_dir.rglob(Path(entry).name), None)
        if fallback is None:
            raise RuntimeError(
                f"Extracted '{entry}' from '{self.archive_path}', but could not find the output file "
                f"under '{output_dir}'."
            )
        if cached_path is None:
            return fallback
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fallback), str(cached_path))
        return cached_path

    def extract_entry(self, entry: str, output_dir: str | Path) -> Path:
        destination_root = Path(output_dir).expanduser().resolve()
        destination_root.mkdir(parents=True, exist_ok=True)
        return self._extract_archive_entry(entry, destination_root)

    def __getitem__(self, index: int) -> torch.Tensor:
        archive_entry = self.archive_entries[index]
        if self.cache_dir is not None:
            extracted_path = self._extract_archive_entry(archive_entry, self.cache_dir)
            sampled_frames = _read_video_clip(
                extracted_path,
                frames=self.frames,
                frame_step=self.frame_step,
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_path = self._extract_archive_entry(archive_entry, Path(temp_dir))
                sampled_frames = _read_video_clip(
                    extracted_path,
                    frames=self.frames,
                    frame_step=self.frame_step,
                )
        clip = self._sample_clip(sampled_frames)
        if self.augmentation is not None:
            clip = random_resized_crop_video(clip, self.augmentation)
        else:
            clip = F.interpolate(
                clip.permute(1, 0, 2, 3),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3).contiguous()
        if clip.size(0) != self.channels:
            if clip.size(0) == 1 and self.channels == 3:
                clip = clip.expand(3, -1, -1, -1)
            else:
                raise ValueError(
                    f"Expected {self.channels} channels but decoded {clip.size(0)} from "
                    f"archive entry '{archive_entry}'"
                )
        return clip


class HuggingFaceVideoDataset(IterableDataset[torch.Tensor]):
    def __init__(
        self,
        repo_id: str,
        split: str,
        channels: int,
        frames: int,
        frame_step: int,
        image_size: int,
        augmentation: VideoAugmentationConfig | None = None,
        config_name: str | None = None,
        video_column: str = "video",
        max_samples: int | None = None,
        sample_seed: int = 0,
        shuffle_buffer_size: int = 256,
        decode_threads: int = 0,
        class_names: list[str] | None = None,
        class_fraction: float | None = None,
        annotation_csv_url: str | None = None,
        annotation_csv_path: str | None = None,
        skip_decode_errors: bool = True,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.channels = channels
        self.frames = frames
        self.frame_step = frame_step
        self.image_size = image_size
        self.augmentation = augmentation
        self.config_name = config_name
        self.video_column = video_column
        self.max_samples = max_samples
        self.sample_seed = sample_seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.decode_threads = decode_threads
        self.class_names = list(class_names) if class_names is not None else None
        self.class_fraction = class_fraction
        self.annotation_csv_url = annotation_csv_url
        self.annotation_csv_path = annotation_csv_path
        self.skip_decode_errors = skip_decode_errors
        self._selected_filenames: set[str] | None = None

    def _decode_hf_clip(self, video_decoder: Any) -> torch.Tensor:
        total_frames = self.frames * self.frame_step
        frame_batch = video_decoder.get_frames_in_range(0, total_frames, self.frame_step)
        clip = frame_batch.data
        if clip.ndim != 4 or clip.size(0) == 0:
            raise ValueError("Decoded Hugging Face video clip has no usable frames")
        clip = clip.float() / 255.0
        if clip.size(0) < self.frames:
            indices = torch.linspace(0, clip.size(0) - 1, self.frames).round().long()
            clip = clip.index_select(0, indices)
        else:
            clip = clip[: self.frames]
        return clip.permute(1, 0, 2, 3).contiguous()

    def _resolve_annotation_source(self) -> str:
        if self.annotation_csv_path is not None:
            return self.annotation_csv_path
        if self.annotation_csv_url is not None:
            return self.annotation_csv_url
        if "kinetics-700" in self.repo_id.lower():
            return DEFAULT_KINETICS700_TRAIN_ANNOTATION_URL
        raise ValueError(
            "Filtering Hugging Face videos by Kinetics class requires annotation_csv_path or annotation_csv_url"
        )

    def _read_annotation_rows(self) -> list[dict[str, str]]:
        source = self._resolve_annotation_source()
        if source.startswith(("http://", "https://")):
            text = urllib.request.urlopen(source, timeout=60).read().decode("utf-8", errors="replace")
        else:
            text = Path(source).expanduser().read_text(encoding="utf-8")
        return list(csv.DictReader(text.splitlines()))

    @staticmethod
    def _format_kinetics_filename(youtube_id: str, time_start: str, time_end: str) -> str:
        start = int(float(time_start))
        end = int(float(time_end))
        return f"{youtube_id}_{start:06d}_{end:06d}.mp4"

    @staticmethod
    def _extract_video_filename(video_value: Any) -> str | None:
        candidates: list[str] = []
        if isinstance(video_value, dict):
            candidates.extend(str(video_value.get(key)) for key in ("path", "filename", "src") if video_value.get(key))
        for attr in ("path", "filename", "src", "source"):
            value = getattr(video_value, attr, None)
            if value:
                candidates.append(str(value))
        for raw_value in candidates:
            value = raw_value.split("::", 1)[0]
            name = Path(value).name
            if name:
                return name
        return None

    def _resolve_selected_filenames(self) -> set[str] | None:
        if self.class_names is None:
            return None
        if self._selected_filenames is not None:
            return self._selected_filenames
        if self.class_fraction is None:
            class_fraction = 1.0
        else:
            class_fraction = float(self.class_fraction)
        if not 0.0 < class_fraction <= 1.0:
            raise ValueError(f"class_fraction must be in (0, 1], got {class_fraction}")

        rows = self._read_annotation_rows()
        by_class: dict[str, list[str]] = {name: [] for name in self.class_names}
        for row in rows:
            label = row.get("label")
            if label in by_class:
                filename = self._format_kinetics_filename(
                    youtube_id=row["youtube_id"],
                    time_start=row["time_start"],
                    time_end=row["time_end"],
                )
                by_class[label].append(filename)

        missing = [name for name, filenames in by_class.items() if not filenames]
        if missing:
            raise ValueError(f"Could not find Kinetics annotations for classes: {', '.join(missing)}")

        selected: set[str] = set()
        base_rng = random.Random(self.sample_seed)
        for class_name in self.class_names:
            filenames = sorted(by_class[class_name])
            rng = random.Random(base_rng.randint(0, 10**9))
            rng.shuffle(filenames)
            keep = max(1, math.ceil(len(filenames) * class_fraction))
            selected.update(filenames[:keep])
        self._selected_filenames = selected
        return selected

    def __iter__(self):
        try:
            from datasets import Video, load_dataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Hugging Face streaming requires the 'datasets' package. "
                "Install requirements.txt to use data.source='huggingface'."
            ) from exc

        dataset = load_dataset(
            self.repo_id,
            self.config_name,
            split=self.split,
            streaming=True,
        )
        if self.shuffle_buffer_size > 1:
            dataset = dataset.shuffle(
                seed=self.sample_seed,
                buffer_size=self.shuffle_buffer_size,
            )
        raw_dataset = dataset.cast_column(self.video_column, Video(decode=False))
        video_feature = Video(decode=True)

        selected_filenames = self._resolve_selected_filenames()
        yielded = 0
        skipped_decode_errors = 0
        for raw_example in raw_dataset:
            if selected_filenames is not None:
                filename = self._extract_video_filename(raw_example[self.video_column])
                if filename is None or filename not in selected_filenames:
                    continue

            try:
                decoded_video = video_feature.decode_example(raw_example[self.video_column])
                clip = self._decode_hf_clip(decoded_video)
            except Exception as exc:
                if not self.skip_decode_errors:
                    raise
                skipped_decode_errors += 1
                if skipped_decode_errors <= 5 or skipped_decode_errors % 25 == 0:
                    filename = self._extract_video_filename(raw_example[self.video_column]) or "<unknown>"
                    print(
                        f"warning: skipping undecodable video {filename} "
                        f"from {self.repo_id}: {type(exc).__name__}: {exc}"
                    )
                continue

            if self.augmentation is not None:
                clip = random_resized_crop_video(clip, self.augmentation)
            else:
                clip = F.interpolate(
                    clip.permute(1, 0, 2, 3),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).permute(1, 0, 2, 3).contiguous()
            if clip.size(0) != self.channels:
                if clip.size(0) == 1 and self.channels == 3:
                    clip = clip.expand(3, -1, -1, -1)
                else:
                    raise ValueError(
                        f"Expected {self.channels} channels but decoded {clip.size(0)} "
                        f"channels from Hugging Face dataset {self.repo_id}"
                    )
            yield clip
            yielded += 1
            if self.max_samples is not None and yielded >= self.max_samples:
                break

        if yielded == 0:
            raise RuntimeError(
                "No clips were yielded from the Hugging Face dataset. "
                "Check your class filter, annotation mapping, and video decoding environment."
            )


class VideoDataset(Dataset[tuple[list[torch.Tensor], int, list[Any]]]):
    """V-JEPA-style dataset that returns (clips, label, clip_indices)."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,
    ) -> None:
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.sample_weights: list[float] | None = None
        self.num_samples_per_dataset: list[int] = []

        if isinstance(data_paths, (str, Path)):
            data_paths = [str(data_paths)]
        self.data_paths = list(data_paths)
        self.dataset_fpcs = (
            [int(frames_per_clip) for _ in self.data_paths]
            if dataset_fpcs is None
            else [int(v) for v in dataset_fpcs]
        )
        if len(self.dataset_fpcs) != len(self.data_paths):
            raise ValueError("dataset_fpcs must match the number of data_paths")
        if self.num_clips < 1:
            raise ValueError("num_clips must be >= 1")

        samples: list[str] = []
        labels: list[int] = []
        for path in self.data_paths:
            p = Path(path).expanduser()
            rows: list[tuple[str, int]] = []
            if p.suffix.lower() == ".csv":
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    sample_path = parts[0]
                    label = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
                    rows.append((sample_path, label))
            else:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        rows.append((line, 0))
            self.num_samples_per_dataset.append(len(rows))
            for sample_path, label in rows:
                samples.append(sample_path)
                labels.append(label)

        if not samples:
            raise ValueError("VideoDataset received no samples from data_paths")
        if self.datasets_weights is not None:
            if len(self.datasets_weights) != len(self.num_samples_per_dataset):
                raise ValueError("datasets_weights length must match data_paths length")
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset, strict=True):
                self.sample_weights.extend([float(dw) / max(ns, 1)] * ns)
        self.samples = samples
        self.labels = labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        clip = _read_video_clip(sample, frames=self.dataset_fpcs[0], frame_step=self.frame_step)
        video = clip.float() / 255.0  # [T,H,W,C]
        total_frames = int(video.size(0))
        fpc = int(self.dataset_fpcs[0])
        clip_span = fpc * self.frame_step
        clips: list[torch.Tensor] = []
        clip_indices: list[Any] = []
        for clip_idx in range(self.num_clips):
            if total_frames >= clip_span:
                max_start = max(0, total_frames - clip_span)
                if self.random_clip_sampling:
                    start = 0 if max_start == 0 else random.randint(0, max_start)
                else:
                    start = 0 if self.num_clips == 1 else int(round((max_start * clip_idx) / (self.num_clips - 1)))
                indices = list(range(start, start + clip_span, self.frame_step))
            else:
                if self.filter_short_videos:
                    raise ValueError(f"Video too short for requested clip span: {sample}")
                indices = torch.linspace(0, max(total_frames - 1, 0), fpc).round().long().tolist()
            clip_indices.append(indices)
            clip_t = video.index_select(0, torch.tensor(indices, dtype=torch.long))
            if self.shared_transform is not None:
                clip_t = self.shared_transform(clip_t)
            if self.transform is not None:
                clip_t = self.transform(clip_t)
            clips.append(clip_t)
        return clips, self.labels[index], clip_indices


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
    )
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.info("VideoDataset dataset created")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("VideoDataset unsupervised data loader created")
    return dataset, data_loader, dist_sampler
