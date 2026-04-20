from .data_manager import build_video_dataset
from .imagenet_related_classes import (
    KINETICS_TO_IMAGENET_RELATED_LABELS,
    related_imagenet_labels_for_actions,
)
from .video_dataset import (
    HuggingFaceVideoDataset,
    SQUASHFS_FILE_EXTENSIONS,
    SquashFSVideoDataset,
    SyntheticVideoDataset,
    VIDEO_FILE_EXTENSIONS,
    VideoAugmentationConfig,
    VideoFileDataset,
    random_resized_crop_video,
)

try:
    from .image_folder_repeated_frame import ImageFolderRepeatedFrameDataset
except Exception:  # pragma: no cover - optional torchvision dependency
    ImageFolderRepeatedFrameDataset = None

__all__ = [
    "HuggingFaceVideoDataset",
    "ImageFolderRepeatedFrameDataset",
    "KINETICS_TO_IMAGENET_RELATED_LABELS",
    "SQUASHFS_FILE_EXTENSIONS",
    "SquashFSVideoDataset",
    "SyntheticVideoDataset",
    "VIDEO_FILE_EXTENSIONS",
    "VideoAugmentationConfig",
    "VideoFileDataset",
    "build_video_dataset",
    "related_imagenet_labels_for_actions",
    "random_resized_crop_video",
]
