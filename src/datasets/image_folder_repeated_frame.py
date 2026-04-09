from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class ImageFolderRepeatedFrameDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        root: str | Path,
        input_size: int,
        frames: int,
        resize_size: int | None = None,
        max_samples: int | None = None,
        sample_seed: int = 0,
        class_names: list[str] | None = None,
    ) -> None:
        dataset = ImageFolder(str(Path(root).expanduser()))
        samples = list(dataset.samples)
        selected_class_names: list[str] | None = None

        if class_names:
            allowed = set(class_names)
            samples = [sample for sample in samples if dataset.classes[sample[1]] in allowed]
            if not samples:
                raise ValueError(f"No images found for class_names={class_names} under {root}")
            selected_class_names = [name for name in class_names if name in dataset.class_to_idx]

        if max_samples is not None and len(samples) > max_samples:
            rng = random.Random(sample_seed)
            rng.shuffle(samples)
            samples = samples[:max_samples]

        self.root = Path(root).expanduser()
        self.samples = samples
        if selected_class_names is not None:
            self.classes = selected_class_names
            self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
            self._label_remap = {
                dataset.class_to_idx[name]: self.class_to_idx[name] for name in self.classes
            }
        else:
            kept_label_ids = sorted({label for _, label in samples})
            self.classes = [dataset.classes[label] for label in kept_label_ids]
            self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
            self._label_remap = {label: idx for idx, label in enumerate(kept_label_ids)}
        self._all_classes = dataset.classes
        self.input_size = input_size
        self.frames = frames
        self.resize_size = resize_size or input_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        path_str, label = self.samples[index]
        path = Path(path_str)
        image = Image.open(path).convert("RGB")
        image = TF.resize(image, self.resize_size, interpolation=InterpolationMode.BICUBIC)
        image = TF.center_crop(image, [self.input_size, self.input_size])
        image_tensor = TF.to_tensor(image)
        video = image_tensor.unsqueeze(1).repeat(1, self.frames, 1, 1)
        return {
            "video": video,
            "image": image_tensor,
            "label": self._all_classes[label],
            "label_index": self._label_remap[label],
            "path": str(path),
        }
