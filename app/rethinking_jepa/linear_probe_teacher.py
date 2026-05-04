# Step 1: Load the trained teacher model, freeze it, and use only its encoder for feature extraction.
# Step 2: Build a labeled video dataset/DataLoader for the probe task.
# Step 3: Convert each video batch into teacher encoder features and pool token features into one vector per video.
# Step 4: Train a small linear classifier on top of those frozen teacher features.
# Step 5: Evaluate the classifier with accuracy metrics and save the best/latest probe checkpoints.
# Step 6: Once the flow is verified, wire this script into the app entrypoint and Slurm config.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rethinking_jepa.utils import build_teacher_from_cfg, resolve_device
from src.datasets.data_manager import _load_manifest_paths
from src.datasets.video_dataset import VideoAugmentationConfig, VideoFileDataset
from src.utils import load_config
from src.utils.wandb import finish_wandb_run, init_wandb_run, log_wandb_metrics


def checkpoint_path_from_args(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser()
    if args.config:
        cfg = load_config(args.config)
        return Path(cfg["train"]["teacher_checkpoint"]).expanduser()
    return Path("checkpoint.pth")


def print_keys(name: str, value: object) -> None:
    print(name, type(value))
    if isinstance(value, dict):
        keys = list(value.keys())
        print(f"{name}.keys()")
        for key in keys:
            print(f"  {key}")


def extract_model_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)!r}")


def load_frozen_teacher_encoder(
    cfg: dict,
    device: torch.device,
    *,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    # this build a teacher skeleton after which we load wts to this 
    teacher, _ = build_teacher_from_cfg(cfg, device)
    if checkpoint_path is None:
        checkpoint_path = Path(cfg["train"]["teacher_checkpoint"]).expanduser()
    ckpt = torch.load(checkpoint_path, map_location=device)
    teacher.load_state_dict(extract_model_state_dict(ckpt))
    encoder = teacher.encoder
    # this stops Batch and normalization updates right ? 
    encoder.eval()
    # for every params in the encoder we stop the flow of the gradients 
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


class LabeledVideoDataset(Dataset):
    """Wraps VideoFileDataset to also return the integer class label per sample.

    Owns its own decode-error fallback so that when a clip fails and we walk
    forward to a different index, the returned label still matches the clip
    that was actually loaded (preventing video/label mismatch).
    """

    def __init__(
        self,
        video_paths: list[Path],
        labels: list[int],
        *,
        channels: int,
        frames: int,
        frame_step: int,
        image_size: int,
        augmentation: VideoAugmentationConfig | None = None,
        skip_decode_errors: bool = False,
        max_decode_attempts: int = 16,
        log_decode_warnings: bool = True,
    ) -> None:
        if len(video_paths) != len(labels):
            raise ValueError(
                f"video_paths ({len(video_paths)}) and labels ({len(labels)}) length mismatch"
            )
        # the inner dataset is built with skip_decode_errors=False so we own the
        # fallback and can return the matching label for whichever index loaded
        self._inner = VideoFileDataset(
            video_paths=video_paths,
            channels=channels,
            frames=frames,
            frame_step=frame_step,
            image_size=image_size,
            augmentation=augmentation,
            skip_decode_errors=False,
            max_decode_attempts=max_decode_attempts,
            log_decode_warnings=log_decode_warnings,
        )
        self._labels = list(labels)
        self._skip_decode_errors = bool(skip_decode_errors)
        self._max_decode_attempts = max(1, int(max_decode_attempts))

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not self._skip_decode_errors:
            return self._inner[index], self._labels[index]
        attempts = min(self._max_decode_attempts, len(self._labels))
        last_error: Exception | None = None
        for attempt in range(attempts):
            candidate = (index + attempt) % len(self._labels)
            try:
                return self._inner[candidate], self._labels[candidate]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"Failed to decode any usable clip in {attempts} attempts starting at index {index}"
        ) from last_error


def resolve_manifest_paths(manifest_path: Path) -> list[Path]:
    base_dir = manifest_path.parent
    raw = _load_manifest_paths(manifest_path)
    return [p if p.is_absolute() else (base_dir / p).resolve() for p in raw]


def load_labeled_paths(
    manifest_path: str | Path,
    *,
    class_to_idx: dict[str, int] | None = None,
) -> tuple[list[Path], list[int], dict[str, int]]:
    # derive labels from each path's parent directory; train builds the mapping,
    # val reuses it so indices stay consistent across splits
    manifest_path = Path(manifest_path).expanduser().resolve()
    paths = resolve_manifest_paths(manifest_path)
    class_names = [p.parent.name for p in paths]
    if class_to_idx is None:
        unique = sorted(set(class_names))
        class_to_idx = {name: idx for idx, name in enumerate(unique)}
    missing = sorted({n for n in class_names if n not in class_to_idx})
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} contains classes not in class_to_idx: {missing}"
        )
    labels = [class_to_idx[n] for n in class_names]
    return paths, labels, class_to_idx


def build_probe_loader(
    cfg: dict,
    manifest_key: str,
    *,
    shuffle: bool,
    class_to_idx: dict[str, int] | None = None,
) -> tuple[DataLoader, dict[str, int]]:
    data_cfg = cfg["data"]
    probe_cfg = cfg.get("probe", {})

    manifest_path = data_cfg[manifest_key]
    paths, labels, class_to_idx = load_labeled_paths(manifest_path, class_to_idx=class_to_idx)

    # train uses the configured augmentation; val gets deterministic resize-only
    augmentation = None
    if shuffle and "augmentation" in cfg:
        aug_cfg = cfg["augmentation"]
        augmentation = VideoAugmentationConfig(
            input_size=int(data_cfg["input_size"]),
            random_resize_aspect_ratio=tuple(aug_cfg["random_resize_aspect_ratio"]),
            random_resize_scale=tuple(aug_cfg["random_resize_scale"]),
        )

    dataset = LabeledVideoDataset(
        video_paths=paths,
        labels=labels,
        channels=int(cfg["model"]["in_channels"]),
        frames=int(data_cfg["frames"]),
        frame_step=int(data_cfg["frame_step"]),
        image_size=int(data_cfg["image_size"]),
        augmentation=augmentation,
        skip_decode_errors=bool(data_cfg.get("skip_decode_errors", False)),
        max_decode_attempts=int(data_cfg.get("max_decode_attempts", 16)),
        log_decode_warnings=bool(data_cfg.get("log_decode_warnings", True)),
    )

    num_workers = int(probe_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(probe_cfg.get("batch_size", 32)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(probe_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **loader_kwargs), class_to_idx


def extract_features(
    encoder: nn.Module,
    video: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # encoder is already eval() + requires_grad_(False); no_grad keeps the
    # output usable as a leaf input to the trainable linear head's autograd
    # graph. Mean over the token dim collapses (B, N, D) -> (B, D).
    video = video.to(device, non_blocking=(device.type == "cuda"))
    with torch.no_grad():
        tokens, _ = encoder(video)
    return tokens.mean(dim=1)


def build_linear_head(embed_dim: int, num_classes: int, device: torch.device) -> nn.Linear:
    head = nn.Linear(embed_dim, num_classes)
    return head.to(device)


def build_probe_optimizer(
    head: nn.Module,
    probe_cfg: dict,
    *,
    steps_per_epoch: int,
    epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=float(probe_cfg.get("lr", 0.1)),
        momentum=float(probe_cfg.get("momentum", 0.9)),
        weight_decay=float(probe_cfg.get("weight_decay", 0.0)),
    )
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return optimizer, scheduler


def train_one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, float]:
    head.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for video, labels in loader:
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        features = extract_features(encoder, video, device)
        logits = head(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_size

    seen = max(1, seen)
    return {
        "loss": loss_sum / seen,
        "acc": correct / seen,
        "lr": optimizer.param_groups[0]["lr"],
    }


def derive_probe_run_name(cfg: dict) -> str:
    # Convention: .../checkpoints/<id>/{best,last,latest}/checkpoint.pth
    # Walk up from the file and take the first segment that isn't a reserved
    # bucket name; falls back to the file stem if the layout is unusual.
    reserved = {"best", "last", "latest", "checkpoints"}
    ckpt = Path(cfg["train"]["teacher_checkpoint"]).expanduser()
    ckpt_id = next(
        (part for part in reversed(ckpt.parts[:-1]) if part and part not in reserved),
        ckpt.stem,
    )
    arch = cfg.get("model", {}).get("architecture", "encoder")
    return f"probe-teacher-{arch}-{ckpt_id}"


def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    seen = 0
    with torch.no_grad():
        for video, labels in loader:
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            features = extract_features(encoder, video, device)
            logits = head(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            # top-k against the number of classes; topk(5) on a 4-class probe would crash
            k = min(5, logits.size(1))
            topk = logits.topk(k, dim=1).indices
            match = topk.eq(labels.unsqueeze(1))
            correct1 += match[:, 0].sum().item()
            correct5 += match.any(dim=1).sum().item()
            seen += batch_size

    seen = max(1, seen)
    return {
        "loss": loss_sum / seen,
        "acc": correct1 / seen,
        "acc5": correct5 / seen,
    }


def save_probe_checkpoint(
    path: Path,
    *,
    epoch: int,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    class_to_idx: dict[str, int],
    best_val_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_to_idx": class_to_idx,
            "best_val_acc": best_val_acc,
        },
        path,
    )


def main(cfg: dict | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args() if cfg is None else argparse.Namespace(
        checkpoint=None,
        config=None,
    )
    if cfg is None and args.config:
        cfg = load_config(args.config)
    checkpoint_path = Path(cfg["train"]["teacher_checkpoint"]).expanduser() if cfg is not None else checkpoint_path_from_args(args)

    print(f"checkpoint_path={checkpoint_path}")
    # ckpt = torch.load(checkpoint_path, map_location="cpu")
    #print_keys("ckpt", ckpt)

    if cfg is None:
        return

    #check which device we are using //cpu, apple gpu, gpu
    device = resolve_device()
    # load the frozen encoder 
    encoder = load_frozen_teacher_encoder(cfg, device, checkpoint_path=checkpoint_path)
    num_params = sum(p.numel() for p in encoder.parameters())
    print(
        "linear-probe teacher: encoder loaded "
        f"device={device} embed_dim={encoder.embed_dim} "
        f"patch_size={encoder.patch_size} tubelet_size={encoder.tubelet_size} "
        f"params={num_params}"
    )

    # Step 2 sanity check: build train + val loaders, peek at one batch.
    train_loader, class_to_idx = build_probe_loader(cfg, "train_manifest", shuffle=True)
    val_loader, _ = build_probe_loader(
        cfg, "val_manifest", shuffle=False, class_to_idx=class_to_idx
    )
    print(
        "linear-probe teacher: loaders built "
        f"num_classes={len(class_to_idx)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )
    video, labels = next(iter(train_loader))
    print(
        "linear-probe teacher: first batch "
        f"video.shape={tuple(video.shape)} labels.shape={tuple(labels.shape)} "
        f"labels[:8]={labels[:8].tolist()}"
    )

    # Step 3 sanity check: encode the batch and pool tokens to one vector per video.
    features = extract_features(encoder, video, device)
    print(
        "linear-probe teacher: pooled features "
        f"features.shape={tuple(features.shape)} dtype={features.dtype} "
        f"mean={features.mean().item():.4f} std={features.std().item():.4f}"
    )

    # Step 4: train a linear classifier on the frozen pooled features.
    probe_cfg = cfg.get("probe", {})
    epochs = int(probe_cfg.get("epochs", 20))
    num_classes = len(class_to_idx)

    head = build_linear_head(encoder.embed_dim, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_probe_optimizer(
        head, probe_cfg, steps_per_epoch=len(train_loader), epochs=epochs
    )

    assert all(not p.requires_grad for p in encoder.parameters()), "encoder must stay frozen"
    assert any(p.requires_grad for p in head.parameters()), "head must be trainable"

    print(
        "linear-probe teacher: training head "
        f"num_classes={num_classes} epochs={epochs} "
        f"lr={optimizer.param_groups[0]['lr']} "
        f"momentum={optimizer.param_groups[0]['momentum']} "
        f"weight_decay={optimizer.param_groups[0]['weight_decay']}"
    )

    checkpoint_dir = Path(probe_cfg["checkpoint_dir"]).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = float("-inf")
    best_epoch = -1

    cfg.setdefault("wandb", {}).setdefault("name", derive_probe_run_name(cfg))
    wandb_run = init_wandb_run(cfg, job_type="probe-teacher")
    if wandb_run is not None:
        # shared util only registers train/* — register probe/* against epoch.
        wandb_run.define_metric("probe/epoch")
        wandb_run.define_metric("probe/*", step_metric="probe/epoch")

    for epoch in range(epochs):
        train_stats = train_one_epoch(
            encoder, head, train_loader, optimizer, criterion, scheduler, device
        )
        val_stats = evaluate(encoder, head, val_loader, criterion, device)

        improved = val_stats["acc"] > best_val_acc
        if improved:
            best_val_acc = val_stats["acc"]
            best_epoch = epoch + 1

        print(
            f"linear-probe teacher: epoch {epoch + 1}/{epochs} "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f} "
            f"val_acc5={val_stats['acc5']:.4f} lr={train_stats['lr']:.5f}"
            + (" [best]" if improved else "")
        )

        log_wandb_metrics(
            wandb_run,
            {
                "probe/epoch": epoch + 1,
                "probe/lr": train_stats["lr"],
                "probe/train_loss": train_stats["loss"],
                "probe/train_acc": train_stats["acc"],
                "probe/val_loss": val_stats["loss"],
                "probe/val_acc": val_stats["acc"],
                "probe/val_acc5": val_stats["acc5"],
                "probe/best_val_acc": best_val_acc,
            },
        )

        save_probe_checkpoint(
            checkpoint_dir / "latest.pth",
            epoch=epoch + 1,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            class_to_idx=class_to_idx,
            best_val_acc=best_val_acc,
        )
        if improved:
            save_probe_checkpoint(
                checkpoint_dir / "best.pth",
                epoch=epoch + 1,
                head=head,
                optimizer=optimizer,
                scheduler=scheduler,
                class_to_idx=class_to_idx,
                best_val_acc=best_val_acc,
            )

    print(
        f"linear-probe teacher: training done "
        f"best_val_acc={best_val_acc:.4f} best_epoch={best_epoch} "
        f"checkpoint_dir={checkpoint_dir}"
    )

    finish_wandb_run(
        wandb_run,
        summary={
            "probe/best_val_acc": best_val_acc,
            "probe/best_epoch": best_epoch,
            "probe/checkpoint_dir": str(checkpoint_dir),
        },
    )


if __name__ == "__main__":
    main()
