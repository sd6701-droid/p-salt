# Rethinking JEPA

This repository is a compact PyTorch scaffold for the method described in
_Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers_.

It follows the paper's high-level recipe:

1. Train a masked video teacher with reconstruction loss.
2. Freeze the teacher.
3. Train a student JEPA model to predict the teacher's masked latents.

The configs now include the main Table 5 training values from the paper:
`16` frames, frame step `4`, input resolution `224`, random resized crop
augmentation, short/long spatial block masking, cosine learning rate scheduling,
warmup, AdamW betas `(0.9, 0.95)`, gradient clipping, and scheduled weight
decay.

The implementation is intentionally small, but the repository now mirrors the
same high-level folder layout used by `facebookresearch/vjepa2`: `app/` for
entrypoints, `src/` for reusable code, `configs/` for runs, and `tests/` for
sanity checks.

## Layout

- `app/main.py`: config-driven launcher in the same spirit as `vjepa2/app/main.py`
- `app/rethinking_jepa/train.py`: stage-1 teacher training
- `app/rethinking_jepa/student.py`: stage-2 frozen-teacher student training
- `app/rethinking_jepa/prepare_kinetics700_subset.py`: local subset extraction helper
- `src/models`: ViT presets, encoder, predictor, and JEPA teacher/student models
- `src/datasets`: synthetic, local-video, Hugging Face, and mixture dataset loaders
- `src/masks`: token and 3D block masking utilities
- `src/utils`: config loading and schedulers
- `rethinking_jepa`: compatibility exports so older imports still work

## Quickstart

```bash
python3 train_teacher.py --config configs/teacher.yaml
python3 train_student.py --config configs/student.yaml
python3 app/main.py --config configs/teacher.yaml
python3 app/main.py --config configs/student.yaml
```

To try another ViT backbone, swap the config:

```bash
python3 train_teacher.py --config configs/teacher_vit_base.yaml
python3 train_student.py --config configs/student_vit_base.yaml
python3 test_architectures.py --config configs/teacher.yaml --architectures vit_tiny vit_small vit_base
```

By default both scripts use a synthetic dataset, which makes it easy to test the
training code path before wiring in a real video loader.

To point training at real video files, use one of the example configs:

```bash
python3 train_teacher.py --config configs/teacher_real_video.yaml
python3 train_teacher.py --config configs/teacher_v36m_mixture.yaml
python3 train_student.py --config configs/student_v36m_mixture.yaml
```

To stream a manageable subset of the Hugging Face `bitmind/Kinetics-700` dataset directly during training:

```bash
python3 train_teacher.py --config configs/teacher_kinetics700_hf.yaml
python3 train_student.py --config configs/student_kinetics700_hf.yaml
```

Those configs keep the run bounded with `max_samples: 1000`, so only a shuffled subset is consumed on the fly.

## W&B Logging

Teacher training can log metrics to Weights & Biases when `wandb.enabled: true`
is set in the config. The training loop logs `train/loss`, `train/lr`, and
`train/weight_decay` every step.

Put your token in `.env` at the repo root:

```bash
WANDB_API_KEY=your_wandb_api_key_here
```

Then enable the `wandb` block in the teacher config and run training as usual.
If your workspace blocks online writes, set `WANDB_MODE=offline` in `.env` or keep
`fallback_to_offline: true` in the config so training continues and stores local W&B logs.

If you prefer to materialize a local subset first:

```bash
python3 prepare_kinetics700_subset.py --max-videos 1000
python3 train_teacher.py --config configs/teacher_kinetics700_subset.yaml
python3 train_student.py --config configs/student_kinetics700_subset.yaml
```

The real-video loader supports:

- `data.source: real`: read videos from a single `root` directory or `manifest`
- `data.source: huggingface`: stream videos directly from a Hugging Face dataset repo
- `data.source: mixture`: concatenate multiple datasets listed under `data.datasets`
- manifest files in `.txt` format (one path per line) or `.csv` format with a `path` column
- optional `max_samples` and `sample_seed` to train on a partial subset of a larger dataset

## Datasets From The Paper

The paper's pretraining data mixture is:

- `Kinetics-710 (K710)`: Kinetics-400/600/700 merged, with validation samples removed
- `Something-Something V2 (SSV2)`
- `Panda70M` 2.8M-video subset
- combined as `V-3.6M`

The paper's frozen-backbone evaluation datasets are:

- `Kinetics-400`
- `Something-Something V2`
- `COIN`
- `Jester`
- `Diving48`
- `ImageNet-1K` using repeated frames

This repo now includes config scaffolding to train on those paper-style pretraining datasets once you provide
your own local manifests or dataset roots.

## Hugging Face Kinetics-700

The public `bitmind/Kinetics-700` dataset on Hugging Face currently exposes a single `train` split with
roughly `491,660` rows and a `video` column. In practice this is too large for a quick local experiment, so
this repo now supports two practical options:

- direct streaming from Hugging Face with `data.source: huggingface`
- local subset extraction with `prepare_kinetics700_subset.py`

For Hugging Face streaming, install dependencies from `requirements.txt`. The `datasets` docs note that
video columns are decoded through `torchcodec`, and `ffmpeg` must also be available on your system.

## Notes

- `global_batch_size` is used as the effective optimizer batch size via
  gradient accumulation, while `device_batch_size` controls the per-device
  micro-batch that must fit in memory.
- The repo still uses a synthetic dataset for smoke tests, so this is closer to
  the paper's training recipe but not yet a full reproduction.
- Each training run now gets a random run folder under `results/checkpoints/`
  with `best/checkpoint.pth`, `last/checkpoint.pth`, `config.yaml`,
  `log.out`, and `log.err`.
# p-salt
