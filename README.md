# Rethinking JEPA

This repository is a compact PyTorch scaffold for the method described in
_Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers_.

It follows the paper's high-level recipe:

1. Train a masked video teacher with reconstruction loss.
2. Freeze the teacher.
3. Train a student JEPA model to predict the teacher's masked latents.

The configs now include the main Table 5 training values from the paper:
`16` frames, frame step `4`, input resolution `224`, random resized crop
augmentation, configurable masking strategies, cosine learning rate scheduling,
warmup, AdamW betas `(0.9, 0.95)`, gradient clipping, and scheduled weight
decay. The active teacher and student configs currently use random token
masking with a `0.75` mask ratio.

The implementation is intentionally small, but the repository now mirrors the
same high-level folder layout used by `facebookresearch/vjepa2`: `app/` for
entrypoints, `src/` for reusable code, `configs/` for runs, and `tests/` for
sanity checks.

## Layout

- `app/main.py`: config-driven launcher in the same spirit as `vjepa2/app/main.py`
- `app/rethinking_jepa/extract_squashfs_subset.py`: stage-0 subset extraction from squashfs archives
- `app/rethinking_jepa/train.py`: stage-1 teacher training
- `app/rethinking_jepa/student.py`: stage-2 frozen-teacher student training
- `src/models`: ViT presets, encoder, predictor, and JEPA teacher/student models
- `src/datasets`: synthetic and local-video dataset loaders
- `src/masks`: token and 3D block masking utilities
- `src/utils`: config loading and schedulers
- `rethinking_jepa`: compatibility exports so older imports still work

## Quickstart

```bash
python3 app/main.py --config configs/extract_kinetics400_local_sqf.yaml
python3 app/main.py --config configs/teacher_kinetics400_local_extracted.yaml
python3 app/main.py --config configs/student_kinetics400_local_extracted.yaml
```

To run the architecture smoke test with the kept config:

```bash
python3 test_architectures.py --config configs/teacher_kinetics400_local_extracted.yaml --architectures vit_tiny vit_small vit_base
```

This repository is currently trimmed to the Kinetics-400 extracted-file workflow:
- Stage 0 extracts the selected subset from squashfs into `/scratch`
- Stage 1 trains the teacher from the extracted local files
- Stage 2 trains the student from the same extracted local files

## W&B Logging

Teacher training can log metrics to Weights & Biases when `wandb.enabled: true`
is set in the config. The training loop logs `train/loss`, `train/lr`, and
`train/weight_decay` at the configured `train.log_interval`.

Put your token in `.env` at the repo root:

```bash
WANDB_API_KEY=your_wandb_api_key_here
```

Then enable the `wandb` block in the teacher config and run training as usual.
If your workspace blocks online writes, set `WANDB_MODE=offline` in `.env` or keep
`fallback_to_offline: true` in the config so training continues and stores local W&B logs.

The real-video loader supports:

- `data.source: real`: read videos from a single `root` directory or `manifest`
- manifest files in `.txt` format (one path per line) or `.csv` format with a `path` column
- optional `max_samples` and `sample_seed` to train on a partial subset of a larger dataset

## Notes

- the current trainers step once per dataloader batch, so `device_batch_size`
  is the true runtime batch size
  as recipe metadata.
- The active configs all use the extracted local Kinetics-400 subset workflow.
- Each training run now gets a random run folder under `results/checkpoints/`
  with `best/checkpoint.pth`, `last/checkpoint.pth`, `config.yaml`,
  `log.out`, and `log.err`.
# p-salt
