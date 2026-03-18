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

The implementation is intentionally small, but the module boundaries mirror the
usual V-JEPA style split so we can evolve it toward `facebookresearch/vjepa2`
components later.

## Layout

- `rethinking_jepa/models.py`: video patch embedder, transformer encoder, decoder, predictor
- `rethinking_jepa/masking.py`: token-mask sampling utilities
- `rethinking_jepa/data.py`: synthetic video dataset for smoke tests
- `train_teacher.py`: stage-1 masked reconstruction pretraining
- `train_student.py`: stage-2 frozen-teacher JEPA training

## Quickstart

```bash
python3 train_teacher.py --config configs/teacher.yaml
python3 train_student.py --config configs/student.yaml
```

To try another ViT backbone, swap the config:

```bash
python3 train_teacher.py --config configs/teacher_vit_base.yaml
python3 train_student.py --config configs/student_vit_base.yaml
python3 test_architectures.py --config configs/teacher.yaml --architectures vit_tiny vit_small vit_base
```

By default both scripts use a synthetic dataset, which makes it easy to test the
training code path before wiring in a real video loader.

## Notes

- `global_batch_size: 3072` is tracked in the config to match the paper, while
  `device_batch_size` stays small enough for a local run.
- The repo still uses a synthetic dataset for smoke tests, so this is closer to
  the paper's training recipe but not yet a full reproduction.
- Each architecture writes a different checkpoint file so switching between
  `vit_small` and `vit_base` does not reuse incompatible weights by accident.
# p-salt
