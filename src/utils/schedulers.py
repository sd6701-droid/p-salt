from __future__ import annotations

import math

from torch.optim import Optimizer


class CosineScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        start_lr: float,
        peak_lr: float,
        final_lr: float,
        start_weight_decay: float,
        end_weight_decay: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.start_weight_decay = start_weight_decay
        self.end_weight_decay = end_weight_decay

    def _cosine_interp(self, start: float, end: float, step: int, total_steps: int) -> float:
        progress = min(1.0, max(0.0, step / max(1, total_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end + (start - end) * cosine

    def step(self, step: int) -> tuple[float, float]:
        if step < self.warmup_steps:
            warmup_progress = step / max(1, self.warmup_steps)
            lr = self.start_lr + warmup_progress * (self.peak_lr - self.start_lr)
        else:
            lr = self._cosine_interp(
                self.peak_lr,
                self.final_lr,
                step - self.warmup_steps,
                self.total_steps - self.warmup_steps,
            )
        wd = self._cosine_interp(
            self.start_weight_decay,
            self.end_weight_decay,
            step,
            self.total_steps,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = wd
        return lr, wd
