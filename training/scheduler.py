"""
Learning rate schedulers for Research LLM training.

Provides:
  - Cosine decay with linear warmup (standard for LLM pretraining)
  - Linear warmup + constant
  - Linear warmup + linear decay

All schedulers are implemented as pure functions operating on step count,
rather than wrapping ``torch.optim.lr_scheduler``, to allow precise control
during gradient accumulation.
"""

from __future__ import annotations

import math
from typing import Callable, Optional


def cosine_lr_with_warmup(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Phase 1 (step < warmup_steps):     Linear ramp from 0 → max_lr
    Phase 2 (warmup ≤ step ≤ decay):   Cosine decay from max_lr → min_lr
    Phase 3 (step > decay_steps):      Constant at min_lr

    Args:
        step:          Current optimiser step.
        max_lr:        Peak learning rate.
        min_lr:        Minimum (floor) learning rate.
        warmup_steps:  Number of warmup steps.
        decay_steps:   Total steps over which to decay.

    Returns:
        Learning rate scalar for this step.
    """
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step > decay_steps:
        return min_lr
    # Cosine portion
    progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def linear_lr_with_warmup(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Linear warmup → linear decay schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= decay_steps:
        return min_lr
    progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
    return max_lr - progress * (max_lr - min_lr)


def constant_lr_with_warmup(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Linear warmup → constant LR schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    return max_lr


# ------------------------------------------------------------------ #
#  Factory                                                             #
# ------------------------------------------------------------------ #

_SCHEDULER_MAP = {
    "cosine_with_warmup": cosine_lr_with_warmup,
    "linear_with_warmup": linear_lr_with_warmup,
    "constant_with_warmup": constant_lr_with_warmup,
}


def build_scheduler(
    schedule_name: str,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: Optional[int] = None,
    max_steps: int = 100_000,
) -> Callable[[int], float]:
    """
    Return a callable ``schedule_fn(step) → lr`` for the named schedule.

    Args:
        schedule_name: One of ``"cosine_with_warmup"``, ``"linear_with_warmup"``,
                       ``"constant_with_warmup"``.
        max_lr:        Peak learning rate.
        min_lr:        Minimum learning rate.
        warmup_steps:  Number of warmup steps.
        decay_steps:   Total decay steps (defaults to ``max_steps``).
        max_steps:     Training budget (used when decay_steps is None).

    Returns:
        A function ``(step: int) → float``.
    """
    if schedule_name not in _SCHEDULER_MAP:
        raise ValueError(
            f"Unknown scheduler '{schedule_name}'. "
            f"Choose from: {list(_SCHEDULER_MAP)}"
        )
    fn = _SCHEDULER_MAP[schedule_name]
    _decay = decay_steps if decay_steps is not None else max_steps

    def schedule(step: int) -> float:
        return fn(step, max_lr, min_lr, warmup_steps, _decay)

    return schedule


def get_lr(optimizer: object) -> float:
    """Read current learning rate from an optimizer's first param group."""
    return optimizer.param_groups[0]["lr"]


def set_lr(optimizer: object, lr: float) -> None:
    """Set learning rate on ALL param groups of an optimizer."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr
