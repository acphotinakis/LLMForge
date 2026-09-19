"""
Training loop for Research LLM.

Implements:
  - Full training loop with gradient accumulation
  - Mixed precision (FP16 / BF16) via torch.amp
  - Gradient clipping
  - Periodic evaluation with perplexity
  - Sample text generation during training
  - Checkpoint save / resume
  - Metric logging (console + optional WandB / TensorBoard)
  - Multi-GPU via torch.nn.parallel.DistributedDataParallel (optional)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model.transformer import GPTModel
from ..utils.checkpoint import CheckpointManager
from ..utils.logging_utils import MetricsLogger, get_logger
from .scheduler import build_scheduler, get_lr, set_lr

logger = get_logger(__name__)


class Trainer:
    """
    End-to-end trainer for a ``GPTModel``.

    Args:
        model:          The GPT model to train.
        cfg:            Top-level Config object.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        tokenizer:      Tokenizer (used for sample generation).
        device:         Target device.
        dtype:          Training dtype (torch.float32 / float16 / bfloat16).
    """

    def __init__(
        self,
        model: GPTModel,
        cfg: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Any,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.tcfg = cfg.training  # shortcut

        # ---- Optimizer ----
        self.optimizer = model.configure_optimiser(
            learning_rate=self.tcfg.learning_rate,
            weight_decay=self.tcfg.weight_decay,
            beta1=self.tcfg.beta1,
            beta2=self.tcfg.beta2,
            eps=self.tcfg.eps,
            device_type=device.type,
        )

        # ---- LR schedule ----
        self.schedule_fn: Callable[[int], float] = build_scheduler(
            schedule_name=self.tcfg.scheduler,
            max_lr=self.tcfg.learning_rate,
            min_lr=self.tcfg.min_lr,
            warmup_steps=self.tcfg.warmup_steps,
            decay_steps=self.tcfg.get("decay_steps", None),
            max_steps=self.tcfg.max_steps,
        )

        # ---- Mixed precision scaler (FP16 only — BF16 doesn't need it) ----
        self.use_amp = dtype in (torch.float16, torch.bfloat16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

        # ---- Checkpoint manager ----
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.tcfg.output_dir,
            keep_last_n=self.tcfg.keep_last_n_checkpoints,
        )

        # ---- Metrics logger ----
        wandb_cfg = cfg.training.get("wandb", None)
        self.metrics_logger = MetricsLogger(
            log_dir=self.tcfg.log_dir,
            run_name=self.tcfg.run_name,
            use_wandb=wandb_cfg.enabled if wandb_cfg else False,
            use_tensorboard=True,
            wandb_project=wandb_cfg.project if wandb_cfg else None,
            config=cfg.to_dict() if hasattr(cfg, "to_dict") else {},
        )

        # ---- State ----
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self._train_iter: Optional[Iterator] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Run the full training loop."""
        # Resume from checkpoint if requested
        start_step = self._maybe_resume()

        logger.info(
            f"Starting training: max_steps={self.tcfg.max_steps}, "
            f"batch={self.tcfg.batch_size}, "
            f"grad_accum={self.tcfg.grad_accumulation_steps}, "
            f"dtype={self.dtype}, device={self.device}"
        )

        self.model.train()
        self._train_iter = iter(self.train_loader)

        accum_loss = 0.0
        accum_tokens = 0
        t0 = time.perf_counter()

        for step in range(start_step, self.tcfg.max_steps):
            self.global_step = step

            # ---- LR update ----
            lr = self.schedule_fn(step)
            set_lr(self.optimizer, lr)

            # ---- Gradient accumulation micro-steps ----
            loss = self._accumulate_gradients()
            accum_loss += loss

            # ---- Optimiser step ----
            if self.tcfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # ---- Logging ----
            if (step + 1) % self.tcfg.log_every_n_steps == 0:
                dt = time.perf_counter() - t0
                avg_loss = accum_loss / self.tcfg.log_every_n_steps
                tokens_per_sec = (
                    self.tcfg.batch_size
                    * self.model.cfg.context_length
                    * self.tcfg.grad_accumulation_steps
                    * self.tcfg.log_every_n_steps
                    / max(dt, 1e-6)
                )
                self.metrics_logger.log(
                    {
                        "train/loss": avg_loss,
                        "train/perplexity": math.exp(min(avg_loss, 20)),
                        "train/lr": lr,
                        "perf/tokens_per_sec": tokens_per_sec,
                    },
                    step=step,
                )
                accum_loss = 0.0
                t0 = time.perf_counter()

            # ---- Evaluation ----
            if (step + 1) % self.tcfg.eval_every_n_steps == 0:
                val_loss = self.evaluate()
                self.model.train()
                self.metrics_logger.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": math.exp(min(val_loss, 20)),
                    },
                    step=step,
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            # ---- Sample generation ----
            if (step + 1) % self.tcfg.generate_every_n_steps == 0:
                self._log_sample_generation(step)
                self.model.train()

            # ---- Checkpoint ----
            if (step + 1) % self.tcfg.save_every_n_steps == 0:
                val_loss_for_ckpt = self.best_val_loss
                self.checkpoint_manager.save(
                    step=step + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    val_loss=val_loss_for_ckpt,
                    extra_meta={
                        "train_loss": avg_loss if "avg_loss" in dir() else None,
                        "lr": lr,
                    },
                )

        # ---- Final eval + checkpoint ----
        logger.info("Training complete.  Running final evaluation …")
        val_loss = self.evaluate()
        self.checkpoint_manager.save(
            step=self.tcfg.max_steps,
            model=self.model,
            optimizer=self.optimizer,
            val_loss=val_loss,
            extra_meta={"final": True},
        )
        self.metrics_logger.close()
        logger.info(f"Final val_loss={val_loss:.4f}  perplexity={math.exp(min(val_loss,20)):.2f}")

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on the validation set.

        Returns:
            Mean cross-entropy loss over ``cfg.training.eval_steps`` batches.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0
        val_iter = iter(self.val_loader)
        max_batches = self.tcfg.eval_steps

        for batch_idx, batch in enumerate(val_iter):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.use_amp,
            ):
                _, loss = self.model(input_ids, labels=labels)

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------ #
    #  Sample generation                                                   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _log_sample_generation(self, step: int) -> None:
        """Generate a sample and log it."""
        self.model.eval()
        sample_prompts = [
            "Abstract: In this paper, we propose",
            "Introduction: Recent advances in deep learning have",
            "The proposed method achieves",
        ]
        for prompt in sample_prompts:
            try:
                ids = self.tokenizer.encode(prompt, add_bos=True)
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                gen_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=128,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
                logger.info(f"\n[step {step}] SAMPLE ─────────────\n{generated}\n─────────────────────\n")
            except Exception as exc:
                logger.warning(f"Sample generation failed: {exc}")

    # ------------------------------------------------------------------ #
    #  Gradient accumulation                                               #
    # ------------------------------------------------------------------ #

    def _accumulate_gradients(self) -> float:
        """
        Run ``grad_accumulation_steps`` micro-batches and accumulate gradients.

        Returns:
            Averaged loss (float) across micro-steps.
        """
        total_loss = 0.0
        n = self.tcfg.grad_accumulation_steps

        for micro_step in range(n):
            batch = self._next_batch()
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Sync gradients only on the last micro-step
            is_last = micro_step == n - 1
            ctx = (
                self.model.no_sync()
                if hasattr(self.model, "no_sync") and not is_last
                else _null_context()
            )

            with ctx:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.use_amp,
                ):
                    _, loss = self.model(input_ids, labels=labels)
                    # Scale loss for accumulation
                    loss = loss / n

                self.scaler.scale(loss).backward()
                total_loss += loss.item()

        return total_loss  # already divided by n

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        """Get the next training batch, resetting the iterator on exhaustion."""
        try:
            return next(self._train_iter)
        except StopIteration:
            logger.info("Training iterator exhausted; starting new epoch.")
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    # ------------------------------------------------------------------ #
    #  Resume                                                              #
    # ------------------------------------------------------------------ #

    def _maybe_resume(self) -> int:
        """Load checkpoint if resume_from is set. Returns starting step."""
        resume_path = self.tcfg.get("resume_from", None)
        if not resume_path:
            # Auto-detect latest checkpoint
            latest = self.checkpoint_manager.latest_checkpoint()
            if latest:
                logger.info(f"Auto-detected checkpoint: {latest}.  Resuming …")
                resume_path = str(latest)

        if resume_path:
            meta = self.checkpoint_manager.load(
                checkpoint_path=resume_path,
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
            )
            start_step = meta.get("step", 0)
            logger.info(f"Resumed from step {start_step}")
            return start_step

        return 0


# ------------------------------------------------------------------ #
#  Context manager helper                                             #
# ------------------------------------------------------------------ #

class _null_context:
    def __enter__(self): return self
    def __exit__(self, *args): pass
