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
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.coverage_sampler import CoverageSampler
from model.transformer import GPTModel
from utils.checkpoint import CheckpointManager
from utils.logging_utils import MetricsLogger, get_logger
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
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=(dtype == torch.float16)
        )

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
        self.best_val_loss: float = self.checkpoint_manager.best_val_loss
        self._train_iter: Optional[Iterator] = None
        self._mps_profile_active = False
        self.coverage_sampler = (
            train_loader.sampler
            if isinstance(train_loader.sampler, CoverageSampler)
            else None
        )
        self.completed_step = 0
        self.safe_to_checkpoint = False

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
        profile_steps = int(self.tcfg.get("mps_profile_steps", 0))
        if profile_steps > 0 and self.device.type == "mps":
            torch.mps.profiler.start(mode="interval", wait_until_completed=False)
            self._mps_profile_active = True
            logger.info("MPS profiling enabled for %d optimizer steps", profile_steps)

        logged_losses: list[torch.Tensor] = []
        t0 = time.perf_counter()
        last_eval_step = -1
        last_eval_loss: Optional[float] = None

        for step in range(start_step, self.tcfg.max_steps):
            self.global_step = step
            self.safe_to_checkpoint = False

            # ---- LR update ----
            lr = self.schedule_fn(step)
            set_lr(self.optimizer, lr)

            # ---- Gradient accumulation micro-steps ----
            logged_losses.extend(self._accumulate_gradients())

            # ---- Optimiser step ----
            if self.tcfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.coverage_sampler is not None:
                self.coverage_sampler.commit(
                    self.tcfg.batch_size * self.tcfg.grad_accumulation_steps
                )
            self.completed_step = step + 1
            self.safe_to_checkpoint = True

            # ---- Logging ----
            if (step + 1) % self.tcfg.log_every_n_steps == 0:
                # One host read per logging interval also waits for queued MPS
                # work, so the elapsed time measures completed GPU work.
                avg_loss = (
                    torch.stack(logged_losses).sum().item()
                    / self.tcfg.log_every_n_steps
                )
                dt = time.perf_counter() - t0
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
                        "perf/window_seconds": dt,
                    },
                    step=step,
                )
                logged_losses.clear()
                t0 = time.perf_counter()

            # ---- Evaluation ----
            step_val_loss: Optional[float] = None
            if (step + 1) % self.tcfg.eval_every_n_steps == 0:
                eval_start = time.perf_counter()
                val_loss = self.evaluate()
                eval_seconds = time.perf_counter() - eval_start
                step_val_loss = val_loss
                last_eval_loss = val_loss
                last_eval_step = step + 1
                self.model.train()
                self.metrics_logger.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": math.exp(min(val_loss, 20)),
                        "perf/eval_seconds": eval_seconds,
                        **self._coverage_metrics(),
                    },
                    step=step,
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.checkpoint_manager.save_best(
                        step=step + 1, model=self.model, val_loss=val_loss
                    )

            # ---- Sample generation ----
            if (
                self.tcfg.generate_every_n_steps > 0
                and (step + 1) % self.tcfg.generate_every_n_steps == 0
            ):
                self._synchronize_device()
                generation_start = time.perf_counter()
                self._log_sample_generation(step)
                self._synchronize_device()
                generation_seconds = time.perf_counter() - generation_start
                self.model.train()
                self.metrics_logger.log(
                    {"perf/generation_seconds": generation_seconds}, step=step
                )

            # ---- Checkpoint ----
            if (step + 1) % self.tcfg.save_every_n_steps == 0:
                self._synchronize_device()
                checkpoint_start = time.perf_counter()
                self.checkpoint_manager.save(
                    step=step + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    val_loss=step_val_loss,
                    extra_meta={
                        "train_loss": (
                            avg_loss
                            if (step + 1) % self.tcfg.log_every_n_steps == 0
                            else None
                        ),
                        "lr": lr,
                        **self._sampler_checkpoint_meta(),
                    },
                )
                self._synchronize_device()
                self.metrics_logger.log(
                    {"perf/checkpoint_seconds": time.perf_counter() - checkpoint_start},
                    step=step,
                )

            if self._mps_profile_active and step - start_step + 1 >= profile_steps:
                self.stop_mps_profile()

        # ---- Final eval + checkpoint ----
        self.stop_mps_profile()
        logger.info("Training complete.  Running final evaluation …")
        val_loss = (
            last_eval_loss if last_eval_step == self.tcfg.max_steps else self.evaluate()
        )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.checkpoint_manager.save_best(
                step=self.tcfg.max_steps, model=self.model, val_loss=val_loss
            )
        self.checkpoint_manager.save(
            step=self.tcfg.max_steps,
            model=self.model,
            optimizer=self.optimizer,
            val_loss=val_loss,
            extra_meta={"final": True, **self._sampler_checkpoint_meta()},
        )
        self.metrics_logger.close()
        logger.info(
            f"Final val_loss={val_loss:.4f}  perplexity={math.exp(min(val_loss,20)):.2f}"
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()

    def stop_mps_profile(self) -> None:
        """Flush optional MPS signposts, including on an interrupted run."""
        if self._mps_profile_active:
            torch.mps.profiler.stop()
            self._mps_profile_active = False
            logger.info("MPS profiling stopped")

    def _sampler_checkpoint_meta(self) -> dict:
        return (
            {"train_sampler": self.coverage_sampler.state_dict()}
            if self.coverage_sampler
            else {}
        )

    def _coverage_metrics(self) -> dict:
        if self.coverage_sampler is None:
            return {}
        progress = self.coverage_sampler.progress()
        return {
            "data/epoch": progress["epoch"],
            "data/blocks_seen_this_epoch": progress["blocks_seen_this_epoch"],
            "data/epoch_coverage_pct": progress["epoch_coverage_pct"],
        }

    def save_emergency_checkpoint(self) -> bool:
        """Save only if the model and data cursor are at a completed update."""
        if not self.safe_to_checkpoint:
            logger.warning(
                "Interrupted during an update; resume from the last complete checkpoint"
            )
            return False
        self.checkpoint_manager.save(
            step=self.completed_step,
            model=self.model,
            optimizer=self.optimizer,
            extra_meta={"interrupted": True, **self._sampler_checkpoint_meta()},
        )
        return True

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
        losses: list[torch.Tensor] = []
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

            losses.append(loss.detach())

        return torch.stack(losses).mean().item() if losses else 0.0

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
                generated = self.tokenizer.decode(
                    gen_ids[0].tolist(), skip_special_tokens=True
                )
                logger.info(
                    f"\n[step {step}] SAMPLE ─────────────\n{generated}\n─────────────────────\n"
                )
            except Exception as exc:
                logger.warning(f"Sample generation failed: {exc}")

    # ------------------------------------------------------------------ #
    #  Gradient accumulation                                               #
    # ------------------------------------------------------------------ #

    def _accumulate_gradients(self) -> list[torch.Tensor]:
        """
        Run ``grad_accumulation_steps`` micro-batches and accumulate gradients.

        Returns:
            Detached, scaled micro-step losses for interval logging.
        """
        losses: list[torch.Tensor] = []
        n = self.tcfg.grad_accumulation_steps

        for _ in range(n):
            batch = self._next_batch()
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.use_amp,
            ):
                _, loss = self.model(input_ids, labels=labels)
                loss = loss / n

            self.scaler.scale(loss).backward()
            losses.append(loss.detach())

        return losses  # each loss is already divided by n

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
            if self.coverage_sampler is not None:
                sampler_state = meta.get("train_sampler")
                if sampler_state is None:
                    logger.warning(
                        "Checkpoint has no tracked data position; starting a new auditable "
                        "shuffle epoch from this model/optimizer checkpoint"
                    )
                else:
                    self.coverage_sampler.load_state_dict(sampler_state)
                    logger.info(
                        "Restored training data position: %s",
                        self.coverage_sampler.progress(),
                    )
            elif meta.get("train_sampler") is not None:
                raise ValueError(
                    "Tracked checkpoint requires its verified train.bin and completion manifest"
                )
            self.completed_step = start_step
            self.safe_to_checkpoint = True
            logger.info(f"Resumed from step {start_step}")
            return start_step

        return 0
