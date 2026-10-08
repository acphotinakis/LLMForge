"""
Checkpoint management for Research LLM.

Handles:
  - Saving full training state (model, optimizer, scheduler, step, loss)
  - Loading / resuming from checkpoints
  - Keeping only the last N checkpoints
  - Tracking the "best" checkpoint by validation loss
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .logging_utils import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages checkpoint saving and loading.

    Directory layout::

        output_dir/
            checkpoint-step-1000/
                model.pt          ← model state dict
                optimizer.pt      ← optimizer state dict
                scheduler.pt      ← scheduler state dict
                meta.json         ← step, loss, config hash, etc.
            checkpoint-step-2000/
                ...
            best.pt               ← copy of best model state dict

    Args:
        output_dir:         Root directory for checkpoints.
        keep_last_n:        How many recent checkpoints to keep (0 = keep all).
        save_optimizer:     Whether to save optimizer state (needed for resume).
    """

    def __init__(
        self,
        output_dir: str,
        keep_last_n: int = 3,
        save_optimizer: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self._checkpoints: List[Path] = self._scan_existing()
        self._best_val_loss: float = float("inf")
        best_meta = self.output_dir / "best_meta.json"
        if best_meta.exists():
            try:
                saved_loss = json.loads(best_meta.read_text()).get("val_loss")
                if saved_loss is not None:
                    self._best_val_loss = float(saved_loss)
            except (OSError, ValueError, TypeError):
                logger.warning(
                    "Ignoring invalid best checkpoint metadata: %s", best_meta
                )

    # ------------------------------------------------------------------ #
    #  Saving                                                              #
    # ------------------------------------------------------------------ #

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        val_loss: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Returns the path to the checkpoint directory.
        """
        ckpt_dir = self.output_dir / f"checkpoint-step-{step:08d}"
        meta: Dict[str, Any] = {
            "step": step,
            "val_loss": val_loss,
        }
        if extra_meta:
            meta.update(extra_meta)

        # A hidden staging directory is ignored by checkpoint discovery. Move it
        # into place only after model, optimizer, and metadata have been written.
        stage = Path(tempfile.mkdtemp(prefix=".checkpoint-stage-", dir=self.output_dir))
        backup = None
        try:
            raw_model = _unwrap_model(model)
            torch.save(raw_model.state_dict(), stage / "model.pt")
            if self.save_optimizer and optimizer is not None:
                torch.save(optimizer.state_dict(), stage / "optimizer.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), stage / "scheduler.pt")
            with open(stage / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            # The final evaluation can save the same step again. Retain the
            # previous complete checkpoint until its replacement is published.
            if ckpt_dir.exists():
                backup = Path(
                    tempfile.mkdtemp(prefix=".checkpoint-backup-", dir=self.output_dir)
                )
                backup.rmdir()
                os.replace(ckpt_dir, backup)
            os.replace(stage, ckpt_dir)
        except BaseException:
            if backup is not None and backup.exists() and not ckpt_dir.exists():
                os.replace(backup, ckpt_dir)
            raise
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)

        self._checkpoints = [p for p in self._checkpoints if p != ckpt_dir]
        self._checkpoints.append(ckpt_dir)
        logger.info(f"Saved checkpoint → {ckpt_dir}")

        # Track best
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._save_best(ckpt_dir)

        # Prune old checkpoints
        self._prune()

        return ckpt_dir

    def _save_best(self, ckpt_dir: Path) -> None:
        """Copy model.pt to best.pt in the output root."""
        best_path = self.output_dir / "best.pt"
        best_stage = best_path.with_suffix(".pt.tmp")
        shutil.copy2(ckpt_dir / "model.pt", best_stage)
        os.replace(best_stage, best_path)
        # Also write best meta
        with open(ckpt_dir / "meta.json") as f:
            meta = json.load(f)
        best_meta_stage = self.output_dir / "best_meta.json.tmp"
        with open(best_meta_stage, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(best_meta_stage, self.output_dir / "best_meta.json")
        logger.info(
            f"New best checkpoint at step {meta['step']} (val_loss={meta['val_loss']:.4f})"
        )

    def save_best(self, step: int, model: torch.nn.Module, val_loss: float) -> bool:
        """Save the exact weights just evaluated, without an optimizer checkpoint."""
        if val_loss >= self._best_val_loss:
            return False
        stage = Path(tempfile.mkdtemp(prefix=".best-stage-", dir=self.output_dir))
        try:
            torch.save(_unwrap_model(model).state_dict(), stage / "model.pt")
            with open(stage / "meta.json", "w") as f:
                json.dump({"step": step, "val_loss": val_loss}, f, indent=2)
            os.replace(stage / "model.pt", self.output_dir / "best.pt")
            os.replace(stage / "meta.json", self.output_dir / "best_meta.json")
        finally:
            shutil.rmtree(stage)
        self._best_val_loss = val_loss
        logger.info(f"New best checkpoint at step {step} (val_loss={val_loss:.4f})")
        return True

    def _prune(self) -> None:
        if self.keep_last_n <= 0:
            return
        while len(self._checkpoints) > self.keep_last_n:
            old = self._checkpoints.pop(0)
            if old.exists():
                shutil.rmtree(old)
                logger.debug(f"Removed old checkpoint: {old}")

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint into model (and optionally optimizer / scheduler).

        Args:
            checkpoint_path: Path to a checkpoint directory OR a ``model.pt`` file
                             OR ``best.pt``.
            model:           Model to load weights into.
            optimizer:       Optional optimizer to restore state.
            scheduler:       Optional scheduler to restore state.
            device:          Target device (default: same as model).
            strict:          Passed to ``load_state_dict``.

        Returns:
            The ``meta.json`` dict (contains ``step``, ``val_loss``, etc.).
        """
        ckpt_path = Path(checkpoint_path)

        # Determine model.pt path
        if ckpt_path.is_dir():
            model_file = ckpt_path / "model.pt"
            opt_file = ckpt_path / "optimizer.pt"
            sched_file = ckpt_path / "scheduler.pt"
            meta_file = ckpt_path / "meta.json"
        else:
            # Assume it's a bare model.pt (e.g. best.pt)
            model_file = ckpt_path
            parent = ckpt_path.parent
            opt_file = parent / "optimizer.pt"
            sched_file = parent / "scheduler.pt"
            meta_file = (
                parent / "best_meta.json"
                if ckpt_path.name == "best.pt"
                else parent / "meta.json"
            )

        if device is None:
            device = next(model.parameters()).device

        # Load model
        state = torch.load(model_file, map_location=device, weights_only=True)
        raw_model = _unwrap_model(model)
        missing, unexpected = raw_model.load_state_dict(state, strict=strict)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        logger.info(f"Loaded model weights from {model_file}")

        # Optimizer
        if optimizer is not None and opt_file.exists():
            optimizer.load_state_dict(
                torch.load(opt_file, map_location=device, weights_only=True)
            )
            logger.info("Restored optimizer state.")

        # Scheduler
        if scheduler is not None and sched_file.exists():
            scheduler.load_state_dict(
                torch.load(sched_file, map_location=device, weights_only=True)
            )
            logger.info("Restored scheduler state.")

        # Meta
        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        return meta

    # ------------------------------------------------------------------ #
    #  Discovery                                                           #
    # ------------------------------------------------------------------ #

    def _scan_existing(self) -> List[Path]:
        """Find and sort existing checkpoint directories."""
        ckpts = sorted(
            [
                d
                for d in self.output_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("checkpoint-step-")
                and d.name.removeprefix("checkpoint-step-").isdigit()
                and (d / "meta.json").is_file()
                and (d / "model.pt").is_file()
                and (not self.save_optimizer or (d / "optimizer.pt").is_file())
            ],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        return ckpts

    def latest_checkpoint(self) -> Optional[Path]:
        """Return the most recently saved checkpoint directory, or None."""
        if self._checkpoints:
            return self._checkpoints[-1]
        return None

    def best_checkpoint(self) -> Optional[Path]:
        """Return path to best.pt if it exists."""
        best = self.output_dir / "best.pt"
        return best if best.exists() else None

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP / DataParallel wrappers."""
    if hasattr(model, "module"):
        return model.module
    return model
