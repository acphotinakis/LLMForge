"""
Logging utilities for Research LLM.

Provides:
  - Structured console logging via `rich`
  - File logging with rotation
  - WandB / TensorBoard integration hooks
  - A lightweight ProgressBar wrapper
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

_LOGGERS: Dict[str, logging.Logger] = {}
_CONSOLE: Optional[Any] = None


def _get_console() -> Any:
    global _CONSOLE
    if _CONSOLE is None and RICH_AVAILABLE:
        _CONSOLE = Console(stderr=False)
    return _CONSOLE


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    run_name: str = "run",
) -> None:
    """
    Configure root logging.  Call once at process start.

    Args:
        log_dir:  If given, also write logs to ``<log_dir>/<run_name>.log``.
        level:    Logging level (default INFO).
        run_name: Used as the log file name.
    """
    handlers = []

    if RICH_AVAILABLE:
        handlers.append(
            RichHandler(
                console=_get_console(),
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
        )
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{run_name}.log")
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (cached)."""
    if name not in _LOGGERS:
        _LOGGERS[name] = logging.getLogger(name)
    return _LOGGERS[name]


# ------------------------------------------------------------------ #
#  ProgressBar                                                         #
# ------------------------------------------------------------------ #

class ProgressBar:
    """
    Thin wrapper around ``rich.progress.Progress`` (falls back to tqdm).
    Tracks multiple tasks.

    Usage::

        with ProgressBar() as pb:
            task = pb.add_task("Training", total=1000)
            for step in range(1000):
                ...
                pb.update(task, advance=1, loss=f"{loss:.4f}")
    """

    def __init__(self, disable: bool = False):
        self.disable = disable
        self._progress: Optional[Any] = None
        self._tqdm_bars: Dict[Any, Any] = {}

    def __enter__(self) -> "ProgressBar":
        if not self.disable and RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[yellow]{task.fields[metrics]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._progress.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._progress is not None:
            self._progress.__exit__(*args)

    def add_task(self, description: str, total: int) -> Any:
        if self._progress is not None:
            return self._progress.add_task(description, total=total, metrics="")
        # Fallback: return a dummy ID
        return description

    def update(self, task_id: Any, advance: int = 1, **metrics) -> None:
        if self._progress is not None:
            metrics_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
            self._progress.update(task_id, advance=advance, metrics=metrics_str)


# ------------------------------------------------------------------ #
#  MetricsLogger                                                       #
# ------------------------------------------------------------------ #

class MetricsLogger:
    """
    Aggregates scalar metrics and flushes to WandB / TensorBoard / file.

    Usage::

        logger = MetricsLogger(log_dir="./logs", use_wandb=True)
        logger.log({"train/loss": 2.34, "train/lr": 3e-4}, step=100)
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        run_name: str = "run",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._logger = get_logger("metrics")
        self._log_dir = Path(log_dir) if log_dir else None
        self._run_name = run_name
        self._use_wandb = use_wandb
        self._use_tb = use_tensorboard
        self._tb_writer = None
        self._wandb = None

        if use_tensorboard and log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(Path(log_dir) / run_name))
            except ImportError:
                self._logger.warning("TensorBoard not available; skipping.")

        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "research-llm",
                    entity=wandb_entity,
                    name=run_name,
                    config=config or {},
                )
                self._wandb = wandb
            except Exception as exc:
                self._logger.warning(f"WandB init failed: {exc}. Skipping.")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a dict of metrics at a given step."""
        # Console
        parts = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items())
        self._logger.info(f"[step {step}] {parts}")

        # TensorBoard
        if self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, global_step=step)

        # WandB
        if self._wandb:
            self._wandb.log({**metrics, "step": step})

    def close(self) -> None:
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb:
            self._wandb.finish()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
