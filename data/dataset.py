"""
Data loading from Parquet files for Research LLM.

Architecture:
  ParquetDataset   → lazy streaming over .parquet files using PyArrow / Polars
  TextDataset      → wraps tokenized token IDs for PyTorch DataLoader
  build_dataloaders → convenience factory that wires everything together

Design goals:
  - Handle datasets that do NOT fit in RAM (streaming / memory-mapped reads)
  - Support millions of documents across hundreds of Parquet files
  - Efficient batching with variable-length sequences packed into fixed blocks
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .preprocessing import TextPreprocessor
from .binary_manifest import valid_manifest
from .coverage_sampler import CoverageSampler
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. Parquet streaming iterator
# ============================================================

class ParquetStreamIterator:
    """
    Streams text from a list of Parquet files without loading all data into RAM.

    Uses PyArrow for low-level streaming; falls back to Polars if PyArrow is
    unavailable.

    Args:
        files:          List of .parquet file paths.
        text_column:    Column that contains raw text.
        batch_size:     Number of rows to read per arrow batch (memory control).
        shuffle_files:  Randomise file order each epoch.
        seed:           RNG seed for file shuffling.
    """

    def __init__(
        self,
        files: List[Path],
        text_column: str = "text",
        batch_size: int = 1024,
        shuffle_files: bool = True,
        seed: int = 42,
    ):
        self.files = files
        self.text_column = text_column
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[str]:
        files = list(self.files)
        if self.shuffle_files:
            self.rng.shuffle(files)

        for file_path in files:
            yield from self._stream_file(file_path)

    def _stream_file(self, path: Path) -> Iterator[str]:
        """Yield text strings from a single Parquet file."""
        try:
            yield from self._stream_pyarrow(path)
        except ImportError:
            yield from self._stream_polars(path)

    def _stream_pyarrow(self, path: Path) -> Iterator[str]:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=self.batch_size, columns=[self.text_column]):
            col = batch.column(self.text_column)
            for val in col:
                text = val.as_py()
                if text:
                    yield str(text)

    def _stream_polars(self, path: Path) -> Iterator[str]:
        import polars as pl

        df = pl.scan_parquet(str(path)).select(self.text_column)
        for row in df.collect().iter_rows():
            text = row[0]
            if text:
                yield str(text)

    @staticmethod
    def discover_files(parquet_dir: Union[str, Path]) -> List[Path]:
        """Recursively find all .parquet files under ``parquet_dir``."""
        root = Path(parquet_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Parquet directory does not exist: {root}")
        files = sorted(root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found under: {root}")
        logger.info(f"Discovered {len(files)} Parquet files in {root}")
        return files


class ParquetTextFactory:
    """Picklable stream factory for macOS DataLoader worker processes."""

    def __init__(self, files: List[Path], text_column: str, shuffle: bool, seed: int):
        self.files = files
        self.text_column = text_column
        self.shuffle = shuffle
        self.seed = seed

    def __call__(self) -> Iterator[str]:
        worker = torch.utils.data.get_worker_info()
        files = self.files
        if worker is not None:
            files = files[worker.id::worker.num_workers]
        return iter(ParquetStreamIterator(
            files=files,
            text_column=self.text_column,
            shuffle_files=self.shuffle,
            seed=self.seed + (worker.id if worker else 0),
        ))


# ============================================================
# 2. Token-block IterableDataset (streaming, no RAM limit)
# ============================================================

class TextDataset(IterableDataset):
    """
    An ``IterableDataset`` that reads a stream of texts, tokenises them,
    concatenates all tokens (with EOS in between), and yields fixed-size
    blocks of token IDs.

    This is the standard "pack-and-chunk" approach used by GPT-style training:
    no padding, maximum GPU utilisation.

    Args:
        text_iter_factory:  Zero-argument callable that returns a fresh iterator
                            of raw text strings.  Called once per epoch.
        tokenizer:          Any object with an ``encode(text) → List[int]`` method.
        context_length:     Token block size (= model context window).
        preprocessor:       Optional text preprocessor applied before tokenisation.
        eos_token_id:       Token ID inserted between documents.
        buffer_tokens:      How many tokens to buffer before yielding blocks.
        shuffle_buffer:     If > 0, shuffle within a rolling window of this many blocks.
        seed:               RNG seed for shuffle buffer.
    """

    def __init__(
        self,
        text_iter_factory: Callable[[], Iterable[str]],
        tokenizer: Any,
        context_length: int,
        preprocessor: Optional[TextPreprocessor] = None,
        eos_token_id: int = 2,
        buffer_tokens: int = 65_536,
        shuffle_buffer: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.text_iter_factory = text_iter_factory
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.preprocessor = preprocessor
        self.eos_token_id = eos_token_id
        self.buffer_tokens = max(buffer_tokens, context_length * 2)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        token_buffer: List[int] = []
        blocks: List[torch.Tensor] = []

        for text in self.text_iter_factory():
            # Optional preprocessing
            if self.preprocessor is not None:
                text = self.preprocessor.process(text)
                if text is None:
                    continue

            # Tokenise
            try:
                ids: List[int] = self.tokenizer.encode(text)
            except Exception:
                continue

            if not ids:
                continue

            token_buffer.extend(ids)
            token_buffer.append(self.eos_token_id)

            # Drain buffer into blocks
            while len(token_buffer) >= self.context_length + 1:
                block = token_buffer[: self.context_length + 1]
                token_buffer = token_buffer[self.context_length:]
                x = torch.tensor(block[:-1], dtype=torch.long)
                y = torch.tensor(block[1:],  dtype=torch.long)
                blocks.append((x, y))

                # Yield from shuffle buffer
                if len(blocks) >= self.shuffle_buffer:
                    if self.shuffle_buffer > 1:
                        rng.shuffle(blocks)
                    for b in blocks:
                        yield {"input_ids": b[0], "labels": b[1]}
                    blocks = []

        # Flush remaining blocks
        if blocks:
            if self.shuffle_buffer > 1:
                rng.shuffle(blocks)
            for b in blocks:
                yield {"input_ids": b[0], "labels": b[1]}


# ============================================================
# 3. Memory-mapped dataset (for pre-tokenised binary files)
# ============================================================

class MemoryMappedDataset(Dataset):
    """
    Fast map-style dataset over a pre-tokenised ``.bin`` file (uint16 array).

    Use ``scripts/tokenize_corpus.py`` to pre-tokenise the corpus, then use
    this class for subsequent training runs.  Supports random access, so it
    works with the standard ``RandomSampler``.

    File format: raw uint16 token IDs concatenated (no length prefix).
    """

    def __init__(self, bin_path: Union[str, Path], context_length: int):
        self.context_length = context_length
        data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self._data = data
        n_tokens = len(data)
        self._n_blocks = (n_tokens - 1) // context_length
        logger.info(
            f"MemoryMappedDataset: {n_tokens:,} tokens → {self._n_blocks:,} blocks "
            f"(context_length={context_length})"
        )

    def __len__(self) -> int:
        return self._n_blocks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.context_length
        block = self._data[start : start + self.context_length + 1].astype(np.int64)
        x = torch.from_numpy(block[:-1])
        y = torch.from_numpy(block[1:])
        return {"input_ids": x, "labels": y}


# ============================================================
# 4. Factory: build_dataloaders
# ============================================================

def build_dataloaders(
    cfg: Any,
    tokenizer: Any,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation ``DataLoader`` objects from config.

    Depending on whether a pre-tokenised binary exists, uses either
    ``MemoryMappedDataset`` (fast, map-style) or ``TextDataset`` (streaming).

    Args:
        cfg:        Config object (see ``config/default.yaml``).
        tokenizer:  Tokenizer with ``encode(str) → List[int]``.

    Returns:
        (train_loader, val_loader)
    """
    parquet_dir = Path(cfg.data.parquet_dir)
    context_length = cfg.model.context_length
    preprocessor = TextPreprocessor(
        min_length=cfg.data.min_text_length,
        max_length=cfg.data.max_text_length,
    )

    # --- Check for pre-tokenised binary ---
    train_bin = parquet_dir / "train.bin"
    val_bin = parquet_dir / "val.bin"

    if valid_manifest(cfg, tokenizer, parquet_dir, context_length):
        logger.info("Verified complete pre-tokenised binaries; using MemoryMappedDataset.")
        if cfg.data.num_workers != 0:
            raise ValueError("Exact coverage requires data.num_workers=0; worker prefetch can outrun committed updates")
        train_ds = MemoryMappedDataset(train_bin, context_length)
        val_ds = MemoryMappedDataset(val_bin, context_length)
        train_sampler = CoverageSampler(
            bin_path=train_bin,
            manifest_path=parquet_dir / "tokenized_manifest.json",
            n_blocks=len(train_ds),
            context_length=context_length,
            order_dir=Path(cfg.training.output_dir) / "shuffle_orders",
            seed=cfg.data.shuffle_seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.training.get("validation_batch_size", 4),
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        return train_loader, val_loader

    if train_bin.exists() or val_bin.exists():
        logger.warning("Pre-tokenised binaries lack a matching completion manifest; streaming Parquet instead.")

    # --- Streaming from Parquet ---
    logger.info("Streaming from Parquet files.")
    files = ParquetStreamIterator.discover_files(parquet_dir)

    # Train/val split by file index
    n_val = max(1, int(len(files) * cfg.data.val_split))
    rng = random.Random(cfg.data.shuffle_seed)
    rng.shuffle(files)
    val_files = files[:n_val]
    train_files = files[n_val:]

    logger.info(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    train_ds = TextDataset(
        text_iter_factory=ParquetTextFactory(train_files, cfg.data.text_column, True, cfg.data.shuffle_seed),
        tokenizer=tokenizer,
        context_length=context_length,
        preprocessor=preprocessor,
        eos_token_id=cfg.tokenizer.eos_token_id,
        shuffle_buffer=256,
        seed=cfg.data.shuffle_seed,
    )
    val_ds = TextDataset(
        text_iter_factory=ParquetTextFactory(val_files, cfg.data.text_column, False, cfg.data.shuffle_seed),
        tokenizer=tokenizer,
        context_length=context_length,
        preprocessor=preprocessor,
        eos_token_id=cfg.tokenizer.eos_token_id,
        shuffle_buffer=0,
        seed=cfg.data.shuffle_seed,
    )

    num_workers = min(cfg.data.num_workers, len(train_files))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=cfg.data.get("prefetch_factor", 2) if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.get("validation_batch_size", 4),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader
