"""Compare CPU DataLoader throughput for resumable and standard shuffling.

This isolates input delivery; it does not claim an MPS training speedup.
Run when training is stopped so disk and memory contention do not skew results.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.coverage_sampler import CoverageSampler
from data.dataset import MemoryMappedDataset
from utils.config import load_config, resolve_model_config


def measure(loader: DataLoader, batches: int, context: int) -> dict:
    start = time.perf_counter()
    iterator = iter(loader)
    first_batch_seconds = None
    blocks = 0
    for i in range(batches):
        batch = next(iterator)
        blocks += batch["input_ids"].shape[0]
        if i == 0:
            first_batch_seconds = time.perf_counter() - start
    elapsed = time.perf_counter() - start
    return {
        "blocks": blocks,
        "first_batch_seconds": round(first_batch_seconds or 0.0, 4),
        "elapsed_seconds": round(elapsed, 4),
        "input_tokens_per_second": round(blocks * context / elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--blocks", type=int, default=1_000_000)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    cfg = resolve_model_config(load_config(args.config))
    root = Path(cfg.data.parquet_dir)
    binary = root / "train.bin"
    manifest = root / "tokenized_manifest.json"
    if not binary.is_file() or not manifest.is_file():
        parser.error("Complete train.bin and tokenized_manifest.json are required")
    dataset = MemoryMappedDataset(binary, cfg.model.context_length)
    n_blocks = min(args.blocks, len(dataset))
    if n_blocks < args.batches * args.batch_size:
        parser.error("--blocks must cover --batches × --batch-size")
    subset = Subset(dataset, range(n_blocks))
    generator = torch.Generator().manual_seed(cfg.data.shuffle_seed)
    reference = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    with tempfile.TemporaryDirectory(prefix="mini-gpt-sampler-benchmark-") as directory:
        sampler = CoverageSampler(
            binary,
            manifest,
            n_blocks,
            cfg.model.context_length,
            Path(directory) / "orders",
            cfg.data.shuffle_seed,
        )
        tracked = DataLoader(
            subset, batch_size=args.batch_size, sampler=sampler, num_workers=0
        )
        reference_result = measure(reference, args.batches, cfg.model.context_length)
        tracked_cold = measure(tracked, args.batches, cfg.model.context_length)
        # Re-open from position zero with the already generated order.
        warm_sampler = CoverageSampler(
            binary,
            manifest,
            n_blocks,
            cfg.model.context_length,
            Path(directory) / "orders",
            cfg.data.shuffle_seed,
        )
        warm = DataLoader(
            subset, batch_size=args.batch_size, sampler=warm_sampler, num_workers=0
        )
        tracked_warm = measure(warm, args.batches, cfg.model.context_length)
    print(
        json.dumps(
            {
                "blocks_in_benchmark": n_blocks,
                "batches_measured": args.batches,
                "batch_size": args.batch_size,
                "standard_shuffle": reference_result,
                "coverage_shuffle_cold": tracked_cold,
                "coverage_shuffle_warm": tracked_warm,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
