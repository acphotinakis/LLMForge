"""
Pre-tokenise a Parquet corpus and write memory-mapped binary files.

Usage::

    python scripts/tokenize_corpus.py \
        --config config/default.yaml \
        --output_dir data/parquet/

This produces:
  - data/parquet/train.bin   (uint16 token IDs)
  - data/parquet/val.bin     (uint16 token IDs)

These files can be loaded by ``MemoryMappedDataset`` for fast training
without on-the-fly tokenisation overhead.

Why uint16?  Modern vocabularies are ≤ 65535 tokens, fitting in 2 bytes.
A 1B-token corpus occupies ~2 GB — trivial to memory-map.
"""

import argparse
import json
import os
import platform
import random
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from data.dataset import ParquetStreamIterator
from data.preprocessing import TextPreprocessor
from tokenizer.tokenizer import ResearchTokenizer
from utils.config import load_config, resolve_model_config
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)

STAGES = (
    "read_seconds",
    "preprocess_seconds",
    "encode_seconds",
    "convert_seconds",
    "write_seconds",
)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _metrics(name, seen, kept, tokens, wall, cpu, stages):
    elapsed = max(wall, 1e-9)
    row = {
        "name": name,
        "documents_seen": seen,
        "documents_kept": kept,
        "documents_filtered": seen - kept,
        "tokens_written": tokens,
        "output_bytes": tokens * 2,
        "write_calls": kept,
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "average_cpu_cores": cpu / elapsed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "documents_per_second": seen / elapsed,
        "tokens_per_second": tokens / elapsed,
        "output_mib_per_second": tokens * 2 / elapsed / 2**20,
        "tokens_per_kept_document": tokens / kept if kept else 0.0,
    }
    row.update(stages)
    row["other_seconds"] = max(0.0, wall - sum(stages.values()))
    return row


def _write_profile(path, profile):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _log_metrics(label, row):
    logger.info(
        "%s: docs=%s kept=%s filtered=%s tokens=%s wall=%.2fs "
        "docs/s=%.0f tok/s=%.0f out=%.2f MiB/s CPU=%.2fs cores=%.2f peak_RSS=%.2f GiB "
        "read=%.2fs clean=%.2fs encode=%.2fs convert=%.2fs write=%.2fs other=%.2fs",
        label,
        f"{row['documents_seen']:,}",
        f"{row['documents_kept']:,}",
        f"{row['documents_filtered']:,}",
        f"{row['tokens_written']:,}",
        row["wall_seconds"],
        row["documents_per_second"],
        row["tokens_per_second"],
        row["output_mib_per_second"],
        row["process_cpu_seconds"],
        row["average_cpu_cores"],
        row["peak_rss_bytes"] / 2**30,
        row["read_seconds"],
        row["preprocess_seconds"],
        row["encode_seconds"],
        row["convert_seconds"],
        row["write_seconds"],
        row["other_seconds"],
    )


def run_tokenization(
    cfg,
    output_dir,
    profile_path,
    max_tokens=None,
    log_every_docs=50_000,
    max_docs_per_file=None,
):
    """Run the existing sequential algorithm and record a comparable baseline."""
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_docs_per_file is not None and max_docs_per_file <= 0:
        raise ValueError("max_docs_per_file must be positive")
    if max_tokens is not None and max_docs_per_file is not None:
        raise ValueError("Use either max_tokens or max_docs_per_file, not both")
    if log_every_docs <= 0:
        raise ValueError("log_every_docs must be positive")
    output_dir = Path(output_dir)
    profile_path = Path(profile_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()

    tokenizer = ResearchTokenizer.load(
        cfg.tokenizer.model_path, backend=cfg.tokenizer.type
    )
    if tokenizer.vocab_size > 65536:
        raise ValueError("uint16 token binaries require vocab_size <= 65536")
    preprocessor = TextPreprocessor(
        min_length=cfg.data.min_text_length,
        max_length=cfg.data.max_text_length,
    )
    files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
    random.Random(cfg.data.shuffle_seed).shuffle(files)
    n_val = max(1, int(len(files) * cfg.data.val_split))
    splits = (("train", files[n_val:]), ("val", files[:n_val]))
    profile = {
        "schema_version": 1,
        "implementation": "sequential-per-document",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": False,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "settings": {
            "parquet_dir": str(cfg.data.parquet_dir),
            "text_column": cfg.data.text_column,
            "min_text_length": cfg.data.min_text_length,
            "max_text_length": cfg.data.max_text_length,
            "val_split": cfg.data.val_split,
            "shuffle_seed": cfg.data.shuffle_seed,
            "tokenizer_path": str(cfg.tokenizer.model_path),
            "tokenizer_type": cfg.tokenizer.type,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "max_tokens_per_split": max_tokens,
            "max_docs_per_file": max_docs_per_file,
            "output_dir": str(output_dir),
        },
        "files": [],
        "splits": [],
    }
    _write_profile(profile_path, profile)

    for split_name, selected_files in splits:
        out_path = output_dir / f"{split_name}.bin"
        logger.info(
            "Tokenising %s split (%d files) → %s",
            split_name,
            len(selected_files),
            out_path,
        )
        split_start = time.perf_counter()
        split_cpu_start = time.process_time()
        split_docs = split_kept = split_tokens = 0
        split_stages = {stage: 0.0 for stage in STAGES}

        with out_path.open("wb") as output, tqdm(
            desc=f"Tokenising {split_name}", unit="doc"
        ) as progress:
            for path in selected_files:
                file_start = time.perf_counter()
                file_cpu_start = time.process_time()
                file_docs = file_kept = file_tokens = 0
                file_stages = {stage: 0.0 for stage in STAGES}
                log_start, log_tokens = file_start, 0
                stream = ParquetStreamIterator(
                    files=[path], text_column=cfg.data.text_column, shuffle_files=False
                )
                iterator = iter(stream)
                while True:
                    start = time.perf_counter()
                    try:
                        text = next(iterator)
                    except StopIteration:
                        file_stages["read_seconds"] += time.perf_counter() - start
                        break
                    file_stages["read_seconds"] += time.perf_counter() - start
                    file_docs += 1
                    progress.update(1)
                    reached_doc_limit = (
                        max_docs_per_file is not None and file_docs >= max_docs_per_file
                    )

                    start = time.perf_counter()
                    clean = preprocessor.process(text)
                    file_stages["preprocess_seconds"] += time.perf_counter() - start
                    if clean is None:
                        if reached_doc_limit:
                            break
                        continue

                    start = time.perf_counter()
                    ids = tokenizer.encode(clean)
                    ids.append(tokenizer.eos_token_id)
                    file_stages["encode_seconds"] += time.perf_counter() - start

                    start = time.perf_counter()
                    token_array = np.asarray(ids, dtype=np.uint16)
                    file_stages["convert_seconds"] += time.perf_counter() - start
                    start = time.perf_counter()
                    token_array.tofile(output)
                    file_stages["write_seconds"] += time.perf_counter() - start
                    file_kept += 1
                    file_tokens += len(ids)
                    split_tokens += len(ids)

                    if file_docs % log_every_docs == 0:
                        now = time.perf_counter()
                        logger.info(
                            "%s %s progress: docs=%s tokens=%s interval_tok/s=%.0f",
                            split_name,
                            path.name,
                            f"{file_docs:,}",
                            f"{file_tokens:,}",
                            (file_tokens - log_tokens) / max(now - log_start, 1e-9),
                        )
                        log_start, log_tokens = now, file_tokens

                    if max_tokens is not None and split_tokens >= max_tokens:
                        logger.info(
                            "Reached max_tokens=%s for %s split", max_tokens, split_name
                        )
                        break
                    if reached_doc_limit:
                        break

                file_row = _metrics(
                    path.name,
                    file_docs,
                    file_kept,
                    file_tokens,
                    time.perf_counter() - file_start,
                    time.process_time() - file_cpu_start,
                    file_stages,
                )
                file_row.update(
                    {
                        "split": split_name,
                        "input_parquet_file_bytes": path.stat().st_size,
                        "reached_max_docs_per_file": (
                            max_docs_per_file is not None
                            and file_docs >= max_docs_per_file
                        ),
                    }
                )
                profile["files"].append(file_row)
                _write_profile(profile_path, profile)
                _log_metrics(f"{split_name}/{path.name}", file_row)
                split_docs += file_docs
                split_kept += file_kept
                for stage in STAGES:
                    split_stages[stage] += file_stages[stage]
                if max_tokens is not None and split_tokens >= max_tokens:
                    break

        split_row = _metrics(
            split_name,
            split_docs,
            split_kept,
            split_tokens,
            time.perf_counter() - split_start,
            time.process_time() - split_cpu_start,
            split_stages,
        )
        split_row.update(
            {
                "output_path": str(out_path),
                "actual_output_bytes": out_path.stat().st_size,
                "reached_max_tokens": max_tokens is not None
                and split_tokens >= max_tokens,
            }
        )
        if split_row["actual_output_bytes"] != split_row["output_bytes"]:
            raise RuntimeError(f"Output size mismatch for {out_path}")
        profile["splits"].append(split_row)
        _write_profile(profile_path, profile)
        _log_metrics(split_name, split_row)

    profile["completed"] = True
    profile["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    profile["total_wall_seconds"] = time.perf_counter() - run_start
    profile["total_tokens_written"] = sum(
        row["tokens_written"] for row in profile["splits"]
    )
    profile["total_documents_seen"] = sum(
        row["documents_seen"] for row in profile["splits"]
    )
    profile["total_tokens_per_second"] = profile["total_tokens_written"] / max(
        profile["total_wall_seconds"], 1e-9
    )
    _write_profile(profile_path, profile)
    logger.info("Pre-tokenisation complete. Profile → %s", profile_path)
    return profile


def main(cfg=None):
    output_override = profile_override = max_tokens_override = log_every_override = None
    max_docs_override = None
    if cfg is None:
        parser = argparse.ArgumentParser(
            description="Pre-tokenise Parquet corpus to binary."
        )
        parser.add_argument("--config", default="config/default.yaml")
        parser.add_argument("--output_dir", default=None)
        parser.add_argument("--max_tokens", type=int, default=None)
        parser.add_argument("--max_docs_per_file", type=int, default=None)
        parser.add_argument("--profile_json", default=None)
        parser.add_argument("--log_every_docs", type=int, default=None)
        args = parser.parse_args()
        setup_logging()
        cfg = resolve_model_config(load_config(args.config))
        output_override, profile_override = args.output_dir, args.profile_json
        max_tokens_override, log_every_override = args.max_tokens, args.log_every_docs
        max_docs_override = args.max_docs_per_file

    settings = cfg.get("tokenization", None)
    output_dir = Path(
        output_override
        or (settings.get("output_dir") if settings else None)
        or cfg.data.parquet_dir
    )
    profile_path = Path(
        profile_override
        or (settings.get("profile_path") if settings else None)
        or output_dir / "tokenization_profile.json"
    )
    max_tokens = (
        max_tokens_override
        if max_tokens_override is not None
        else (settings.get("max_tokens") if settings else None)
    )
    log_every_docs = (
        log_every_override
        if log_every_override is not None
        else (settings.get("log_every_docs", 50_000) if settings else 50_000)
    )
    max_docs_per_file = (
        max_docs_override
        if max_docs_override is not None
        else (settings.get("max_docs_per_file") if settings else None)
    )
    return run_tokenization(
        cfg, output_dir, profile_path, max_tokens, log_every_docs, max_docs_per_file
    )


if __name__ == "__main__":
    main()
