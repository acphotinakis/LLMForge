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
import multiprocessing
import os
import platform
import queue
import resource
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from data.dataset import ParquetStreamIterator
from data.binary_manifest import (
    MANIFEST_NAME,
    manifest_for,
    source_records,
    split_sources,
    tokenizer_sha256,
)
from data.preprocessing import TextPreprocessor
from scripts.tokenize_workers import init_worker, tokenize_chunk, tokenize_file
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


def _estimated_documents(paths, max_docs_per_file):
    """Use Parquet row metadata for a progress estimate without scanning text."""
    import pyarrow.parquet as pq

    return sum(
        (
            min(pq.ParquetFile(path).metadata.num_rows, max_docs_per_file)
            if max_docs_per_file is not None
            else pq.ParquetFile(path).metadata.num_rows
        )
        for path in paths
    )


def _reconcile_progress(progress):
    # Empty/null text rows are skipped by ParquetStreamIterator, so the metadata
    # row count can be slightly higher than the documents actually emitted.
    if progress.total is not None and progress.n != progress.total:
        progress.total = progress.n


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


def _run_sequential(
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
    selected = split_sources(cfg, files)
    splits = selected.items()
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
            desc=f"Tokenising {split_name}",
            unit="doc",
            total=(
                _estimated_documents(selected_files, max_docs_per_file)
                if max_tokens is None
                else None
            ),
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
            _reconcile_progress(progress)

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


def _ordered_jobs(executor, fn, jobs, max_pending, on_progress=None):
    """Submit at most max_pending jobs and yield results in source order."""

    def wait_for(future):
        if on_progress is None:
            return future.result()
        while True:
            try:
                result = future.result(timeout=0.2)
                on_progress()
                return result
            except FutureTimeoutError:
                if future.done():
                    raise
                on_progress()

    pending = deque()
    for label, args in jobs:
        pending.append((label, executor.submit(fn, *args)))
        if len(pending) >= max_pending:
            label, future = pending.popleft()
            start = time.perf_counter()
            result = wait_for(future)
            yield label, result, time.perf_counter() - start
    while pending:
        label, future = pending.popleft()
        start = time.perf_counter()
        result = wait_for(future)
        yield label, result, time.perf_counter() - start


def _merge_part(result, output):
    part = Path(result["part_path"])
    if part.stat().st_size != result["tokens_written"] * 2:
        raise RuntimeError(f"Worker output size mismatch: {part}")
    start = time.perf_counter()
    with part.open("rb") as source:
        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    part.unlink()
    return time.perf_counter() - start


def _worker_row(name, result):
    row = _metrics(
        name,
        result["documents_seen"],
        result["documents_kept"],
        result["tokens_written"],
        result["wall_seconds"],
        result["process_cpu_seconds"],
        result["stages"],
    )
    row["write_calls"] = result["write_calls"]
    row["peak_rss_bytes"] = result["peak_rss_bytes"]
    return row


def _chunk_jobs(path, text_column, max_docs, chunk_docs, stage_dir, read_state):
    iterator = iter(
        ParquetStreamIterator(
            files=[path], text_column=text_column, shuffle_files=False
        )
    )
    chunk_index = 0
    while max_docs is None or read_state["documents"] < max_docs:
        texts = []
        while len(texts) < chunk_docs and (
            max_docs is None or read_state["documents"] < max_docs
        ):
            start = time.perf_counter()
            try:
                text = next(iterator)
            except StopIteration:
                read_state["seconds"] += time.perf_counter() - start
                break
            read_state["seconds"] += time.perf_counter() - start
            read_state["documents"] += 1
            texts.append(text)
        if not texts:
            break
        part = stage_dir / f"val-chunk-{path.stem}-{chunk_index:08d}.part"
        job_id = f"val:{path}:{chunk_index}"
        yield job_id, (texts, str(part), job_id)
        chunk_index += 1
        if len(texts) < chunk_docs:
            break


def _run_hybrid(
    cfg,
    output_dir,
    profile_path,
    max_docs_per_file,
    log_every_docs,
    workers,
    chunk_docs,
    max_pending,
    buffer_tokens,
):
    run_start = time.perf_counter()
    parent_cpu_start = time.process_time()
    tokenizer = ResearchTokenizer.load(
        cfg.tokenizer.model_path, backend=cfg.tokenizer.type
    )
    if tokenizer.vocab_size > 65536:
        raise ValueError("uint16 token binaries require vocab_size <= 65536")
    files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
    splits = split_sources(cfg, files)
    profile = {
        "schema_version": 1,
        "implementation": "hybrid-processes-buffered",
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
            "max_tokens_per_split": None,
            "max_docs_per_file": max_docs_per_file,
            "output_dir": str(output_dir),
            "workers": workers,
            "chunk_docs": chunk_docs,
            "max_pending": max_pending,
            "write_buffer_tokens": buffer_tokens,
        },
        "files": [],
        "splits": [],
    }
    _write_profile(profile_path, profile)
    total_wait = total_assembly = total_worker_cpu = 0.0
    peak_worker_rss = 0
    context = multiprocessing.get_context("spawn")
    with closing(context.Queue()) as progress_queue, ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(
            str(cfg.tokenizer.model_path),
            cfg.tokenizer.type,
            cfg.data.min_text_length,
            cfg.data.max_text_length,
            buffer_tokens,
            progress_queue,
            min(log_every_docs, 5_000),
        ),
    ) as executor:
        for split_name, paths in splits.items():
            split_start = time.perf_counter()
            split_seen = split_kept = split_tokens = split_writes = 0
            split_cpu = split_wait = split_assembly = 0.0
            split_stages = {stage: 0.0 for stage in STAGES}
            out_path = output_dir / f"{split_name}.bin"
            logger.info(
                "Tokenising %s split (%d files) with %d workers → %s",
                split_name,
                len(paths),
                workers,
                out_path,
            )
            with out_path.open("wb") as output, tqdm(
                total=_estimated_documents(paths, max_docs_per_file),
                desc=f"Tokenising {split_name}",
                unit="doc",
                unit_scale=True,
                mininterval=1,
                dynamic_ncols=True,
            ) as progress:
                reported = {}

                def advance(job_id, seen):
                    previous = reported.get(job_id, 0)
                    if seen > previous:
                        progress.update(seen - previous)
                        reported[job_id] = seen

                def drain_progress():
                    while True:
                        try:
                            job_id, seen = progress_queue.get_nowait()
                        except queue.Empty:
                            break
                        if job_id.startswith(f"{split_name}:"):
                            advance(job_id, seen)

                if split_name == "train":
                    jobs = (
                        (
                            f"train:{path}",
                            (
                                str(path),
                                cfg.data.text_column,
                                max_docs_per_file,
                                str(output_dir / f"train-{index:04d}.part"),
                                f"train:{path}",
                            ),
                        )
                        for index, path in enumerate(paths)
                    )
                    for job_id, result, wait in _ordered_jobs(
                        executor, tokenize_file, jobs, max_pending, drain_progress
                    ):
                        advance(job_id, result["documents_seen"])
                        path = Path(job_id.removeprefix("train:"))
                        assembly = _merge_part(result, output)
                        row = _worker_row(path.name, result)
                        row.update(
                            split=split_name,
                            input_parquet_file_bytes=path.stat().st_size,
                            reached_max_docs_per_file=max_docs_per_file is not None
                            and result["documents_seen"] >= max_docs_per_file,
                        )
                        profile["files"].append(row)
                        _log_metrics(f"{split_name}/{path.name}", row)
                        _write_profile(profile_path, profile)
                        split_wait += wait
                        split_assembly += assembly
                        split_seen += result["documents_seen"]
                        split_kept += result["documents_kept"]
                        split_tokens += result["tokens_written"]
                        split_writes += result["write_calls"]
                        split_cpu += result["process_cpu_seconds"]
                        peak_worker_rss = max(peak_worker_rss, result["peak_rss_bytes"])
                        for stage in STAGES:
                            split_stages[stage] += result["stages"][stage]
                else:
                    for path in paths:
                        file_start = time.perf_counter()
                        read_state = {"seconds": 0.0, "documents": 0}
                        file_results = []
                        next_log = log_every_docs
                        completed_docs = completed_tokens = 0
                        jobs = _chunk_jobs(
                            path,
                            cfg.data.text_column,
                            max_docs_per_file,
                            chunk_docs,
                            output_dir,
                            read_state,
                        )
                        for job_id, result, wait in _ordered_jobs(
                            executor, tokenize_chunk, jobs, max_pending, drain_progress
                        ):
                            advance(job_id, result["documents_seen"])
                            assembly = _merge_part(result, output)
                            file_results.append(result)
                            split_wait += wait
                            split_assembly += assembly
                            peak_worker_rss = max(
                                peak_worker_rss, result["peak_rss_bytes"]
                            )
                            completed_docs += result["documents_seen"]
                            completed_tokens += result["tokens_written"]
                            if completed_docs >= next_log:
                                logger.info(
                                    "%s/%s progress: docs=%s tokens=%s",
                                    split_name,
                                    path.name,
                                    f"{completed_docs:,}",
                                    f"{completed_tokens:,}",
                                )
                                next_log = (
                                    completed_docs // log_every_docs + 1
                                ) * log_every_docs
                        file_stages = {
                            stage: sum(r["stages"][stage] for r in file_results)
                            for stage in STAGES
                        }
                        file_stages["read_seconds"] += read_state["seconds"]
                        file_result = {
                            "documents_seen": read_state["documents"],
                            "documents_kept": sum(
                                r["documents_kept"] for r in file_results
                            ),
                            "tokens_written": sum(
                                r["tokens_written"] for r in file_results
                            ),
                            "write_calls": sum(r["write_calls"] for r in file_results),
                            "wall_seconds": time.perf_counter() - file_start,
                            "process_cpu_seconds": sum(
                                r["process_cpu_seconds"] for r in file_results
                            ),
                            "peak_rss_bytes": max(
                                (r["peak_rss_bytes"] for r in file_results), default=0
                            ),
                            "stages": file_stages,
                        }
                        row = _worker_row(path.name, file_result)
                        row.update(
                            split=split_name,
                            input_parquet_file_bytes=path.stat().st_size,
                            reached_max_docs_per_file=max_docs_per_file is not None
                            and read_state["documents"] >= max_docs_per_file,
                        )
                        profile["files"].append(row)
                        _log_metrics(f"{split_name}/{path.name}", row)
                        _write_profile(profile_path, profile)
                        split_seen += file_result["documents_seen"]
                        split_kept += file_result["documents_kept"]
                        split_tokens += file_result["tokens_written"]
                        split_writes += file_result["write_calls"]
                        split_cpu += file_result["process_cpu_seconds"]
                        for stage in STAGES:
                            split_stages[stage] += file_stages[stage]
                drain_progress()
                _reconcile_progress(progress)
            split_row = _metrics(
                split_name,
                split_seen,
                split_kept,
                split_tokens,
                time.perf_counter() - split_start,
                split_cpu,
                split_stages,
            )
            split_row.update(
                output_path=str(out_path),
                actual_output_bytes=out_path.stat().st_size,
                write_calls=split_writes,
                waiting_seconds=split_wait,
                assembly_seconds=split_assembly,
                reached_max_tokens=False,
            )
            if split_row["actual_output_bytes"] != split_row["output_bytes"]:
                raise RuntimeError(f"Output size mismatch for {out_path}")
            profile["splits"].append(split_row)
            _log_metrics(split_name, split_row)
            _write_profile(profile_path, profile)
            total_wait += split_wait
            total_assembly += split_assembly
            total_worker_cpu += split_cpu
    profile.update(
        completed=True,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        total_wall_seconds=time.perf_counter() - run_start,
        total_worker_cpu_seconds=total_worker_cpu,
        parent_cpu_seconds=time.process_time() - parent_cpu_start,
        total_waiting_seconds=total_wait,
        total_assembly_seconds=total_assembly,
        peak_worker_rss_bytes=peak_worker_rss,
        total_tokens_written=sum(row["tokens_written"] for row in profile["splits"]),
        total_documents_seen=sum(row["documents_seen"] for row in profile["splits"]),
    )
    profile["total_tokens_per_second"] = profile["total_tokens_written"] / max(
        profile["total_wall_seconds"], 1e-9
    )
    _write_profile(profile_path, profile)
    return profile


def run_tokenization(
    cfg,
    output_dir,
    profile_path,
    max_tokens=None,
    log_every_docs=50_000,
    max_docs_per_file=None,
    mode="sequential",
    workers=4,
    chunk_docs=1024,
    max_pending=8,
    write_buffer_tokens=100_000,
):
    """Create a staged pair and publish a manifest only for a full conversion."""
    if mode not in ("sequential", "hybrid"):
        raise ValueError("mode must be sequential or hybrid")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_docs_per_file is not None and max_docs_per_file <= 0:
        raise ValueError("max_docs_per_file must be positive")
    if max_tokens is not None and max_docs_per_file is not None:
        raise ValueError("Use either max_tokens or max_docs_per_file, not both")
    if mode == "hybrid" and max_tokens is not None:
        raise ValueError("Hybrid mode supports max_docs_per_file, not max_tokens")
    if min(workers, chunk_docs, max_pending, write_buffer_tokens, log_every_docs) <= 0:
        raise ValueError(
            "Worker, chunk, queue, buffer, and log limits must be positive"
        )
    output_dir = Path(output_dir)
    profile_path = Path(profile_path)
    if (
        max_tokens is not None or max_docs_per_file is not None
    ) and output_dir.resolve() == Path(cfg.data.parquet_dir).resolve():
        raise ValueError(
            "Bounded runs must use an output directory separate from data.parquet_dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    full = max_tokens is None and max_docs_per_file is None
    if full:
        initial_files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
        initial_splits = split_sources(cfg, initial_files)
        initial_sources = {
            name: source_records(paths, cfg.data.parquet_dir)
            for name, paths in initial_splits.items()
        }
        initial_tokenizer_sha = tokenizer_sha256(cfg.tokenizer.model_path)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tokenize-stage-", dir=output_dir
    ) as temporary:
        stage = Path(temporary)
        if mode == "sequential":
            profile = _run_sequential(
                cfg, stage, profile_path, max_tokens, log_every_docs, max_docs_per_file
            )
        else:
            profile = _run_hybrid(
                cfg,
                stage,
                profile_path,
                max_docs_per_file,
                log_every_docs,
                workers,
                chunk_docs,
                max_pending,
                write_buffer_tokens,
            )
        for row in profile["splits"]:
            name = row["name"]
            if (stage / f"{name}.bin").stat().st_size != row["output_bytes"]:
                raise RuntimeError(f"Staged {name}.bin size mismatch")
        if full:
            current_files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
            current_splits = split_sources(cfg, current_files)
            if initial_sources != {
                name: source_records(paths, cfg.data.parquet_dir)
                for name, paths in current_splits.items()
            } or initial_tokenizer_sha != tokenizer_sha256(cfg.tokenizer.model_path):
                raise RuntimeError(
                    "Parquet sources or tokenizer changed during tokenization"
                )
        for row in profile["splits"]:
            name = row["name"]
            (stage / f"{name}.bin").replace(output_dir / f"{name}.bin")
            row["output_path"] = str(output_dir / f"{name}.bin")
        profile["settings"]["output_dir"] = str(output_dir)
        if full:
            tokenizer = ResearchTokenizer.load(
                cfg.tokenizer.model_path, backend=cfg.tokenizer.type
            )
            split_rows = {row["name"]: row for row in profile["splits"]}
            manifest = manifest_for(cfg, tokenizer, current_splits, split_rows)
            _write_profile(manifest_path, manifest)
        profile["published"] = True
        profile["full_corpus"] = full
        _write_profile(profile_path, profile)
        return profile


def main(cfg=None):
    output_override = profile_override = max_tokens_override = log_every_override = None
    max_docs_override = None
    mode_override = workers_override = chunk_override = pending_override = (
        buffer_override
    ) = None
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
        parser.add_argument("--mode", choices=("sequential", "hybrid"), default=None)
        parser.add_argument("--workers", type=int, default=None)
        parser.add_argument("--chunk_docs", type=int, default=None)
        parser.add_argument("--max_pending", type=int, default=None)
        parser.add_argument("--write_buffer_tokens", type=int, default=None)
        args = parser.parse_args()
        setup_logging()
        cfg = resolve_model_config(load_config(args.config))
        output_override, profile_override = args.output_dir, args.profile_json
        max_tokens_override, log_every_override = args.max_tokens, args.log_every_docs
        max_docs_override = args.max_docs_per_file
        mode_override, workers_override = args.mode, args.workers
        chunk_override, pending_override = args.chunk_docs, args.max_pending
        buffer_override = args.write_buffer_tokens

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
    mode = mode_override or (
        settings.get("mode", "sequential") if settings else "sequential"
    )
    workers = workers_override or (settings.get("workers", 4) if settings else 4)
    chunk_docs = chunk_override or (
        settings.get("chunk_docs", 1024) if settings else 1024
    )
    max_pending = pending_override or (
        settings.get("max_pending", 8) if settings else 8
    )
    buffer_tokens = buffer_override or (
        settings.get("write_buffer_tokens", 100_000) if settings else 100_000
    )
    return run_tokenization(
        cfg,
        output_dir,
        profile_path,
        max_tokens,
        log_every_docs,
        max_docs_per_file,
        mode,
        workers,
        chunk_docs,
        max_pending,
        buffer_tokens,
    )


if __name__ == "__main__":
    main()
