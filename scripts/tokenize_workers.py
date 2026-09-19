"""Spawn-safe workers for ordered, bounded corpus tokenization."""

import resource
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np

from data.dataset import ParquetStreamIterator
from data.preprocessing import TextPreprocessor
from tokenizer.tokenizer import ResearchTokenizer

_tokenizer = None
_preprocessor = None
_buffer_tokens = None
_progress_queue = None
_report_every_docs = None


def init_worker(
    model_path,
    backend,
    min_length,
    max_length,
    buffer_tokens,
    progress_queue,
    report_every_docs,
):
    global _tokenizer, _preprocessor, _buffer_tokens, _progress_queue, _report_every_docs
    _tokenizer = ResearchTokenizer.load(model_path, backend=backend)
    if _tokenizer.vocab_size > 65536:
        raise ValueError("uint16 token binaries require vocab_size <= 65536")
    _preprocessor = TextPreprocessor(min_length=min_length, max_length=max_length)
    _buffer_tokens = buffer_tokens
    _progress_queue = progress_queue
    _report_every_docs = report_every_docs


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _encode_to_part(texts, out_path, job_id, read_seconds=0.0):
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    stages = dict(
        read_seconds=read_seconds,
        preprocess_seconds=0.0,
        encode_seconds=0.0,
        convert_seconds=0.0,
        write_seconds=0.0,
    )
    seen = kept = tokens = writes = 0
    buffer = []
    with Path(out_path).open("wb") as output:
        for text in texts:
            seen += 1
            stamp = time.perf_counter()
            clean = _preprocessor.process(text)
            stages["preprocess_seconds"] += time.perf_counter() - stamp
            if clean is None:
                if seen % _report_every_docs == 0:
                    _progress_queue.put((job_id, seen))
                continue
            stamp = time.perf_counter()
            ids = _tokenizer.encode(clean)
            ids.append(_tokenizer.eos_token_id)
            stages["encode_seconds"] += time.perf_counter() - stamp
            buffer.extend(ids)
            tokens += len(ids)
            kept += 1
            if len(buffer) >= _buffer_tokens:
                stamp = time.perf_counter()
                array = np.asarray(buffer, dtype=np.uint16)
                stages["convert_seconds"] += time.perf_counter() - stamp
                stamp = time.perf_counter()
                array.tofile(output)
                stages["write_seconds"] += time.perf_counter() - stamp
                writes += 1
                buffer.clear()
            if seen % _report_every_docs == 0:
                _progress_queue.put((job_id, seen))
        if buffer:
            stamp = time.perf_counter()
            array = np.asarray(buffer, dtype=np.uint16)
            stages["convert_seconds"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            array.tofile(output)
            stages["write_seconds"] += time.perf_counter() - stamp
            writes += 1
    return {
        "documents_seen": seen,
        "documents_kept": kept,
        "tokens_written": tokens,
        "write_calls": writes,
        "wall_seconds": time.perf_counter() - start_wall,
        "process_cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_bytes": _peak_rss_bytes(),
        "stages": stages,
        "part_path": str(out_path),
    }


def tokenize_file(path, text_column, max_docs, out_path, job_id):
    stream = ParquetStreamIterator(
        files=[Path(path)], text_column=text_column, shuffle_files=False
    )

    def texts():
        yield from (stream if max_docs is None else islice(stream, max_docs))

    # Record the read/decode time separately from preprocessing and encoding.
    read = [0.0]

    def timed_texts():
        iterator = iter(texts())
        while True:
            stamp = time.perf_counter()
            try:
                value = next(iterator)
            except StopIteration:
                read[0] += time.perf_counter() - stamp
                return
            read[0] += time.perf_counter() - stamp
            yield value

    result = _encode_to_part(timed_texts(), out_path, job_id)
    result["stages"]["read_seconds"] = read[0]
    return result


def tokenize_chunk(texts, out_path, job_id):
    return _encode_to_part(texts, out_path, job_id)
