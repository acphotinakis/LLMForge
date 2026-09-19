"""Completion record for a pair of pre-tokenized corpus binaries."""

import hashlib
import json
import random
from pathlib import Path

MANIFEST_NAME = "tokenized_manifest.json"


def tokenizer_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_sources(cfg, files):
    ordered = sorted(Path(path) for path in files)
    random.Random(cfg.data.shuffle_seed).shuffle(ordered)
    n_val = max(1, int(len(ordered) * cfg.data.val_split))
    return {"train": ordered[n_val:], "val": ordered[:n_val]}


def source_records(paths, root):
    root = Path(root).resolve()
    return [
        {
            "path": str(path.resolve().relative_to(root)),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def manifest_for(cfg, tokenizer, splits, split_rows):
    return {
        "schema_version": 1,
        "completed": True,
        "bounded": False,
        "tokenizer": {
            "type": cfg.tokenizer.type,
            "sha256": tokenizer_sha256(cfg.tokenizer.model_path),
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "data": {
            "text_column": cfg.data.text_column,
            "min_text_length": cfg.data.min_text_length,
            "max_text_length": cfg.data.max_text_length,
            "val_split": cfg.data.val_split,
            "shuffle_seed": cfg.data.shuffle_seed,
        },
        "splits": {
            name: {
                "sources": source_records(paths, cfg.data.parquet_dir),
                "tokens": split_rows[name]["tokens_written"],
                "bytes": split_rows[name]["actual_output_bytes"],
            }
            for name, paths in splits.items()
        },
    }


def valid_manifest(cfg, tokenizer, parquet_dir, context_length):
    """Return whether the full binaries match the active data and tokenizer."""
    root = Path(parquet_dir)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or not manifest.get("completed")
            or manifest.get("bounded")
        ):
            return False
        files = sorted(Path(cfg.data.parquet_dir).rglob("*.parquet"))
        if not files:
            return False
        splits = split_sources(cfg, files)
        if manifest["tokenizer"] != {
            "type": cfg.tokenizer.type,
            "sha256": tokenizer_sha256(cfg.tokenizer.model_path),
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id,
        }:
            return False
        if manifest["data"] != {
            "text_column": cfg.data.text_column,
            "min_text_length": cfg.data.min_text_length,
            "max_text_length": cfg.data.max_text_length,
            "val_split": cfg.data.val_split,
            "shuffle_seed": cfg.data.shuffle_seed,
        }:
            return False
        for name, paths in splits.items():
            row = manifest["splits"][name]
            binary = root / f"{name}.bin"
            if row["sources"] != source_records(paths, cfg.data.parquet_dir):
                return False
            if row["tokens"] <= context_length or row["bytes"] != row["tokens"] * 2:
                return False
            if binary.stat().st_size != row["bytes"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
