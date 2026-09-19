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
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from research_llm.data.dataset import ParquetStreamIterator
from research_llm.data.preprocessing import TextPreprocessor
from research_llm.tokenizer.tokenizer import ResearchTokenizer
from research_llm.utils.config import load_config, resolve_model_config
from research_llm.utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenise Parquet corpus to binary.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Stop after N tokens.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    cfg = resolve_model_config(cfg)

    output_dir = Path(args.output_dir or cfg.data.parquet_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = ResearchTokenizer.load(
        cfg.tokenizer.model_path,
        backend=cfg.tokenizer.type,
    )
    preprocessor = TextPreprocessor(
        min_length=cfg.data.min_text_length,
        max_length=cfg.data.max_text_length,
    )

    files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
    import random
    rng = random.Random(cfg.data.shuffle_seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * cfg.data.val_split))
    val_files = files[:n_val]
    train_files = files[n_val:]

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        out_path = output_dir / f"{split_name}.bin"
        logger.info(f"Tokenising {split_name} split ({len(split_files)} files) → {out_path}")

        token_chunks = []
        total = 0

        stream = ParquetStreamIterator(
            files=split_files,
            text_column=cfg.data.text_column,
            shuffle_files=False,
        )

        for text in tqdm(stream, desc=f"Tokenising {split_name}", unit="doc"):
            clean = preprocessor.process(text)
            if clean is None:
                continue
            ids = tokenizer.encode(clean)
            ids.append(tokenizer.eos_token_id)
            token_chunks.append(np.array(ids, dtype=np.uint16))
            total += len(ids)

            if args.max_tokens and total >= args.max_tokens:
                logger.info(f"Reached max_tokens={args.max_tokens}; stopping early.")
                break

        all_tokens = np.concatenate(token_chunks)
        all_tokens.tofile(out_path)
        logger.info(
            f"Wrote {total:,} tokens ({total * 2 / 1e9:.2f} GB) to {out_path}"
        )

    logger.info("Pre-tokenisation complete.")


if __name__ == "__main__":
    main()
