"""
Train a new tokenizer vocabulary on the Parquet corpus.

Usage::

    python scripts/train_tokenizer.py --config config/default.yaml

This streams text from the Parquet files, applies preprocessing, and trains
a SentencePiece or HuggingFace BPE model, saving it to the path specified
in the config (``tokenizer.model_path``).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_llm.data.dataset import ParquetStreamIterator
from research_llm.data.preprocessing import TextPreprocessor
from research_llm.tokenizer.tokenizer import ResearchTokenizer
from research_llm.utils.config import load_config, apply_overrides
from research_llm.utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer vocabulary.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. tokenizer.vocab_size=64000")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    tok_cfg = cfg.tokenizer
    preprocessor = TextPreprocessor(
        min_length=cfg.data.min_text_length,
        max_length=cfg.data.max_text_length,
    )

    files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
    stream = ParquetStreamIterator(
        files=files,
        text_column=cfg.data.text_column,
        shuffle_files=True,
        seed=cfg.data.shuffle_seed,
    )

    def text_generator():
        for text in stream:
            clean = preprocessor.process(text)
            if clean is not None:
                yield clean

    logger.info(
        f"Training {tok_cfg.type} tokenizer "
        f"(vocab_size={tok_cfg.vocab_size}) on corpus …"
    )
    ResearchTokenizer.train(
        texts=text_generator(),
        vocab_size=tok_cfg.vocab_size,
        model_path=tok_cfg.model_path,
        model_type=tok_cfg.get("model_type", "bpe"),
        character_coverage=tok_cfg.get("character_coverage", 0.9995),
        max_sentence_length=tok_cfg.get("max_sentence_length", 16384),
        num_threads=tok_cfg.get("num_threads", 16),
        max_texts=tok_cfg.get("train_on_n_docs", 2_000_000),
        backend=tok_cfg.type,
    )
    logger.info(f"Tokenizer saved to {tok_cfg.model_path}")


if __name__ == "__main__":
    main()
