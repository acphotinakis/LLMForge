#!/usr/bin/env python3
"""
Research LLM — Main Entry Point
================================

Modes
-----
train
    Full end-to-end training pipeline:
      1. Load config
      2. Load / train tokenizer
      3. Build DataLoaders
      4. Build model
      5. Run training loop with eval and checkpointing

generate
    Interactive / batch text generation from a trained checkpoint.

tokenize
    Pre-tokenise the corpus to binary files (faster subsequent training).

Usage examples::

    # Training (will also train tokenizer if model_path doesn't exist)
    python main.py train --config config/default.yaml

    # Override specific config values from the CLI
    python main.py train --config config/default.yaml \\
        training.batch_size=16 model.n_layers=24 training.max_steps=50000

    # Generate text interactively
    python main.py generate \\
        --config config/default.yaml \\
        --checkpoint checkpoints/best.pt \\
        --prompt "Abstract: In this paper we propose"

    # Pre-tokenise corpus
    python main.py tokenize --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="research_llm",
        description="Research LLM — train, generate, or tokenize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode", choices=["train", "generate", "tokenize"], help="Operating mode."
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to YAML config file."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (generate mode, or resume for train).",
    )
    parser.add_argument(
        "--prompt", default=None, help="Text prompt for generation mode."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Max tokens to generate (generate mode).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (generate mode).",
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling (generate mode)."
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Nucleus sampling p (generate mode)."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens to stdout (generate mode).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive prompt loop (generate mode).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Config overrides, e.g. training.batch_size=16",
    )
    args = parser.parse_intermixed_args()

    # ------------------------------------------------------------------ #
    # Bootstrap                                                            #
    # ------------------------------------------------------------------ #
    from utils.config import load_config, apply_overrides, resolve_model_config
    from utils.logging_utils import setup_logging, get_logger
    from utils.seed import set_seed
    from utils.device import get_device, get_dtype

    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    if cfg.system.get("backend", None) is not None:
        raise ValueError("system.backend has been removed; use system.device=mps")
    if args.checkpoint:
        cfg["training"]["resume_from"] = args.checkpoint

    cfg = resolve_model_config(cfg)

    setup_logging(
        log_dir=cfg.training.log_dir,
        run_name=cfg.training.run_name,
    )
    logger = get_logger("main")
    set_seed(cfg.system.seed)

    # ------------------------------------------------------------------ #
    # MODE: tokenize                                                       #
    # ------------------------------------------------------------------ #
    if args.mode == "tokenize":
        logger.info("Mode: tokenize  |  Device: cpu")
        _run_tokenize(cfg)
        return

    device = get_device(cfg.system.device)
    dtype = get_dtype(cfg.training.dtype)
    logger.info(f"Mode: {args.mode}  |  Device: {device}  |  Dtype: {dtype}")

    # ------------------------------------------------------------------ #
    # MODE: train                                                          #
    # ------------------------------------------------------------------ #
    if args.mode == "train":
        _run_train(cfg, device, dtype, logger)
        return

    # ------------------------------------------------------------------ #
    # MODE: generate                                                       #
    # ------------------------------------------------------------------ #
    if args.mode == "generate":
        _run_generate(cfg, args, device, dtype, logger)
        return


# ====================================================================== #
# Sub-routines                                                            #
# ====================================================================== #


def _run_tokenize(cfg) -> None:
    """Pre-tokenise the Parquet corpus to binary files."""
    from scripts.tokenize_corpus import main as tok_main

    tok_main(cfg)


def _run_train(cfg, device, dtype, logger) -> None:
    from tokenizer.tokenizer import ResearchTokenizer
    from data.dataset import build_dataloaders
    from model.transformer import build_model

    # ---- Tokenizer ----
    tok_path = cfg.tokenizer.model_path
    if not Path(tok_path).exists():
        logger.info(f"Tokenizer not found at {tok_path}; training a new one …")
        _train_tokenizer(cfg)

    tokenizer = ResearchTokenizer.load(tok_path, backend=cfg.tokenizer.type)
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Patch vocab_size into model config
    cfg["model"]["vocab_size"] = tokenizer.vocab_size

    # ---- DataLoaders ----
    logger.info("Building DataLoaders …")
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)

    from training.trainer import Trainer

    # ---- Model ----
    logger.info("Building model …")
    model = build_model(cfg, vocab_size=tokenizer.vocab_size)
    model.to(device)
    n_params = model.num_parameters()

    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.stop_mps_profile()
        logger.info("Training interrupted by user.  Saving emergency checkpoint …")
        trainer.checkpoint_manager.save(
            step=trainer.global_step,
            model=model,
            optimizer=trainer.optimizer,
            extra_meta={"interrupted": True},
        )


def _train_tokenizer(cfg) -> None:
    """Train a new tokenizer from the corpus."""
    from data.dataset import ParquetStreamIterator
    from data.preprocessing import TextPreprocessor
    from tokenizer.tokenizer import ResearchTokenizer

    preprocessor = TextPreprocessor(
        min_length=cfg.data.min_text_length,
        max_length=cfg.data.max_text_length,
    )
    files = ParquetStreamIterator.discover_files(cfg.data.parquet_dir)
    stream = ParquetStreamIterator(
        files=files,
        text_column=cfg.data.text_column,
        shuffle_files=True,
    )

    def gen():
        for t in stream:
            c = preprocessor.process(t)
            if c:
                yield c

    tok_cfg = cfg.tokenizer
    ResearchTokenizer.train(
        texts=gen(),
        vocab_size=tok_cfg.vocab_size,
        model_path=tok_cfg.model_path,
        model_type=tok_cfg.get("model_type", "bpe"),
        character_coverage=tok_cfg.get("character_coverage", 0.9995),
        max_sentence_length=tok_cfg.get("max_sentence_length", 16384),
        num_threads=tok_cfg.get("num_threads", 16),
        max_texts=tok_cfg.get("train_on_n_docs", 2_000_000),
        backend=tok_cfg.type,
    )


def _run_generate(cfg, args, device, dtype, logger) -> None:
    from inference.generator import TextGenerator
    from model.transformer import ModelConfig

    checkpoint = args.checkpoint or cfg.inference.get("checkpoint_path", None)
    if not checkpoint:
        logger.error(
            "--checkpoint or cfg.inference.checkpoint_path is required for generate mode."
        )
        sys.exit(1)

    # Patch vocab size
    cfg["model"]["vocab_size"] = cfg.tokenizer.vocab_size

    model_cfg = ModelConfig.from_config(cfg)

    generator = TextGenerator(
        checkpoint_path=checkpoint,
        tokenizer_path=cfg.tokenizer.model_path,
        model_config=model_cfg,
        device_str=str(device),
        dtype_str=cfg.training.dtype,
        tokenizer_backend=cfg.tokenizer.type,
    )

    inf_cfg = cfg.inference
    max_new_tokens = args.max_new_tokens or inf_cfg.max_new_tokens
    temperature = args.temperature or inf_cfg.temperature
    top_k = args.top_k or inf_cfg.top_k
    top_p = args.top_p or inf_cfg.top_p

    if args.interactive:
        print("\nResearch LLM — Interactive Generation")
        print("Type a prompt and press Enter.  Ctrl-C to exit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if not prompt:
                    continue
                results = generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=inf_cfg.repetition_penalty,
                    stream=args.stream,
                )
                if not args.stream:
                    print(f"\n{results[0]}\n")
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        prompt = args.prompt or "The key finding of this research is that"
        results = generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=inf_cfg.repetition_penalty,
            stream=args.stream,
        )
        if not args.stream:
            print(results[0])


if __name__ == "__main__":
    main()
