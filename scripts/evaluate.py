"""
Evaluate a trained model on a text file or Parquet split.

Usage::

    python scripts/evaluate.py \
        --config config/default.yaml \
        --checkpoint checkpoints/best.pt \
        --text "path/to/test.txt"

    # Or evaluate on the validation split:
    python scripts/evaluate.py \
        --config config/default.yaml \
        --checkpoint checkpoints/best.pt
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from inference.generator import TextGenerator
from utils.config import load_config, resolve_model_config, apply_overrides
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def evaluate_on_file(generator: TextGenerator, text_path: str) -> float:
    """Compute perplexity on every paragraph of a text file."""
    text = Path(text_path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    total_loss = 0.0
    count = 0
    for para in tqdm(paragraphs, desc="Evaluating"):
        try:
            ids = generator.tokenizer.encode(para, add_bos=True)
            if len(ids) < 4:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=generator.device)
            y = torch.tensor([ids[1:]],  dtype=torch.long, device=generator.device)
            import torch.nn.functional as F
            with torch.no_grad(), torch.autocast(
                device_type=generator.device.type,
                dtype=generator.dtype,
                enabled=generator.use_amp,
            ):
                logits, _ = generator.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            total_loss += loss.item()
            count += 1
        except Exception as exc:
            logger.warning(f"Skipping paragraph: {exc}")
    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--text", default=None, help="Optional .txt file to evaluate.")
    parser.add_argument("--overrides", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    cfg = resolve_model_config(cfg)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    from model.transformer import ModelConfig
    model_cfg = ModelConfig.from_config(cfg)
    model_cfg.vocab_size = cfg.tokenizer.vocab_size

    generator = TextGenerator(
        checkpoint_path=args.checkpoint,
        tokenizer_path=cfg.tokenizer.model_path,
        model_config=model_cfg,
        device_str=cfg.system.device,
        dtype_str=cfg.training.dtype,
        tokenizer_backend=cfg.tokenizer.type,
    )

    if args.text:
        avg_loss = evaluate_on_file(generator, args.text)
    else:
        logger.info("No --text provided; running a quick self-test with sample sentences.")
        sample = (
            "Transformer architectures have revolutionised natural language processing. "
            "The attention mechanism allows models to relate tokens across long distances. "
            "Pre-training on large corpora followed by fine-tuning has become the dominant paradigm."
        )
        ppl = generator.perplexity(sample)
        logger.info(f"Sample perplexity: {ppl:.2f}")
        return

    ppl = math.exp(min(avg_loss, 20))
    logger.info(f"Average loss: {avg_loss:.4f}  |  Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
