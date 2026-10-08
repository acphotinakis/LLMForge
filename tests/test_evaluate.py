"""Unit tests for scripts/evaluate.py token-weighted perplexity evaluation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model.transformer import GPTModel, ModelConfig
from scripts.evaluate import evaluate_on_file


class DummyTokenizer:
    """Deterministic token generator for testing."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.bos_token_id = 1

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        tokens = [self.bos_token_id] if add_bos else []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 2
            tokens.append(self.vocab[word])
        return tokens


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.context_length = 4
        self.model_cfg = ModelConfig(
            vocab_size=64,
            n_layers=1,
            n_heads=2,
            d_model=8,
            d_ff=16,
            context_length=self.context_length,
            dropout=0.0,
        )
        self.model = GPTModel(self.model_cfg)
        self.tokenizer = DummyTokenizer()
        self.generator = SimpleNamespace(
            model=self.model,
            tokenizer=self.tokenizer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            use_amp=False,
        )

    def test_token_weighted_aggregation_vs_manual_reference(self):
        """Verify token-weighted cross-entropy matches manual reference across variable-length paragraphs."""
        para_short = "alpha beta"  # 2 words -> 3 tokens with BOS -> 2 target tokens
        para_long = "gamma delta epsilon zeta eta theta"  # 6 words -> 7 tokens with BOS -> 6 target tokens

        content = f"{para_short}\n\n{para_long}"
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Manual calculation reference
            self.model.eval()
            total_nll = 0.0
            total_tokens = 0
            para_means = []

            for para in [para_short, para_long]:
                ids = self.tokenizer.encode(para, add_bos=True)
                p_nll = 0.0
                p_tokens = 0
                for start in range(0, len(ids) - 1, self.context_length):
                    chunk = ids[start : start + self.context_length + 1]
                    x = torch.tensor([chunk[:-1]], dtype=torch.long)
                    y = torch.tensor([chunk[1:]], dtype=torch.long)
                    with torch.inference_mode():
                        logits, _ = self.model(x)
                        chunk_nll = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            y.reshape(-1),
                            reduction="sum",
                        ).item()
                    p_nll += chunk_nll
                    p_tokens += y.numel()

                para_means.append(p_nll / p_tokens)
                total_nll += p_nll
                total_tokens += p_tokens

            expected_weighted_loss = total_nll / total_tokens
            macro_average_loss = sum(para_means) / len(para_means)

            # Ensure the test data has differing paragraph lengths such that macro and weighted differ
            self.assertNotAlmostEqual(
                expected_weighted_loss,
                macro_average_loss,
                places=3,
                msg="Macro-average and token-weighted loss should differ on unequal paragraph lengths",
            )

            actual_loss = evaluate_on_file(self.generator, temp_path)
            self.assertAlmostEqual(actual_loss, expected_weighted_loss, places=6)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_sequences_longer_than_model_context(self):
        """Sequences exceeding model context are chunked without errors and fully evaluated."""
        # 11 words + 1 BOS = 12 tokens > context_length (4)
        words = ["word" + str(i) for i in range(11)]
        content = " ".join(words)

        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Without chunking, passing 12 tokens to context_length=4 would fail.
            loss = evaluate_on_file(self.generator, temp_path)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            self.assertGreater(loss, 0.0)

            # Manual verification that all 11 target tokens were evaluated
            ids = self.tokenizer.encode(content, add_bos=True)
            self.assertEqual(len(ids), 12)
            total_nll = 0.0
            total_tokens = 0
            for start in range(0, len(ids) - 1, self.context_length):
                chunk = ids[start : start + self.context_length + 1]
                x = torch.tensor([chunk[:-1]], dtype=torch.long)
                y = torch.tensor([chunk[1:]], dtype=torch.long)
                with torch.inference_mode():
                    logits, _ = self.model(x)
                    total_nll += F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        reduction="sum",
                    ).item()
                total_tokens += y.numel()

            self.assertEqual(total_tokens, 11)
            self.assertAlmostEqual(loss, total_nll / total_tokens, places=6)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_input_raises_value_error(self):
        """Empty files or files with no evaluable tokens raise ValueError."""
        # Completely empty file
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write("")
            empty_path = f.name

        # Whitespace only
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write("   \n\n   \n\t  ")
            ws_path = f.name

        # Paragraph with only 1 token (BOS without any subsequent token, if encode produces len < 2)
        # Using a custom tokenizer mock returning single token
        single_token_gen = SimpleNamespace(
            model=self.model,
            tokenizer=SimpleNamespace(encode=lambda text, add_bos=True: [1]),
            device=torch.device("cpu"),
            dtype=torch.float32,
            use_amp=False,
        )
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f:
            f.write("single")
            single_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx1:
                evaluate_on_file(self.generator, empty_path)
            self.assertIn("No tokens available", str(ctx1.exception))

            with self.assertRaises(ValueError) as ctx2:
                evaluate_on_file(self.generator, ws_path)
            self.assertIn("No tokens available", str(ctx2.exception))

            with self.assertRaises(ValueError) as ctx3:
                evaluate_on_file(single_token_gen, single_path)
            self.assertIn("No tokens available", str(ctx3.exception))
        finally:
            Path(empty_path).unlink(missing_ok=True)
            Path(ws_path).unlink(missing_ok=True)
            Path(single_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
