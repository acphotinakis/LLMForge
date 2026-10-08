"""End-to-end parity and completion checks for spawned tokenization workers."""

import json
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
import numpy as np

from data.binary_manifest import MANIFEST_NAME, valid_manifest
from data.dataset import MemoryMappedDataset, TextDataset, build_dataloaders
from scripts.tokenize_corpus import _ordered_jobs, run_tokenization
from tokenizer.tokenizer import ResearchTokenizer
from utils.config import Config


class HybridTokenizationTest(unittest.TestCase):
    def test_completed_jobs_are_assembled_in_submission_order(self):
        def task(index, delay):
            time.sleep(delay)
            return index

        jobs = [(0, (0, 0.05)), (1, (1, 0.0)), (2, (2, 0.0))]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(_ordered_jobs(executor, task, jobs, max_pending=2))
        self.assertEqual(
            [(label, result) for label, result, _ in results], [(0, 0), (1, 1), (2, 2)]
        )

    def test_parity_order_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "parquet"
            corpus.mkdir()
            lines = [
                "Alpha beta gamma delta epsilon.",
                "x",
                "This document has a different sequence of useful words.",
                "Another example contains some repeated words words words.",
                "Short",
                "Final valid document for this source file.",
                "One more valid document crosses the chunk boundary.",
            ]
            for index in range(4):
                text_rows = [
                    f"Shard {index}: {line}" if len(line) > 10 else line
                    for line in lines
                ] + ["", None]
                pq.write_table(
                    pa.table({"text": text_rows}),
                    corpus / f"{index:03}_00000.parquet",
                    row_group_size=2,
                )
            training_text = root / "tokenizer.txt"
            training_text.write_text("\n".join(lines * 20), encoding="utf-8")
            model_prefix = root / "spm"
            spm.SentencePieceTrainer.train(
                input=str(training_text),
                model_prefix=str(model_prefix),
                vocab_size=128,
                model_type="bpe",
                character_coverage=1.0,
                hard_vocab_limit=False,
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3,
            )
            cfg = Config(
                {
                    "data": {
                        "parquet_dir": str(corpus),
                        "text_column": "text",
                        "min_text_length": 20,
                        "max_text_length": 100,
                        "val_split": 0.25,
                        "shuffle_seed": 42,
                        "num_workers": 0,
                    },
                    "tokenizer": {
                        "model_path": str(model_prefix) + ".model",
                        "type": "sentencepiece",
                        "eos_token_id": 2,
                    },
                    "model": {"context_length": 8},
                    "training": {
                        "batch_size": 2,
                        "output_dir": str(root / "checkpoints"),
                    },
                }
            )
            baseline = root / "baseline"
            hybrid = root / "hybrid"
            for limit in (5, None):
                sequential_profile = run_tokenization(
                    cfg,
                    baseline,
                    baseline / "profile.json",
                    max_docs_per_file=limit,
                    mode="sequential",
                    log_every_docs=1000,
                )
                hybrid_profile = run_tokenization(
                    cfg,
                    hybrid,
                    hybrid / "profile.json",
                    max_docs_per_file=limit,
                    mode="hybrid",
                    workers=2,
                    chunk_docs=2,
                    max_pending=2,
                    write_buffer_tokens=8,
                )
                for split in ("train", "val"):
                    self.assertEqual(
                        (baseline / f"{split}.bin").read_bytes(),
                        (hybrid / f"{split}.bin").read_bytes(),
                    )
                self.assertEqual(
                    sequential_profile["total_tokens_written"],
                    hybrid_profile["total_tokens_written"],
                )
                self.assertEqual(
                    sequential_profile["total_documents_seen"],
                    hybrid_profile["total_documents_seen"],
                )
                self.assertEqual(
                    hybrid_profile["total_documents_seen"],
                    20 if limit is not None else 28,
                )
                for split in ("train", "val"):
                    token_ids = np.fromfile(hybrid / f"{split}.bin", dtype=np.uint16)
                    kept = next(
                        row for row in hybrid_profile["splits"] if row["name"] == split
                    )["documents_kept"]
                    self.assertEqual(int((token_ids == 2).sum()), kept)
                self.assertEqual((hybrid / MANIFEST_NAME).exists(), limit is None)
                self.assertEqual((baseline / MANIFEST_NAME).exists(), limit is None)
            tokenizer = ResearchTokenizer.load(cfg.tokenizer.model_path)
            self.assertTrue(valid_manifest(cfg, tokenizer, hybrid, 8))
            with self.assertRaisesRegex(ValueError, "Bounded runs"):
                run_tokenization(
                    cfg,
                    corpus,
                    corpus / "profile.json",
                    max_docs_per_file=2,
                    mode="hybrid",
                )
            alternate_model = root / "changed.model"
            alternate_model.write_bytes(
                Path(cfg.tokenizer.model_path).read_bytes() + b"changed"
            )
            original_model = cfg.tokenizer.model_path
            cfg.tokenizer.model_path = str(alternate_model)
            self.assertFalse(valid_manifest(cfg, tokenizer, hybrid, 8))
            cfg.tokenizer.model_path = original_model
            (hybrid / MANIFEST_NAME).unlink()
            self.assertFalse(valid_manifest(cfg, tokenizer, hybrid, 8))
            for name in ("train.bin", "val.bin"):
                shutil.copy2(hybrid / name, corpus / name)
            train_loader, _ = build_dataloaders(cfg, tokenizer)
            self.assertIsInstance(train_loader.dataset, TextDataset)
            shutil.copy2(baseline / MANIFEST_NAME, corpus / MANIFEST_NAME)
            train_loader, val_loader = build_dataloaders(cfg, tokenizer)
            self.assertIsInstance(train_loader.dataset, MemoryMappedDataset)
            self.assertIsInstance(val_loader.dataset, MemoryMappedDataset)
            self.assertEqual(next(iter(train_loader))["input_ids"].shape[1], 8)
            cfg.data.min_text_length += 1
            self.assertFalse(valid_manifest(cfg, tokenizer, corpus, 8))
            cfg.data.min_text_length -= 1
            manifest = json.loads((baseline / MANIFEST_NAME).read_text())
            self.assertFalse(manifest["bounded"])
            (corpus / "val.bin").write_bytes((corpus / "val.bin").read_bytes()[:-2])
            self.assertFalse(valid_manifest(cfg, tokenizer, corpus, 8))
            train_loader, _ = build_dataloaders(cfg, tokenizer)
            self.assertIsInstance(train_loader.dataset, TextDataset)


if __name__ == "__main__":
    unittest.main()
