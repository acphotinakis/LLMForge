"""Verify that profiling preserves binary token order and reports actual work."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tokenize_corpus import run_tokenization
from utils.config import Config


class FakeTokenizer:
    vocab_size = 100
    eos_token_id = 2

    def encode(self, text):
        return [len(text)]


class TokenizeProfileTest(unittest.TestCase):
    def test_profile_and_uint16_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for index in range(2):
                pq.write_table(
                    pa.table({"text": ["alpha", "x", "beta"]}),
                    data / f"{index:03}_00000.parquet",
                )
            cfg = Config(
                {
                    "data": {
                        "parquet_dir": str(data),
                        "text_column": "text",
                        "min_text_length": 2,
                        "max_text_length": 100,
                        "shuffle_seed": 42,
                        "val_split": 0.5,
                    },
                    "tokenizer": {
                        "model_path": str(root / "fake.model"),
                        "type": "sentencepiece",
                    },
                }
            )
            output = root / "output"
            profile_path = output / "profile.json"
            with patch(
                "scripts.tokenize_corpus.ResearchTokenizer.load",
                return_value=FakeTokenizer(),
            ):
                profile = run_tokenization(
                    cfg, output, profile_path, max_tokens=3, log_every_docs=2
                )

            self.assertTrue(profile["completed"])
            self.assertEqual(len(profile["splits"]), 2)
            self.assertEqual(len(profile["files"]), 2)
            self.assertEqual(profile["total_tokens_written"], 8)
            for split in ("train", "val"):
                self.assertEqual(
                    np.fromfile(output / f"{split}.bin", dtype=np.uint16).tolist(),
                    [5, 2, 4, 2],
                )
            self.assertEqual(
                [
                    (
                        row["documents_seen"],
                        row["documents_kept"],
                        row["documents_filtered"],
                    )
                    for row in profile["splits"]
                ],
                [(3, 2, 1), (3, 2, 1)],
            )
            self.assertEqual(json.loads(profile_path.read_text())["schema_version"], 1)

            sampled_output = root / "sampled"
            with patch(
                "scripts.tokenize_corpus.ResearchTokenizer.load",
                return_value=FakeTokenizer(),
            ):
                sampled = run_tokenization(
                    cfg,
                    sampled_output,
                    sampled_output / "profile.json",
                    max_docs_per_file=2,
                )
            self.assertEqual(
                [row["documents_seen"] for row in sampled["files"]], [2, 2]
            )
            self.assertEqual(sampled["total_tokens_written"], 4)
            for split in ("train", "val"):
                self.assertEqual(
                    np.fromfile(
                        sampled_output / f"{split}.bin", dtype=np.uint16
                    ).tolist(),
                    [5, 2],
                )

            with self.assertRaises(ValueError):
                run_tokenization(
                    cfg,
                    root / "invalid",
                    root / "invalid" / "profile.json",
                    max_tokens=3,
                    max_docs_per_file=2,
                )


if __name__ == "__main__":
    unittest.main()
