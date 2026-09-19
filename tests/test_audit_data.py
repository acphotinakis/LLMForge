"""Checks that the audit counts and split match a small, known Parquet corpus."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_data import audit
from utils.config import Config


class DataAuditTest(unittest.TestCase):
    def test_inventory_split_duplicates_and_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            shared = "The same document appears in both shards."
            records = [
                [(shared, "a", 3), ("A different train or validation record.", "b", 5)],
                [
                    (shared, "c", 7),
                    ("Another distinct document for the audit.", "d", 11),
                ],
            ]
            for number, docs in enumerate(records):
                table = pa.table(
                    {
                        "text": [x[0] for x in docs],
                        "id": [x[1] for x in docs],
                        "url": [f"https://example.com/{x[1]}" for x in docs],
                        "language": ["en"] * 2,
                        "language_score": [0.9] * 2,
                        "token_count": [x[2] for x in docs],
                        "score": [3.0] * 2,
                        "int_score": [3] * 2,
                    }
                )
                pq.write_table(
                    table, folder / f"{number:03}_00000.parquet", row_group_size=2
                )

            cfg = Config(
                {
                    "data": {
                        "parquet_dir": str(folder),
                        "text_column": "text",
                        "train_split": 0.5,
                        "val_split": 0.5,
                        "min_text_length": 10,
                        "max_text_length": 100,
                        "shuffle_seed": 42,
                    },
                    "tokenizer": {
                        "model_path": str(folder / "missing.model"),
                        "type": "sentencepiece",
                        "train_on_n_docs": 2,
                        "max_sentence_length": 16,
                    },
                }
            )
            out = folder / "report"
            summary = audit(cfg, out, sample_per_shard=2, groups_per_shard=1)

            self.assertEqual(summary["total_rows_exact"], 4)
            self.assertEqual(summary["sample_count"], 4)
            self.assertEqual(summary["cross_split_text_groups"], 1)
            self.assertEqual(summary["cross_split_id_groups"], 0)
            self.assertEqual(summary["tokenizer_candidates_scanned"], 2)
            with (out / "split_summary.csv").open() as f:
                splits = list(csv.DictReader(f))
            self.assertEqual([int(x["rows_exact"]) for x in splits], [2, 2])
            self.assertEqual(sum(int(x["token_count_sum_exact"]) for x in splits), 26)
            self.assertIn(
                "Tokenizer-training provenance", (out / "report.html").read_text()
            )


if __name__ == "__main__":
    unittest.main()
