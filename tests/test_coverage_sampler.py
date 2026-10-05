"""Exact block coverage and stop/resume behavior on a small binary."""

import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.coverage_sampler import CoverageSampler
from data.dataset import MemoryMappedDataset
from model.transformer import GPTModel, ModelConfig
from training.trainer import Trainer
from utils.config import Config


def _binary(root: Path, n_blocks: int = 10, context: int = 4) -> tuple[Path, Path]:
    path = root / "train.bin"
    np.arange(n_blocks * context + 1, dtype=np.uint16).tofile(path)
    manifest = root / "tokenized_manifest.json"
    manifest.write_text(json.dumps({"completed": True, "tokens": n_blocks * context + 1}))
    return path, manifest


def _sampler(root: Path, bin_path: Path, manifest: Path, n_blocks: int = 10) -> CoverageSampler:
    return CoverageSampler(bin_path, manifest, n_blocks, 4, root / "orders", seed=42)


def _trainer(root: Path, bin_path: Path, manifest: Path, max_steps: int, history: list[int]) -> Trainer:
    torch.manual_seed(7)
    cfg = Config({"training": {
        "output_dir": str(root / "checkpoints"),
        "log_dir": str(root / "logs"),
        "run_name": "coverage_test",
        "keep_last_n_checkpoints": 3,
        "learning_rate": 1e-3, "min_lr": 1e-4, "weight_decay": 0.0,
        "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
        "scheduler": "constant_with_warmup", "warmup_steps": 0,
        "max_steps": max_steps, "batch_size": 4,
        "grad_accumulation_steps": 1, "grad_clip": 1.0,
        "log_every_n_steps": 1, "eval_every_n_steps": 100,
        "eval_steps": 1, "generate_every_n_steps": 0,
        "save_every_n_steps": 2,
    }})
    sampler = _sampler(root, bin_path, manifest)
    train = DataLoader(MemoryMappedDataset(bin_path, 4), batch_size=4, sampler=sampler, num_workers=0)
    val = DataLoader(MemoryMappedDataset(bin_path, 4), batch_size=4, shuffle=False, num_workers=0)
    model = GPTModel(ModelConfig(
        vocab_size=64, n_layers=1, n_heads=2, d_model=8, d_ff=32,
        context_length=4, dropout=0.0,
    ))
    original_forward = model.forward

    def record_forward(input_ids, labels=None, mask=None):
        if model.training:
            history.extend((input_ids[:, 0] // 4).tolist())
        return original_forward(input_ids, labels=labels, mask=mask)

    model.forward = record_forward
    return Trainer(model, cfg, train, val, None, torch.device("cpu"), torch.float32)


class CoverageSamplerTests(TestCase):
    def test_every_block_once_and_resume_from_committed_cursor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary, manifest = _binary(root)
            reference = list(itertools.islice(iter(_sampler(root, binary, manifest)), 24))
            self.assertEqual(sorted(reference[:10]), list(range(10)))
            self.assertEqual(sorted(reference[10:20]), list(range(10)))

            sampler = _sampler(root, binary, manifest)
            iterator = iter(sampler)
            self.assertEqual([next(iterator) for _ in range(4)], reference[:4])
            sampler.commit(4)
            state = sampler.state_dict()
            [next(iterator) for _ in range(2)]  # interrupted, never committed
            with self.assertRaises(RuntimeError):
                sampler.state_dict()

            resumed = _sampler(root, binary, manifest)
            resumed.load_state_dict(state)
            self.assertEqual(list(itertools.islice(iter(resumed), 20)), reference[4:24])

    def test_dataloader_batch_crosses_epoch_without_dropping_tail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary, manifest = _binary(root)
            sampler = _sampler(root, binary, manifest)
            loader = DataLoader(MemoryMappedDataset(binary, 4), batch_size=4, sampler=sampler,
                                num_workers=0, drop_last=False)
            iterator = iter(loader)
            seen = []
            for _ in range(3):
                seen.extend((next(iterator)["input_ids"][:, 0] // 4).tolist())
                sampler.commit(4)
            self.assertEqual(sorted(seen[:10]), list(range(10)))
            self.assertEqual(sampler.progress()["epoch"], 1)
            self.assertEqual(sampler.progress()["blocks_seen_this_epoch"], 2)

    def test_changed_binary_or_shuffle_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary, manifest = _binary(root)
            sampler = _sampler(root, binary, manifest)
            state = sampler.state_dict()
            manifest.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                _sampler(root, binary, manifest).load_state_dict(state)

            manifest.write_text(json.dumps({"completed": True, "tokens": 41}))
            order_file = next((root / "orders").glob("*.u32"))
            with open(order_file, "r+b") as file:
                file.write(b"\xff")
            with self.assertRaises(ValueError):
                _sampler(root, binary, manifest).load_state_dict(state)

    def test_training_resume_matches_uninterrupted_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary, manifest = _binary(root)
            straight_history: list[int] = []
            straight = _trainer(root / "straight", binary, manifest, 4, straight_history)
            straight.train()

            resumed_history: list[int] = []
            first = _trainer(root / "resumed", binary, manifest, 2, resumed_history)
            first.train()
            second = _trainer(root / "resumed", binary, manifest, 4, resumed_history)
            second.train()

            self.assertEqual(resumed_history, straight_history)
            self.assertEqual(sorted(straight_history[:10]), list(range(10)))
            self.assertTrue(all(
                torch.equal(straight.model.state_dict()[name], value)
                for name, value in second.model.state_dict().items()
            ))
            self.assertEqual(json.loads((root / "resumed" / "checkpoints" /
                                         "checkpoint-step-00000004" / "meta.json").read_text())
                             ["train_sampler"]["committed_position"], 16)

    def test_interrupted_optimizer_update_does_not_publish_cursor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary, manifest = _binary(root)
            trainer = _trainer(root, binary, manifest, 2, [])
            trainer.train()
            resumed = _trainer(root, binary, manifest, 4, [])
            real_step = resumed.optimizer.step

            def interrupted_step(*args, **kwargs):
                real_step(*args, **kwargs)
                raise KeyboardInterrupt()

            resumed.optimizer.step = interrupted_step
            with self.assertRaises(KeyboardInterrupt):
                resumed.train()
            self.assertFalse(resumed.save_emergency_checkpoint())
            self.assertEqual(resumed.checkpoint_manager.latest_checkpoint().name,
                             "checkpoint-step-00000002")
            resumed.metrics_logger.close()


if __name__ == "__main__":
    main()
