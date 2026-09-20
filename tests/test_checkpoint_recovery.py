"""Checkpoint publication and recovery from interrupted writes."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import torch

from utils.checkpoint import CheckpointManager


def _model_and_optimizer():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


class CheckpointRecoveryTests(TestCase):
    def test_latest_skips_incomplete_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer = _model_and_optimizer()
            manager = CheckpointManager(str(root))
            complete = manager.save(100, model, optimizer)
            incomplete = root / "checkpoint-step-00000101"
            incomplete.mkdir()
            torch.save(model.state_dict(), incomplete / "model.pt")
            (incomplete / "optimizer.pt").write_bytes(b"interrupted save")

            recovered = CheckpointManager(str(root))
            self.assertEqual(recovered.latest_checkpoint(), complete)
            recovered.load(str(complete), model, optimizer, device=torch.device("cpu"))

    def test_failed_save_does_not_publish_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer = _model_and_optimizer()
            manager = CheckpointManager(str(root))
            complete = manager.save(100, model, optimizer)
            real_save = torch.save

            def interrupted_save(obj, path):
                if Path(path).name == "optimizer.pt":
                    raise KeyboardInterrupt()
                return real_save(obj, path)

            with patch("torch.save", side_effect=interrupted_save):
                with self.assertRaises(KeyboardInterrupt):
                    manager.save(101, model, optimizer)

            self.assertEqual(manager.latest_checkpoint(), complete)
            self.assertFalse((root / "checkpoint-step-00000101").exists())
            self.assertFalse(list(root.glob(".checkpoint-stage-*")))

    def test_saving_same_step_replaces_complete_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, optimizer = _model_and_optimizer()
            manager = CheckpointManager(str(root))
            first = manager.save(100, model, optimizer, val_loss=2.0)
            second = manager.save(100, model, optimizer, val_loss=1.5)

            self.assertEqual(first, second)
            self.assertEqual(len(manager._checkpoints), 1)
            self.assertEqual(CheckpointManager(str(root)).latest_checkpoint(), second)
            self.assertFalse(list(root.glob(".checkpoint-backup-*")))


if __name__ == "__main__":
    main()
