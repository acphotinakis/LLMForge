"""Small CPU checks for training metrics and best-model bookkeeping."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import torch
from torch.utils.data import DataLoader

from model.transformer import GPTModel, ModelConfig
from training.trainer import Trainer
from utils.config import Config


def _trainer(root: Path) -> Trainer:
    cfg = Config(
        {
            "training": {
                "output_dir": str(root / "checkpoints"),
                "log_dir": str(root / "logs"),
                "run_name": "trainer_test",
                "keep_last_n_checkpoints": 2,
                "learning_rate": 1e-3,
                "min_lr": 1e-4,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
                "scheduler": "constant_with_warmup",
                "warmup_steps": 0,
                "max_steps": 2,
                "batch_size": 2,
                "grad_accumulation_steps": 2,
                "grad_clip": 1.0,
                "log_every_n_steps": 1,
                "eval_every_n_steps": 1,
                "eval_steps": 1,
                "generate_every_n_steps": 0,
                "save_every_n_steps": 2,
            },
        }
    )
    model = GPTModel(
        ModelConfig(
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            d_ff=32,
            context_length=4,
            dropout=0.0,
        )
    )
    sample = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([2, 3, 4, 5]),
    }
    loader = DataLoader([sample] * 8, batch_size=2)
    return Trainer(model, cfg, loader, loader, None, torch.device("cpu"), torch.float32)


class TrainerMetricsTests(TestCase):
    def test_microbatch_losses_remain_detached_tensors(self):
        with TemporaryDirectory() as directory:
            trainer = _trainer(Path(directory))
            trainer._train_iter = iter(trainer.train_loader)
            losses = trainer._accumulate_gradients()
            self.assertEqual(len(losses), 2)
            self.assertTrue(
                all(loss.ndim == 0 and not loss.requires_grad for loss in losses)
            )
            trainer.metrics_logger.close()

    def test_best_weights_match_evaluated_step_and_survive_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _trainer(root)
            measured_weights = {}
            scores = iter((1.0, 2.0))

            def evaluate():
                if not measured_weights:
                    measured_weights.update(
                        {
                            name: tensor.detach().clone()
                            for name, tensor in trainer.model.state_dict().items()
                        }
                    )
                return next(scores)

            trainer.evaluate = evaluate
            trainer.train()

            output = root / "checkpoints"
            best_meta = json.loads((output / "best_meta.json").read_text())
            regular_meta = json.loads(
                (output / "checkpoint-step-00000002" / "meta.json").read_text()
            )
            best_weights = torch.load(
                output / "best.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(best_meta, {"step": 1, "val_loss": 1.0})
            self.assertEqual(regular_meta["val_loss"], 2.0)
            self.assertTrue(
                all(
                    torch.equal(best_weights[name], value)
                    for name, value in measured_weights.items()
                )
            )

            resumed = _trainer(root)
            self.assertEqual(resumed.best_val_loss, 1.0)
            resumed.metrics_logger.close()


if __name__ == "__main__":
    main()
