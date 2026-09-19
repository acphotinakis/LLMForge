"""Seed setting and device/dtype resolution utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_str: str = "mps") -> torch.device:
    """Use Apple's GPU by default; allow CPU for local debugging."""
    if device_str == "auto":
        device_str = "mps"
    if device_str not in ("mps", "cpu"):
        raise ValueError("system.device must be 'mps' or 'cpu'")
    device = torch.device(device_str)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this PyTorch installation or on this Mac")
    return device


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from: {list(mapping)}")
    return mapping[dtype_str]
