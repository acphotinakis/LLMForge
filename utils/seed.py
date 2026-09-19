"""Seed setting and device/dtype resolution utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string.

    "auto" → CUDA if available, else MPS (Apple Silicon), else CPU.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


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


def supports_bfloat16(device: torch.device) -> bool:
    """Check if the device natively supports BF16."""
    if device.type == "cuda":
        # BF16 requires Ampere (compute capability 8.0+) or newer
        cap = torch.cuda.get_device_capability(device)
        return cap[0] >= 8
    # CPU and MPS support BF16 via software path
    return True
