"""Training and inference precision controls."""
from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def amp_enabled(config, device: torch.device) -> bool:
    """Check whether CUDA autocast is enabled."""
    if device.type != "cuda":
        return False
    if "precision" not in config:
        return False
    return bool(getattr(config.precision, "amp", False))


def amp_dtype(config) -> torch.dtype:
    """Resolve the configured autocast dtype."""
    dtype_name = str(getattr(config.precision, "dtype", "bf16")).lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unknown precision.dtype: {dtype_name}")


def autocast_context(config, device: torch.device) -> ContextManager:
    """Return an autocast context appropriate for the device."""
    if not amp_enabled(config, device):
        return nullcontext()

    dtype = amp_dtype(config)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The current CUDA device does not support BF16 autocast")
    return torch.autocast(device_type="cuda", dtype=dtype)


def precision_summary(config, device: torch.device) -> str:
    """Return a readable precision summary."""
    if not amp_enabled(config, device):
        return "fp32"
    dtype = amp_dtype(config)
    return "bf16 autocast" if dtype is torch.bfloat16 else "fp16 autocast"
