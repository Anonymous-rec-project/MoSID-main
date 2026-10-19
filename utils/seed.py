"""Random seeds and device selection."""
from __future__ import annotations

import os
import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Seed random generators and enable deterministic algorithms."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def set_device(gpu_id: int | list[int]) -> torch.device:
    """Select the primary CUDA device, or CPU when CUDA is unavailable or the device list is empty."""

    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    if not gpu_id or not torch.cuda.is_available():
        return torch.device("cpu")

    primary = gpu_id[0]
    torch.cuda.set_device(primary)
    return torch.device(f"cuda:{primary}")
