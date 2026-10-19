"""Feed-forward activations shared by MoSID's Transformer layers."""
from __future__ import annotations

from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor


def swiglu(projected: Tensor) -> Tensor:
    """Map packed gate/value projections of width 2F to SiLU(gate) * value."""
    gate, value = projected.chunk(2, dim=-1)
    return F.silu(gate) * value


def resolve_ff_activation(name: str) -> str | Callable[[Tensor], Tensor]:
    # Functions survive Transformer layer cloning, including decoder __setstate__.
    if name == "swiglu":
        return swiglu
    if name == "leaky_relu":
        return F.leaky_relu
    return name
