"""Missing-aware sequence-memory orthogonality for MoSID Stage 2."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from utils.data import MPSID_INPUT_MODALITIES


MISSING_CONFIDENCE = 0.8
NUMERICAL_EPS = 1.0e-12

_CONTENT_PRESENCE_INDEX = {"text": 0, "vision": 1}


def memory_orth_loss(
    token_hidden_states: Mapping[str, Tensor],
    attention_mask: Tensor,
    history_presence: Tensor,
    direct: bool = False,
    center: bool = True,
    uniform_weights: bool = False,
) -> Tensor:
    """Compute residual orthogonality, or the mean of three direct memory pairs."""
    memories, attention, presence, sid_depth = _prepare_inputs(
        token_hidden_states,
        attention_mask,
        history_presence,
    )
    confidence = {
        "text": MISSING_CONFIDENCE
        + (1.0 - MISSING_CONFIDENCE) * presence[..., _CONTENT_PRESENCE_INDEX["text"]],
        "vision": MISSING_CONFIDENCE
        + (1.0 - MISSING_CONFIDENCE) * presence[..., _CONTENT_PRESENCE_INDEX["vision"]],
    }
    if uniform_weights:
        confidence = {branch: torch.ones_like(attention) for branch in ("text", "vision")}

    if direct:
        confidence["collab"] = torch.ones_like(attention)
        pair_losses = []
        for left, right in (("collab", "text"), ("collab", "vision"), ("text", "vision")):
            weights = _pair_token_weights(attention, confidence[left], confidence[right], sid_depth)
            sample_loss, valid = _weighted_pair_loss(memories[left], memories[right], weights, center=center)
            pair_losses.append(_mean_valid(sample_loss, valid))
        return torch.stack(pair_losses).mean()

    collab_anchor = memories["collab"].detach()
    text_residual = memories["text"] - collab_anchor
    vision_residual = memories["vision"] - collab_anchor
    weights = _pair_token_weights(
        attention,
        confidence["text"],
        confidence["vision"],
        sid_depth,
    )
    sample_loss, valid = _weighted_pair_loss(text_residual, vision_residual, weights, center=center)
    return _mean_valid(sample_loss, valid)


def _prepare_inputs(
    token_hidden_states: Mapping[str, Tensor],
    attention_mask: Tensor,
    history_presence: Tensor,
) -> tuple[dict[str, Tensor], Tensor, Tensor, int]:
    if not isinstance(token_hidden_states, Mapping):
        raise TypeError("token_hidden_states must be a mapping")
    required = set(MPSID_INPUT_MODALITIES)
    actual = set(token_hidden_states)
    if actual != required:
        raise ValueError(
            "token_hidden_states must define exactly "
            f"{sorted(required)!r}, got {sorted(str(key) for key in actual)!r}"
        )

    reference = token_hidden_states["collab"]
    if not isinstance(reference, Tensor):
        raise TypeError("token_hidden_states['collab'] must be a tensor")
    if reference.dim() != 4:
        raise ValueError(
            "token memories must have shape [B, L, K, H], "
            f"got collab={tuple(reference.shape)}"
        )
    expected_shape = tuple(reference.shape)
    reference_device = reference.device
    for branch in MPSID_INPUT_MODALITIES:
        memory = token_hidden_states[branch]
        if not isinstance(memory, Tensor):
            raise TypeError(f"token_hidden_states[{branch!r}] must be a tensor")
        if tuple(memory.shape) != expected_shape:
            raise ValueError(
                f"all token memories must share shape {expected_shape}, "
                f"got {branch}={tuple(memory.shape)}"
            )
        if not memory.is_floating_point():
            raise TypeError(f"token_hidden_states[{branch!r}] must be floating point")
        if memory.device != reference_device:
            raise ValueError("all token memories must be on the same device")

    if not isinstance(attention_mask, Tensor):
        raise TypeError("attention_mask must be a tensor")
    if not isinstance(history_presence, Tensor):
        raise TypeError("history_presence must be a tensor")
    batch_size, seq_len, sid_depth, hidden_dim = expected_shape
    if tuple(attention_mask.shape) != (batch_size, seq_len):
        raise ValueError(
            f"attention_mask must have shape {(batch_size, seq_len)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if tuple(history_presence.shape) != (batch_size, seq_len, 2):
        raise ValueError(
            f"history_presence must have shape {(batch_size, seq_len, 2)}, "
            f"got {tuple(history_presence.shape)}"
        )
    if attention_mask.dtype != torch.bool:
        raise TypeError(f"attention_mask must have dtype bool, got {attention_mask.dtype}")
    if history_presence.dtype != torch.bool:
        raise TypeError(f"history_presence must have dtype bool, got {history_presence.dtype}")
    if attention_mask.device != reference_device or history_presence.device != reference_device:
        raise ValueError("memory, attention_mask, and history_presence must be on the same device")

    flat_memories = {
        branch: token_hidden_states[branch].float().reshape(batch_size, seq_len * sid_depth, hidden_dim)
        for branch in MPSID_INPUT_MODALITIES
    }
    return flat_memories, attention_mask.float(), history_presence.float(), sid_depth


def _pair_token_weights(
    attention: Tensor,
    left_confidence: Tensor,
    right_confidence: Tensor,
    sid_depth: int,
) -> Tensor:
    item_weights = attention * left_confidence * right_confidence
    return item_weights.unsqueeze(-1).expand(-1, -1, sid_depth).reshape(item_weights.shape[0], -1)


def _weighted_pair_loss(left: Tensor, right: Tensor, weights: Tensor, center: bool = True) -> tuple[Tensor, Tensor]:
    expanded_weights = weights.unsqueeze(-1)
    centered_left, centered_right = left, right
    if center:
        weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(NUMERICAL_EPS)
        left_mean = (left * expanded_weights).sum(dim=1) / weight_sum
        right_mean = (right * expanded_weights).sum(dim=1) / weight_sum
        centered_left = left - left_mean.unsqueeze(1)
        centered_right = right - right_mean.unsqueeze(1)

    inner_product = (centered_left * centered_right * expanded_weights).sum(dim=(1, 2))
    left_norm = (centered_left.square() * expanded_weights).sum(dim=(1, 2))
    right_norm = (centered_right.square() * expanded_weights).sum(dim=(1, 2))
    valid = (left_norm > NUMERICAL_EPS) & (right_norm > NUMERICAL_EPS)
    squared_cosine = inner_product.square() / (left_norm * right_norm + NUMERICAL_EPS)
    return torch.where(valid, squared_cosine, torch.zeros_like(squared_cosine)), valid


def _mean_valid(losses: Tensor, valid: Tensor) -> Tensor:
    valid_float = valid.to(losses.dtype)
    return (losses * valid_float).sum() / valid_float.sum().clamp_min(1.0)
