"""Generation loss helpers for autoregressive SID prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GenLossOutput:
    loss: Tensor
    ce_loss: Tensor
    valid_tokens: int
    token_acc: float


def weighted_autoregressive_ce_loss(
    logits: dict[str, Tensor],
    target_sid: dict[str, Tensor],
    target_available: dict[str, Tensor] | None,
    branch_weights: dict[str, float],
    ignore_index: int = 0,
    logits_finite_verified: bool = False,
) -> GenLossOutput:
    """Compute branch-weighted L_rec normalized by each sample's available targets."""
    if not isinstance(logits, dict) or not isinstance(target_sid, dict):
        raise TypeError("weighted_autoregressive_ce_loss expects dict logits and target_sid")
    if set(logits) != set(target_sid):
        raise ValueError(f"logits branches {sorted(logits)} != target branches {sorted(target_sid)}")
    if target_available is not None and set(target_available) != set(logits):
        raise ValueError(f"target_available branches {sorted(target_available)} != logits branches {sorted(logits)}")
    if not logits:
        raise ValueError("branch logits dict is empty")

    first_logits = next(iter(logits.values()))
    batch_size = int(first_logits.shape[0])
    numerator = first_logits.float().new_zeros(batch_size)
    denominator = first_logits.float().new_zeros(batch_size)
    acc_numerator = first_logits.float().new_zeros(batch_size)
    total_valid_tokens = 0

    for branch in sorted(logits):
        weight = float(branch_weights.get(branch, 1.0))
        if weight < 0.0:
            raise ValueError(f"branch weight for {branch!r} must be non-negative, got {weight}")
        masked_target = _mask_target_by_availability(
            target_sid[branch],
            None if target_available is None else target_available[branch],
            ignore_index,
        )
        branch_loss, branch_acc, branch_valid = _per_sample_ce_loss(
            logits[branch],
            masked_target,
            ignore_index,
            check_logits_finite=not logits_finite_verified,
        )
        active = branch_valid > 0
        total_valid_tokens += int(branch_valid[active].sum().item())
        if weight == 0.0 or not active.any().item():
            continue
        branch_weight = torch.full_like(denominator, weight)
        numerator = numerator + branch_loss * branch_weight * active.float()
        acc_numerator = acc_numerator + branch_acc * branch_weight * active.float()
        denominator = denominator + branch_weight * active.float()

    valid_samples = denominator > 0
    if not valid_samples.any().item():
        return _zero_loss_output(first_logits)

    sample_loss = numerator[valid_samples] / denominator[valid_samples]
    sample_acc = acc_numerator[valid_samples] / denominator[valid_samples]
    loss = sample_loss.mean()
    return GenLossOutput(
        loss=loss,
        ce_loss=loss,
        valid_tokens=total_valid_tokens,
        token_acc=float(sample_acc.mean().item()),
    )


def _per_sample_ce_loss(
    logits: Tensor,
    target_sid: Tensor,
    ignore_index: int,
    check_logits_finite: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    logits_fp32, valid_mask = _validate_ce_inputs(
        logits,
        target_sid,
        ignore_index,
        check_logits_finite=check_logits_finite,
    )
    vocab_size = logits.shape[-1]
    token_loss = F.cross_entropy(
        logits_fp32.reshape(-1, vocab_size),
        target_sid.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape_as(target_sid)
    valid_float = valid_mask.float()
    valid_counts = valid_float.sum(dim=1)
    loss_sum = (token_loss * valid_float).sum(dim=1)
    sample_loss = torch.where(valid_counts > 0, loss_sum / valid_counts.clamp_min(1.0), loss_sum)
    with torch.no_grad():
        pred = logits_fp32.argmax(dim=-1)
        acc_sum = ((pred == target_sid) & valid_mask).float().sum(dim=1)
        sample_acc = torch.where(valid_counts > 0, acc_sum / valid_counts.clamp_min(1.0), acc_sum)
    return sample_loss, sample_acc, valid_counts.long()


def _validate_ce_inputs(
    logits: Tensor,
    target_sid: Tensor,
    ignore_index: int,
    check_logits_finite: bool = True,
) -> tuple[Tensor, Tensor]:
    if logits.dim() != 3:
        raise ValueError(f"logits shape must be (B, K, V), got {tuple(logits.shape)}")
    if target_sid.dim() != 2:
        raise ValueError(f"target_sid shape must be (B, K), got {tuple(target_sid.shape)}")
    if logits.shape[:2] != target_sid.shape:
        raise ValueError(f"logits and target_sid leading dims differ: {tuple(logits.shape)} vs {tuple(target_sid.shape)}")
    logits_fp32 = logits.float()
    if check_logits_finite and not torch.isfinite(logits_fp32).all().item():
        bad_count = int((~torch.isfinite(logits_fp32)).sum().item())
        raise ValueError(f"logits contain NaN/Inf values: {bad_count}")

    valid_mask = target_sid != ignore_index
    if valid_mask.any().item():
        valid_target = target_sid[valid_mask]
        if valid_target.min().item() < 0:
            raise ValueError(f"target_sid minimum valid value {valid_target.min().item()} is below 0")
        if valid_target.max().item() >= logits.shape[-1]:
            raise ValueError(
                f"target_sid maximum valid value {valid_target.max().item()} "
                f"exceeds logits classes {logits.shape[-1]}"
            )
    return logits_fp32, valid_mask


def _mask_target_by_availability(
    target_sid: Tensor,
    target_available: Tensor | None,
    ignore_index: int,
) -> Tensor:
    if target_available is None:
        return target_sid
    if target_sid.dim() != 2:
        raise ValueError(f"target_sid shape must be (B, K), got {tuple(target_sid.shape)}")
    if target_available.dim() != 1 or target_available.shape[0] != target_sid.shape[0]:
        raise ValueError(f"target_available shape must be ({target_sid.shape[0]},), got {tuple(target_available.shape)}")
    masked_target = target_sid.clone()
    masked_target[~target_available.bool()] = ignore_index
    return masked_target


def _zero_loss_output(reference: Tensor) -> GenLossOutput:
    zero = reference.float().sum() * 0.0
    return GenLossOutput(
        loss=zero,
        ce_loss=zero,
        valid_tokens=0,
        token_acc=0.0,
    )
