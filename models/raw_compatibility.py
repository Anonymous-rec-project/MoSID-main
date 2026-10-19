"""Raw item compatibility scoring and availability-aware listwise loss."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.data import MPSID_INPUT_MODALITIES


@dataclass(frozen=True)
class RawBranchDiagnostics:
    valid_samples: int
    valid_candidates: int
    positive_margin: float
    top1: float


@dataclass
class RawCompatibilityOutput:
    loss: Tensor
    scores: dict[str, Tensor]
    aggregated_scores: Tensor
    valid_candidate_mask: Tensor
    modality_weights: Tensor
    diagnostics: dict[str, RawBranchDiagnostics]
    valid_samples: int


def unique_target_candidates(target_item: Tensor) -> tuple[Tensor, Tensor]:
    """Return unique target item ids and the positive column of every sample."""
    if not isinstance(target_item, Tensor):
        raise TypeError("target_item must be a tensor")
    if target_item.dim() != 1 or target_item.numel() == 0:
        raise ValueError(f"target_item must be a non-empty [B] tensor, got {tuple(target_item.shape)}")
    if target_item.dtype == torch.bool or target_item.is_floating_point() or target_item.is_complex():
        raise TypeError(f"target_item must use an integer dtype, got {target_item.dtype}")
    target_item = target_item.long()
    if int(target_item.min().item()) <= 0:
        raise ValueError("target_item ids must be positive")

    candidate_ids, target_columns = torch.unique(
        target_item,
        sorted=False,
        return_inverse=True,
    )
    expected_candidates = int(target_columns.max().item()) + 1
    if candidate_ids.numel() != expected_candidates:
        raise RuntimeError("unique target columns do not cover every candidate")
    if not torch.equal(candidate_ids.index_select(0, target_columns), target_item):
        raise RuntimeError("inverse target columns do not map back to target_item")
    return candidate_ids, target_columns


class RawCompatibilityScorer(nn.Module):
    """Project frozen branch raw embeddings and score them against BOS summaries."""

    def __init__(
        self,
        raw_dimensions: Mapping[str, int],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        expected = set(MPSID_INPUT_MODALITIES)
        if set(raw_dimensions) != expected:
            raise ValueError(
                f"raw_dimensions must define exactly {MPSID_INPUT_MODALITIES!r}, "
                f"got {sorted(raw_dimensions)!r}"
            )
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.raw_dimensions = {
            branch: int(raw_dimensions[branch])
            for branch in MPSID_INPUT_MODALITIES
        }
        for branch, dimension in self.raw_dimensions.items():
            if dimension <= 0:
                raise ValueError(f"raw dimension for {branch!r} must be positive, got {dimension}")
        self.projectors = nn.ModuleDict(
            {
                branch: nn.Linear(
                    self.raw_dimensions[branch],
                    self.hidden_dim,
                    bias=False,
                )
                for branch in MPSID_INPUT_MODALITIES
            }
        )
        self.raw_modality_logits = nn.Parameter(
            torch.zeros(len(MPSID_INPUT_MODALITIES), dtype=torch.float32)
        )

    def modality_weights(self) -> Tensor:
        """Return positive global modality weights in canonical branch order."""
        return F.softmax(self.raw_modality_logits.float(), dim=0)

    def score_branch(
        self,
        branch: str,
        query: Tensor,
        candidate_raw: Tensor,
    ) -> Tensor:
        """Return the float32 cosine score matrix ``[B, C]`` for one branch."""
        if branch not in self.projectors:
            raise ValueError(
                f"unknown raw compatibility branch {branch!r}; "
                f"expected one of {MPSID_INPUT_MODALITIES!r}"
            )
        self._validate_score_inputs(branch, query, candidate_raw)
        projector = self.projectors[branch]
        frozen_raw = candidate_raw.detach().to(dtype=projector.weight.dtype)
        candidate = projector(frozen_raw)
        normalized_query = F.normalize(query.float(), dim=-1)
        normalized_candidate = F.normalize(candidate.float(), dim=-1)
        return normalized_query @ normalized_candidate.transpose(0, 1)

    def score_aligned_branch(
        self,
        branch: str,
        query: Tensor,
        candidate_raw: Tensor,
    ) -> Tensor:
        """Score per-sample candidate rows with shapes ``[B, H]`` and ``[B, C, D]``."""
        if branch not in self.projectors:
            raise ValueError(
                f"unknown raw compatibility branch {branch!r}; "
                f"expected one of {MPSID_INPUT_MODALITIES!r}"
            )
        if query.dim() != 2 or query.shape[1] != self.hidden_dim:
            raise ValueError(
                f"query[{branch!r}] shape must be [B, {self.hidden_dim}], "
                f"got {tuple(query.shape)}"
            )
        expected_raw_dim = self.raw_dimensions[branch]
        if candidate_raw.dim() != 3 or candidate_raw.shape[2] != expected_raw_dim:
            raise ValueError(
                f"candidate_raw[{branch!r}] shape must be [B, C, {expected_raw_dim}], "
                f"got {tuple(candidate_raw.shape)}"
            )
        if candidate_raw.shape[0] != query.shape[0]:
            raise ValueError(
                f"candidate_raw[{branch!r}] batch size {candidate_raw.shape[0]} "
                f"!= query batch size {query.shape[0]}"
            )
        if not query.is_floating_point() or not candidate_raw.is_floating_point():
            raise TypeError(f"query and candidate_raw for {branch!r} must be floating tensors")
        if query.device != candidate_raw.device:
            raise ValueError(
                f"query and candidate_raw for {branch!r} must share a device, "
                f"got {query.device} and {candidate_raw.device}"
            )

        projector = self.projectors[branch]
        frozen_raw = candidate_raw.detach().to(dtype=projector.weight.dtype)
        candidate = projector(frozen_raw)
        normalized_query = F.normalize(query.float(), dim=-1)
        normalized_candidate = F.normalize(candidate.float(), dim=-1)
        return torch.bmm(
            normalized_query.unsqueeze(1),
            normalized_candidate.transpose(1, 2),
        ).squeeze(1)

    def forward(
        self,
        queries: Mapping[str, Tensor],
        candidate_raw: Mapping[str, Tensor],
        candidate_available: Mapping[str, Tensor],
        history_reliability: Mapping[str, Tensor],
        target_columns: Tensor,
        temperature: float,
    ) -> RawCompatibilityOutput:
        self._validate_branch_mappings(
            queries,
            candidate_raw,
            candidate_available,
            history_reliability,
        )
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be finite and positive, got {temperature}")

        batch_size, candidate_count, device = self._validate_loss_inputs(
            queries,
            candidate_raw,
            candidate_available,
            history_reliability,
            target_columns,
        )
        target_columns = target_columns.long()
        score_numerator = torch.zeros(
            (batch_size, candidate_count),
            dtype=torch.float32,
            device=device,
        )
        score_denominator = torch.zeros_like(score_numerator)
        modality_weights = self.modality_weights()
        scores: dict[str, Tensor] = {}
        diagnostic_tensors: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}

        for branch_index, branch in enumerate(MPSID_INPUT_MODALITIES):
            branch_scores = self.score_branch(
                branch,
                queries[branch],
                candidate_raw[branch],
            )
            scores[branch] = branch_scores
            availability = candidate_available[branch]
            reliability = history_reliability[branch].float()
            effective_gate = (
                reliability.unsqueeze(1)
                * availability.float().unsqueeze(0)
                * modality_weights[branch_index]
            )
            score_numerator = score_numerator + branch_scores * effective_gate
            score_denominator = score_denominator + effective_gate

            valid_candidate_count = availability.sum()
            positive_available = availability.index_select(0, target_columns)
            valid_rows = reliability.gt(0.0) & positive_available & (valid_candidate_count >= 2)
            diagnostic_tensors[branch] = self._branch_diagnostic_tensors(
                branch_scores,
                availability,
                target_columns,
                valid_rows,
            )

        valid_candidate_mask = score_denominator > 0.0
        aggregated_scores = torch.where(
            valid_candidate_mask,
            score_numerator / score_denominator.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(score_numerator),
        )
        rows = torch.arange(batch_size, device=device)
        positive_valid = valid_candidate_mask[rows, target_columns]
        valid_rows = positive_valid & valid_candidate_mask.sum(dim=1).ge(2)
        valid_count = valid_rows.sum()
        zero_loss = (score_numerator.sum() + score_denominator.sum()) * 0.0
        if valid_rows.any().item():
            masked_logits = (aggregated_scores / temperature).masked_fill(
                ~valid_candidate_mask,
                torch.finfo(aggregated_scores.dtype).min,
            )
            loss = F.cross_entropy(masked_logits[valid_rows], target_columns[valid_rows])
        else:
            loss = zero_loss
        diagnostics, valid_sample_count = self._materialize_diagnostics(
            diagnostic_tensors,
            valid_count,
        )
        return RawCompatibilityOutput(
            loss=loss,
            scores=scores,
            aggregated_scores=aggregated_scores,
            valid_candidate_mask=valid_candidate_mask,
            modality_weights=modality_weights,
            diagnostics=diagnostics,
            valid_samples=valid_sample_count,
        )

    def _validate_score_inputs(
        self,
        branch: str,
        query: Tensor,
        candidate_raw: Tensor,
    ) -> None:
        if query.dim() != 2 or query.shape[1] != self.hidden_dim:
            raise ValueError(
                f"query[{branch!r}] shape must be [B, {self.hidden_dim}], "
                f"got {tuple(query.shape)}"
            )
        expected_raw_dim = self.raw_dimensions[branch]
        if candidate_raw.dim() != 2 or candidate_raw.shape[1] != expected_raw_dim:
            raise ValueError(
                f"candidate_raw[{branch!r}] shape must be [C, {expected_raw_dim}], "
                f"got {tuple(candidate_raw.shape)}"
            )
        if not query.is_floating_point() or not candidate_raw.is_floating_point():
            raise TypeError(f"query and candidate_raw for {branch!r} must be floating tensors")
        if query.device != candidate_raw.device:
            raise ValueError(
                f"query and candidate_raw for {branch!r} must share a device, "
                f"got {query.device} and {candidate_raw.device}"
            )

    @staticmethod
    def _validate_branch_mappings(
        queries: Mapping[str, Tensor],
        candidate_raw: Mapping[str, Tensor],
        candidate_available: Mapping[str, Tensor],
        history_reliability: Mapping[str, Tensor],
    ) -> None:
        expected = set(MPSID_INPUT_MODALITIES)
        for name, values in (
            ("queries", queries),
            ("candidate_raw", candidate_raw),
            ("candidate_available", candidate_available),
            ("history_reliability", history_reliability),
        ):
            if not isinstance(values, Mapping) or set(values) != expected:
                keys = sorted(values) if isinstance(values, Mapping) else type(values).__name__
                raise ValueError(f"{name} must define exactly {MPSID_INPUT_MODALITIES!r}, got {keys!r}")

    def _validate_loss_inputs(
        self,
        queries: Mapping[str, Tensor],
        candidate_raw: Mapping[str, Tensor],
        candidate_available: Mapping[str, Tensor],
        history_reliability: Mapping[str, Tensor],
        target_columns: Tensor,
    ) -> tuple[int, int, torch.device]:
        reference = queries[MPSID_INPUT_MODALITIES[0]]
        if reference.dim() != 2:
            raise ValueError(f"query shape must be [B, H], got {tuple(reference.shape)}")
        batch_size = int(reference.shape[0])
        candidate_count = int(candidate_raw[MPSID_INPUT_MODALITIES[0]].shape[0])
        if batch_size == 0 or candidate_count == 0:
            raise ValueError("raw compatibility requires non-empty samples and candidates")
        device = reference.device

        if not isinstance(target_columns, Tensor) or target_columns.dim() != 1:
            shape = None if not isinstance(target_columns, Tensor) else tuple(target_columns.shape)
            raise ValueError(f"target_columns shape must be [B], got {shape}")
        if target_columns.dtype == torch.bool or target_columns.is_floating_point() or target_columns.is_complex():
            raise TypeError(f"target_columns must use an integer dtype, got {target_columns.dtype}")
        if target_columns.shape[0] != batch_size:
            raise ValueError(
                f"target_columns length {target_columns.shape[0]} != batch size {batch_size}"
            )
        if target_columns.device != device:
            raise ValueError(f"target_columns device {target_columns.device} != query device {device}")
        min_column = int(target_columns.min().item())
        max_column = int(target_columns.max().item())
        if min_column < 0 or max_column >= candidate_count:
            raise IndexError(
                f"target_columns must be in [0, {candidate_count - 1}], "
                f"got min={min_column}, max={max_column}"
            )

        for branch in MPSID_INPUT_MODALITIES:
            query = queries[branch]
            raw = candidate_raw[branch]
            availability = candidate_available[branch]
            reliability = history_reliability[branch]
            self._validate_score_inputs(branch, query, raw)
            if query.shape[0] != batch_size:
                raise ValueError(f"query[{branch!r}] batch size {query.shape[0]} != {batch_size}")
            if raw.shape[0] != candidate_count:
                raise ValueError(f"candidate_raw[{branch!r}] rows {raw.shape[0]} != {candidate_count}")
            if not isinstance(availability, Tensor) or availability.dtype != torch.bool:
                raise TypeError(f"candidate_available[{branch!r}] must be a bool tensor")
            if tuple(availability.shape) != (candidate_count,):
                raise ValueError(
                    f"candidate_available[{branch!r}] shape must be ({candidate_count},), "
                    f"got {tuple(availability.shape)}"
                )
            if availability.device != device or raw.device != device:
                raise ValueError(f"all raw compatibility tensors for {branch!r} must be on {device}")
            if not isinstance(reliability, Tensor) or not reliability.is_floating_point():
                raise TypeError(f"history_reliability[{branch!r}] must be a floating tensor")
            if tuple(reliability.shape) != (batch_size,):
                raise ValueError(
                    f"history_reliability[{branch!r}] shape must be ({batch_size},), "
                    f"got {tuple(reliability.shape)}"
                )
            if reliability.device != device:
                raise ValueError(f"history_reliability[{branch!r}] must be on {device}")
        return batch_size, candidate_count, device

    @staticmethod
    def _branch_diagnostic_tensors(
        scores: Tensor,
        availability: Tensor,
        target_columns: Tensor,
        valid_rows: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            batch_size, candidate_count = scores.shape
            rows = torch.arange(batch_size, device=scores.device)
            positive = scores[rows, target_columns]
            negative_mask = availability.view(1, candidate_count).expand(batch_size, -1).clone()
            negative_mask[rows, target_columns] = False
            negative = scores.masked_fill(
                ~negative_mask,
                torch.finfo(scores.dtype).min,
            ).max(dim=1).values
            masked_scores = scores.masked_fill(
                ~availability.view(1, candidate_count),
                torch.finfo(scores.dtype).min,
            )
            top1 = masked_scores.argmax(dim=1).eq(target_columns)
            margin = positive[valid_rows] - negative[valid_rows]
            valid_candidates = availability.sum()
            valid_samples = valid_rows.sum()
            zero = scores.new_zeros(())
            positive_margin = torch.where(valid_samples > 0, margin.mean(), zero)
            top1_rate = torch.where(
                valid_samples > 0,
                top1[valid_rows].float().mean(),
                zero,
            )
            return valid_candidates, valid_samples, positive_margin, top1_rate

    @staticmethod
    def _materialize_diagnostics(
        diagnostic_tensors: Mapping[str, tuple[Tensor, Tensor, Tensor, Tensor]],
        valid_sample_count: Tensor,
    ) -> tuple[dict[str, RawBranchDiagnostics], int]:
        branches = list(MPSID_INPUT_MODALITIES)
        packed = torch.stack(
            [
                value.detach().to(dtype=torch.float32)
                for branch in branches
                for value in diagnostic_tensors[branch]
            ]
            + [valid_sample_count.detach().to(dtype=torch.float32)]
        ).cpu().tolist()
        diagnostics: dict[str, RawBranchDiagnostics] = {}
        for index, branch in enumerate(branches):
            offset = index * 4
            diagnostics[branch] = RawBranchDiagnostics(
                valid_candidates=int(packed[offset]),
                valid_samples=int(packed[offset + 1]),
                positive_margin=float(packed[offset + 2]),
                top1=float(packed[offset + 3]),
            )
        return diagnostics, int(packed[-1])
