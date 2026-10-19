"""Beam-search evaluation for MoSID stages."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
import random
import time
from typing import Iterable, Mapping

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, **_kwargs):
        return iterable

from configs.config import DotDict
from data.raw_embedding_catalog import RawEmbeddingCatalog
from evaluate.metrics import compute_ranking_metrics
from inference.beam_search import BeamResult, TrieBeamSearcher
from inference.rank_fusion import aggregate_raw_compatibility_scores, rrf_rank
from inference.trie import SIDTrie
from utils.data import MPSID_INPUT_MODALITIES
from utils.logger import format_horizontal_metrics_table
from utils.precision import autocast_context


@dataclass(frozen=True)
class RRFSettings:
    k: float
    rank_fusion_weight: float = 1.0
    raw_compat_weight: float = 0.0


@dataclass
class BeamEvalOutput:
    metrics: dict[str, float]
    rrf_settings: RRFSettings | None = None
    rrf_tuned: bool = False
    rrf_search_exhaustive: bool = False
    fixed_rrf_metrics: list[dict[str, float]] | None = None


MOSID_CHECKPOINT_SCHEMA_VERSION = 3
RRF_SETTINGS_SCHEMA_VERSION = 3


_BRANCH_METRIC_SUFFIX = {
    "text": "t",
    "vision": "v",
    "collab": "id",
}


@dataclass(frozen=True)
class _RRFEvidenceCache:
    """Evaluation-local compact evidence; never persisted across model instances."""

    branch_order: tuple[str, ...]
    candidate_ids: torch.Tensor
    ranks: torch.Tensor
    raw_scores: torch.Tensor
    raw_available: torch.Tensor
    effective_branch_weights: torch.Tensor | None
    targets: torch.Tensor
    model_identity: int | None = None

    @classmethod
    def from_sparse(
        cls,
        sample_branch_ranks: list[dict[str, dict[int, int]]],
        targets: list[int],
        branches: Iterable[str],
        sample_raw_branch_scores: list[dict[str, dict[int, float]]] | None = None,
        sample_effective_branch_weights: list[dict[str, float]] | None = None,
        model_identity: int | None = None,
    ) -> "_RRFEvidenceCache":
        if len(sample_branch_ranks) != len(targets):
            raise ValueError("cached branch ranks must align with targets")
        if sample_raw_branch_scores is not None and len(sample_raw_branch_scores) != len(targets):
            raise ValueError("cached raw score evidence must align with targets")
        if (
            sample_effective_branch_weights is not None
            and len(sample_effective_branch_weights) != len(targets)
        ):
            raise ValueError("cached effective raw weights must align with targets")
        if (sample_raw_branch_scores is None) != (sample_effective_branch_weights is None):
            raise ValueError("cached raw scores and effective branch weights must be provided together")
        branch_order = tuple(sorted(str(branch) for branch in branches))
        if not branch_order or len(set(branch_order)) != len(branch_order):
            raise ValueError("RRF evidence cache requires unique branches")

        candidate_lists = [
            sorted({int(item_id) for item_ranks in branch_ranks.values() for item_id in item_ranks})
            for branch_ranks in sample_branch_ranks
        ]
        max_candidates = max((len(items) for items in candidate_lists), default=0)
        n_samples = len(targets)
        n_branches = len(branch_order)
        candidate_ids = torch.zeros((n_samples, max_candidates), dtype=torch.long)
        ranks = torch.zeros((n_samples, max_candidates, n_branches), dtype=torch.int32)
        raw_scores = torch.zeros((n_samples, max_candidates, n_branches), dtype=torch.float64)
        raw_available = torch.zeros((n_samples, max_candidates, n_branches), dtype=torch.bool)
        effective_branch_weights = (
            None
            if sample_effective_branch_weights is None
            else torch.zeros((n_samples, n_branches), dtype=torch.float64)
        )

        for sample_index, (items, branch_ranks) in enumerate(zip(candidate_lists, sample_branch_ranks)):
            if not items:
                continue
            if items[0] <= 0:
                raise ValueError("RRF candidate item ids must be positive")
            candidate_ids[sample_index, : len(items)] = torch.tensor(items, dtype=torch.long)
            columns = {item_id: column for column, item_id in enumerate(items)}
            for branch_index, branch in enumerate(branch_order):
                for item_id, rank in branch_ranks.get(branch, {}).items():
                    rank = int(rank)
                    if rank <= 0:
                        raise ValueError(f"RRF rank must be positive for item {item_id}, got {rank}")
                    ranks[sample_index, columns[int(item_id)], branch_index] = rank

            if sample_raw_branch_scores is None:
                continue
            raw_evidence = sample_raw_branch_scores[sample_index]
            effective_weights = sample_effective_branch_weights[sample_index]
            if set(raw_evidence) != set(branch_order):
                raise ValueError(
                    "cached raw score evidence must define exactly the cache branches; "
                    f"got {sorted(raw_evidence)!r}"
                )
            if set(effective_weights) != set(branch_order):
                raise ValueError(
                    "cached effective raw weights must define exactly the cache branches; "
                    f"got {sorted(effective_weights)!r}"
                )
            for branch_index, branch in enumerate(branch_order):
                weight = float(effective_weights[branch])
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(
                        f"effective raw weight for {branch!r} must be finite and non-negative"
                    )
                effective_branch_weights[sample_index, branch_index] = weight
                for item_id, score in raw_evidence.get(branch, {}).items():
                    item_id = int(item_id)
                    if item_id not in columns:
                        raise ValueError(
                            f"raw evidence item {item_id} is outside the sample beam candidate union"
                        )
                    score = float(score)
                    if not math.isfinite(score):
                        raise ValueError("raw branch scores must contain only finite values")
                    column = columns[item_id]
                    raw_scores[sample_index, column, branch_index] = score
                    raw_available[sample_index, column, branch_index] = True

        return cls(
            branch_order=branch_order,
            candidate_ids=candidate_ids,
            ranks=ranks,
            raw_scores=raw_scores,
            raw_available=raw_available,
            effective_branch_weights=effective_branch_weights,
            targets=torch.tensor(targets, dtype=torch.long),
            model_identity=model_identity,
        )

    def assert_model(self, model) -> None:
        if self.model_identity is not None and self.model_identity != id(model):
            raise ValueError("RRF evidence cache belongs to a different model instance")

    def components(
        self,
        setting: RRFSettings,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_rrf_settings(setting)
        shape = self.candidate_ids.shape
        rank_component = torch.zeros(shape, dtype=torch.float64)
        for branch_index, _branch in enumerate(self.branch_order):
            branch_present = self.ranks[:, :, branch_index] > 0
            branch_ranks = self.ranks[:, :, branch_index].to(torch.float64)
            rank_component += torch.where(
                branch_present,
                1.0 / (float(setting.k) + branch_ranks),
                0.0,
            )

        if self.effective_branch_weights is None:
            if setting.raw_compat_weight > 0.0:
                raise ValueError(
                    "positive raw compatibility weight requires cached effective branch weights"
                )
            raw_denominator = torch.zeros(shape, dtype=torch.float64)
            raw_numerator = torch.zeros(shape, dtype=torch.float64)
            empty_history = torch.zeros(shape[0], dtype=torch.bool)
        else:
            weights = self.effective_branch_weights.unsqueeze(1)
            weighted_available = self.raw_available.to(torch.float64) * weights
            raw_numerator = (self.raw_scores * weighted_available).sum(dim=2)
            raw_denominator = weighted_available.sum(dim=2)
            empty_history = self.effective_branch_weights.sum(dim=1).eq(0.0)

        raw_mean = torch.where(
            raw_denominator > 0.0,
            raw_numerator / raw_denominator.clamp_min(torch.finfo(torch.float64).tiny),
            0.0,
        )
        candidate_mask = self.candidate_ids > 0
        raw_evidenced = raw_denominator > 0.0
        if setting.raw_compat_weight > 0.0:
            missing = candidate_mask & ~raw_evidenced & ~empty_history.unsqueeze(1)
            if missing.any().item():
                missing_ids = self.candidate_ids[missing].tolist()
                raise ValueError(
                    "non-empty history candidate has no effective raw compatibility evidence; "
                    f"missing item ids: {missing_ids[:10]}"
                )

        raw_rank_component = torch.zeros(shape, dtype=torch.float64)
        if raw_evidenced.any().item():
            masked_raw = raw_mean.masked_fill(~raw_evidenced, float("-inf"))
            sorted_scores, order = torch.sort(masked_raw, dim=1, descending=True, stable=True)
            positions = torch.arange(
                1,
                shape[1] + 1,
                dtype=torch.int64,
            ).view(1, -1).expand(shape[0], -1)
            new_score = torch.ones(shape, dtype=torch.bool)
            if shape[1] > 1:
                new_score[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
            sorted_ranks = torch.where(new_score, positions, 0).cummax(dim=1).values
            raw_ranks = torch.zeros(shape, dtype=torch.int64)
            raw_ranks.scatter_(1, order, sorted_ranks)
            raw_rank_component = torch.where(
                raw_evidenced,
                1.0 / (float(setting.k) + raw_ranks.to(torch.float64)),
                0.0,
            )
        return rank_component, raw_rank_component, raw_evidenced

    def component_key(self, setting: RRFSettings) -> tuple:
        return (float(setting.k), bool(setting.raw_compat_weight > 0.0))

    def target_ranks(
        self,
        setting: RRFSettings,
        components: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if components is None:
            components = self.components(setting)
        rank_component, raw_rank_component, _ = components
        rank_weight = torch.full(
            (self.candidate_ids.shape[0], 1),
            float(setting.rank_fusion_weight),
            dtype=torch.float64,
        )
        if setting.raw_compat_weight > 0.0 and self.effective_branch_weights is not None:
            empty_history = self.effective_branch_weights.sum(dim=1).eq(0.0)
            if setting.rank_fusion_weight == 0.0:
                rank_weight[empty_history] = 1.0
        scores = (
            rank_weight * rank_component
            + float(setting.raw_compat_weight) * raw_rank_component
        )
        valid = self.candidate_ids > 0
        target_mask = valid & self.candidate_ids.eq(self.targets.view(-1, 1))
        present = target_mask.any(dim=1)
        target_scores = torch.where(target_mask, scores, 0.0).sum(dim=1)
        outranks_target = scores > target_scores.view(-1, 1)
        tied_before_target = scores.eq(target_scores.view(-1, 1)) & self.candidate_ids.lt(
            self.targets.view(-1, 1)
        )
        ranks = 1 + (valid & (outranks_target | tied_before_target)).sum(dim=1)
        return torch.where(present, ranks, 0).long()

    def target_metrics(
        self,
        setting: RRFSettings,
        top_k: list[int],
        components: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        target_ranks = self.target_ranks(setting, components=components)
        denom = int(target_ranks.numel())
        if denom == 0:
            return _metrics_from_target_ranks([], top_k)
        present = target_ranks > 0
        metrics: dict[str, float] = {}
        for k in top_k:
            hits = present & target_ranks.le(int(k))
            metrics[f"Recall@{k}"] = float(hits.sum().item()) / denom
            discounts = torch.where(
                hits,
                1.0 / torch.log2(target_ranks.clamp_min(1).to(torch.float64) + 1.0),
                0.0,
            )
            metrics[f"NDCG@{k}"] = float(discounts.sum().item()) / denom
        metrics["BeamCoverage"] = float(present.sum().item()) / denom
        return metrics

def get_top_k(config: DotDict) -> list[int]:
    if "evaluation" in config and "top_k" in config.evaluation:
        return [int(k) for k in config.evaluation.top_k]
    if "eval" in config and "top_k" in config.eval:
        return [int(k) for k in config.eval.top_k]
    return [5, 10, 20]


def _get_split_beam_size(config: DotDict, field: str) -> int:
    inference = getattr(config, "inference", None)
    value = getattr(inference, field, None) if inference is not None else None
    if value is None and inference is not None:
        value = getattr(inference, "beam_size", None)
    if value is None:
        raise ValueError(f"missing required config field 'inference.{field}'")
    beam_size = int(value)
    if beam_size <= 0:
        raise ValueError(f"inference.{field} must be positive, got {beam_size}")
    return beam_size


def get_valid_beam_size(config: DotDict) -> int:
    return _get_split_beam_size(config, "valid_beam_size")


def get_test_beam_size(config: DotDict) -> int:
    return _get_split_beam_size(config, "test_beam_size")


def format_metrics_table(title: str, metrics: dict[str, float]) -> str:
    pairs = [(key, f"{value:.6f}") for key, value in metrics.items()]
    return format_horizontal_metrics_table(title, pairs, pairs_per_row=4)


def format_beam_metrics_summary(
    title: str,
    metrics: Mapping[str, float],
    selection_metric: str,
    selection_score: float,
    top_k: Iterable[int],
    active_channels: Iterable[str],
    rrf_enabled: bool,
) -> str:
    """Format the compact Beam metrics shared by training and test evaluation."""

    def metric_text(key: str, fmt: str) -> str:
        return fmt.format(float(metrics[key])) if key in metrics else "-"

    channels = [str(channel) for channel in active_channels]
    pairs = [("select", str(selection_metric)), ("score", f"{float(selection_score):.6f}")]
    for k in list(top_k)[:3]:
        pairs.append((f"beam_recall@{k}", f"{float(metrics.get(f'beam_recall@{k}', 0.0)):.6f}"))
        pairs.append((f"beam_ndcg@{k}", f"{float(metrics.get(f'beam_ndcg@{k}', 0.0)):.6f}"))
        if rrf_enabled:
            for branch in channels:
                suffix = _BRANCH_METRIC_SUFFIX.get(branch, branch)
                pairs.append((f"branch_hr_{suffix}@{k}", metric_text(f"branch_hr_{suffix}@{k}", "{:.6f}")))
                pairs.append(
                    (f"branch_ndcg_{suffix}@{k}", metric_text(f"branch_ndcg_{suffix}@{k}", "{:.6f}"))
                )
            pairs.append((f"oracle_union@{k}", metric_text(f"oracle_union@{k}", "{:.6f}")))
            pairs.append((f"jaccard_all@{k}", metric_text(f"jaccard_all@{k}", "{:.6f}")))
    pairs.extend(
        [
            ("beam_coverage", f"{float(metrics.get('beam_coverage', 0.0)):.6f}"),
            ("n_eval", f"{float(metrics.get('n_eval', 0.0)):.0f}"),
        ]
    )
    if rrf_enabled:
        pairs.extend(
            [
                ("rrf_tuned", f"{float(metrics.get('rrf_tuned', 0.0)):.0f}"),
                ("rrf_grid", f"{float(metrics.get('rrf_grid_size', 0.0)):.0f}"),
                ("rrf_exhaustive", f"{float(metrics.get('rrf_search_exhaustive', 0.0)):.0f}"),
                ("rrf_k", metric_text("rrf_k", "{:.1f}")),
                ("rrf_rank_w", metric_text("rrf_rank_fusion_weight", "{:.3f}")),
                ("rrf_raw_w", metric_text("rrf_raw_compat_weight", "{:.3f}")),
                ("raw_rescue@10", metric_text("raw_rescue@10", "{:.6f}")),
                ("raw_harm@10", metric_text("raw_harm@10", "{:.6f}")),
                ("rank_mean", metric_text("rrf_rank_component_mean", "{:.6f}")),
                ("raw_rank_mean", metric_text("rrf_raw_rank_component_mean", "{:.6f}")),
            ]
        )
    return format_horizontal_metrics_table(title, pairs, pairs_per_row=4)


def get_merge_branches(config: DotDict) -> list[str] | None:
    if "inference" not in config or "merge_branches" not in config.inference:
        return None
    merge_branches = config.inference.merge_branches
    if merge_branches is None:
        return None
    return [str(branch) for branch in merge_branches]


def get_max_batches(config: DotDict) -> int:
    if "evaluation" in config and "max_batches" in config.evaluation:
        return int(config.evaluation.max_batches)
    if "eval" in config and "max_batches" in config.eval:
        return int(config.eval.max_batches)
    return 0


def get_beam_logits_chunk_size(config: DotDict) -> int:
    raw_value = None
    if "inference" in config:
        raw_value = getattr(config.inference, "beam_logits_chunk_size", None)
    if raw_value is None:
        return 2048
    return int(raw_value)


def get_single_branch_name(config: DotDict) -> str:
    if "model" in config and "active_channels" in config.model and len(config.model.active_channels) == 1:
        return str(config.model.active_channels[0])
    return "collab"


def is_single_channel_config(config: DotDict) -> bool:
    return (
        "model" in config
        and "active_channels" in config.model
        and len(config.model.active_channels) == 1
    )


def get_rrf_settings(
    config: DotDict,
    override: RRFSettings | None = None,
) -> RRFSettings:
    """Read reciprocal-rank-fusion settings from the configuration."""
    if override is not None:
        return override
    rrf = getattr(config, "rrf", None)
    k_rrf = float(getattr(rrf, "k", 60.0) if rrf is not None else 60.0)
    rank_fusion_weight = float(
        getattr(rrf, "rank_fusion_weight", 1.0) if rrf is not None else 1.0
    )
    raw_compat_weight = float(
        getattr(rrf, "raw_compat_weight", 1.0) if rrf is not None else 1.0
    )
    settings = RRFSettings(
        k=k_rrf,
        rank_fusion_weight=rank_fusion_weight,
        raw_compat_weight=raw_compat_weight,
    )
    _validate_rrf_settings(settings)
    return settings


def serialize_rrf_settings(settings: RRFSettings) -> dict[str, object]:
    """Serialize the complete validation-selected fusion setting."""
    _validate_rrf_settings(settings)
    return {
        "schema_version": RRF_SETTINGS_SCHEMA_VERSION,
        "k": float(settings.k),
        "rank_fusion_weight": float(settings.rank_fusion_weight),
        "raw_compat_weight": float(settings.raw_compat_weight),
    }


def deserialize_rrf_settings(
    payload: Mapping[str, object],
) -> RRFSettings:
    """Strictly restore a versioned RRF setting written by Stage 2."""
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint selected_rrf must be a mapping")
    required = {
        "schema_version",
        "k",
        "rank_fusion_weight",
        "raw_compat_weight",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"versioned checkpoint selected_rrf is missing fields: {missing}")
    unexpected = sorted(set(payload).difference(required))
    if unexpected:
        raise ValueError(f"versioned checkpoint selected_rrf has unexpected fields: {unexpected}")
    version = int(payload["schema_version"])
    if version != RRF_SETTINGS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported selected_rrf schema_version={version}; "
            f"expected {RRF_SETTINGS_SCHEMA_VERSION}"
        )
    settings = RRFSettings(
        k=float(payload["k"]),
        rank_fusion_weight=float(payload["rank_fusion_weight"]),
        raw_compat_weight=float(payload["raw_compat_weight"]),
    )
    _validate_rrf_settings(settings)
    return settings


def rrf_settings_from_checkpoint_metadata(
    metadata: Mapping[str, object],
) -> RRFSettings | None:
    """Restore strict rank-space RRF metadata and reject legacy schemas."""
    validate_mosid_checkpoint_schema(metadata)
    extra_state = metadata.get("extra_state", {})
    if "selected_rrf" not in extra_state:
        raise ValueError("versioned multi-branch checkpoint is missing selected_rrf")
    selection = extra_state.get("rrf_selection")
    required_selection_fields = {
        "source",
        "selection_metric",
        "validation_max_batches",
        "validation_complete",
        "search_exhaustive",
        "grid_size",
    }
    if not isinstance(selection, Mapping):
        raise ValueError("versioned multi-branch checkpoint is missing rrf_selection")
    missing_selection = sorted(required_selection_fields.difference(selection))
    if missing_selection:
        raise ValueError(
            f"versioned checkpoint rrf_selection is missing fields: {missing_selection}"
        )
    return deserialize_rrf_settings(extra_state["selected_rrf"])


def validate_mosid_checkpoint_schema(metadata: Mapping[str, object]) -> None:
    """Fail before model restoration when a checkpoint predates the current schema."""
    extra_state = metadata.get("extra_state", {})
    if not isinstance(extra_state, Mapping) or "mosid_checkpoint_schema_version" not in extra_state:
        raise ValueError(
            "legacy MoSID checkpoint schema is unsupported; "
            f"expected schema_version={MOSID_CHECKPOINT_SCHEMA_VERSION}"
        )
    version = int(extra_state["mosid_checkpoint_schema_version"])
    if version != MOSID_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported MoSID checkpoint schema_version={version}; "
            f"expected {MOSID_CHECKPOINT_SCHEMA_VERSION}"
        )


def validate_rrf_checkpoint_for_test(metadata: Mapping[str, object]) -> None:
    """Reject label-selected settings produced from an incomplete validation search."""
    extra_state = metadata.get("extra_state", {})
    if not isinstance(extra_state, Mapping) or "mosid_checkpoint_schema_version" not in extra_state:
        return
    selection = extra_state.get("rrf_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("versioned multi-branch checkpoint is missing rrf_selection")
    if str(selection.get("source")) != "validation_grid":
        return
    if not bool(selection.get("validation_complete")):
        raise ValueError(
            "test evaluation refuses RRF settings selected on a truncated validation split"
        )
    if not bool(selection.get("search_exhaustive")):
        raise ValueError(
            "test evaluation refuses RRF settings selected from a sampled parameter grid"
        )


def rrf_settings_metrics(settings: RRFSettings) -> dict[str, float]:
    return {
        "rrf_k": float(settings.k),
        "rrf_rank_fusion_weight": float(settings.rank_fusion_weight),
        "rrf_raw_compat_weight": float(settings.raw_compat_weight),
    }


def _validate_rrf_settings(settings: RRFSettings) -> None:
    numeric = {
        "k": settings.k,
        "rank_fusion_weight": settings.rank_fusion_weight,
        "raw_compat_weight": settings.raw_compat_weight,
    }
    if not all(math.isfinite(float(value)) for value in numeric.values()):
        raise ValueError("checkpoint selected_rrf values must be finite")
    if settings.k <= 0.0:
        raise ValueError(f"checkpoint selected_rrf.k must be positive, got {settings.k}")
    for name, value in numeric.items():
        if name != "k" and float(value) < 0.0:
            raise ValueError(f"checkpoint selected_rrf.{name} must be non-negative, got {value}")
    if settings.rank_fusion_weight == 0.0 and settings.raw_compat_weight == 0.0:
        raise ValueError(
            "checkpoint selected_rrf rank_fusion_weight and raw_compat_weight must not both be zero"
        )


def get_rrf_search_grid(
    config: DotDict,
    base_settings: RRFSettings | None = None,
    seed: int | None = None,
    exhaustive: bool = False,
) -> list[RRFSettings]:
    """Read candidate values for reciprocal-rank-fusion parameter selection."""
    base = base_settings if base_settings is not None else get_rrf_settings(config)
    rrf = getattr(config, "rrf", None)

    param_choices = getattr(rrf, "param_choices", None) if rrf is not None else None
    if hasattr(param_choices, "to_dict"):
        param_choices = param_choices.to_dict()
    sample_count = None if exhaustive else int(getattr(rrf, "sample_count", 0) or 0) if rrf is not None else 0
    if param_choices:
        return _search_grid_from_param_choices(param_choices, base, sample_count, seed)
    _validate_rrf_settings(base)
    return [base]


def _search_grid_from_param_choices(
    param_choices: dict,
    base: RRFSettings,
    sample_count: int | None,
    seed: int | None,
) -> list[RRFSettings]:
    if sample_count is not None and sample_count < 0:
        raise ValueError(f"rrf.sample_count must be non-negative, got {sample_count}")
    _validate_rrf_settings(base)

    rank_fusion_weight_values = _float_values(
        param_choices.get("rank_fusion_weight"),
        base.rank_fusion_weight,
    )
    k_values = _float_values(param_choices.get("k"), base.k)
    raw_compat_weight_values = _float_values(
        param_choices.get("raw_compat_weight"),
        base.raw_compat_weight,
    )

    candidates = [base]
    for rank_weight, k_rrf, raw_weight in product(
        rank_fusion_weight_values,
        k_values,
        raw_compat_weight_values,
    ):
        if float(rank_weight) == 0.0 and float(raw_weight) == 0.0:
            continue
        candidates.append(_rrf_settings_from_values(rank_weight, k_rrf, raw_weight))

    canonical_by_ratio: dict[tuple, RRFSettings] = {}
    ratio_order: list[tuple] = []
    for setting in candidates:
        _validate_rrf_settings(setting)
        key = _rrf_ratio_key(setting)
        current = canonical_by_ratio.get(key)
        if current is None:
            ratio_order.append(key)
            canonical_by_ratio[key] = setting
        elif _rrf_weight_scale(setting) > _rrf_weight_scale(current):
            canonical_by_ratio[key] = setting

    base_key = _rrf_ratio_key(base)
    canonical_base = canonical_by_ratio[base_key]
    alternatives = [
        canonical_by_ratio[key]
        for key in ratio_order
        if key != base_key
    ]
    if sample_count is not None:
        random.Random(seed).shuffle(alternatives)
        alternatives = alternatives[:sample_count]
    return [canonical_base, *alternatives]


def _float_values(raw_values, fallback: float) -> list[float]:
    if raw_values is None:
        return [float(fallback)]
    values = [float(value) for value in raw_values]
    if not values:
        return [float(fallback)]
    return values


def _rrf_settings_from_values(
    rank_fusion_weight: float,
    k_rrf: float,
    raw_compat_weight: float,
) -> RRFSettings:
    return RRFSettings(
        k=float(k_rrf),
        rank_fusion_weight=float(rank_fusion_weight),
        raw_compat_weight=float(raw_compat_weight),
    )


def _rrf_ratio_key(setting: RRFSettings) -> tuple:
    scale = _rrf_weight_scale(setting)
    return (
        round(setting.k, 12),
        round(setting.rank_fusion_weight / scale, 12),
        round(setting.raw_compat_weight / scale, 12),
    )


def _rrf_weight_scale(setting: RRFSettings) -> float:
    return max(float(setting.rank_fusion_weight), float(setting.raw_compat_weight))


def _add_rrf_metrics(
    metrics: dict[str, float],
    setting: RRFSettings,
    tuned: bool,
    grid_size: int,
    exhaustive: bool = False,
) -> None:
    metrics.update(rrf_settings_metrics(setting))
    metrics["rrf_tuned"] = 1.0 if tuned else 0.0
    metrics["rrf_grid_size"] = float(grid_size)
    metrics["rrf_search_exhaustive"] = 1.0 if tuned and exhaustive else 0.0


def resolve_beam_metric_score(metrics: dict[str, float], metric: str, label: str = "metric") -> float:
    candidates = [metric]
    lower = metric.lower()
    if lower.startswith("beam_"):
        suffix = metric[5:]
        suffix_lower = suffix.lower()
        if suffix_lower.startswith("ndcg@"):
            candidates.append("NDCG@" + suffix.split("@", 1)[1])
        elif suffix_lower.startswith("recall@"):
            candidates.append("Recall@" + suffix.split("@", 1)[1])
        elif suffix_lower == "coverage":
            candidates.append("BeamCoverage")
    for key in candidates:
        if key in metrics:
            return float(metrics[key])
    available = ", ".join(metrics.keys())
    raise KeyError(f"{label} {metric!r} not found in beam metrics; available: {available}")


def _rank_cached_branch_ranks(
    sample_branch_ranks: list[dict[str, dict[int, int]]],
    setting: RRFSettings,
    max_items: int,
    sample_raw_branch_scores: list[dict[str, dict[int, float]]] | None = None,
    sample_effective_branch_weights: list[dict[str, float]] | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    if sample_raw_branch_scores is not None and len(sample_raw_branch_scores) != len(sample_branch_ranks):
        raise ValueError("cached raw score evidence must align with cached branch ranks")
    if (
        sample_effective_branch_weights is not None
        and len(sample_effective_branch_weights) != len(sample_branch_ranks)
    ):
        raise ValueError("cached effective raw weights must align with cached branch ranks")
    if (sample_raw_branch_scores is None) != (sample_effective_branch_weights is None):
        raise ValueError("cached raw scores and effective branch weights must be provided together")
    predictions: list[list[int]] = []
    candidates: list[list[int]] = []
    for sample_index, branch_ranks in enumerate(sample_branch_ranks):
        candidate_ids = sorted(
            {
                int(item_id)
                for item_ranks in branch_ranks.values()
                for item_id in item_ranks
            }
        )
        raw_scores = _raw_scores_for_setting(
            None if sample_raw_branch_scores is None else sample_raw_branch_scores[sample_index],
            (
                None
                if sample_effective_branch_weights is None
                else sample_effective_branch_weights[sample_index]
            ),
            setting,
            candidate_ids,
        )
        sample_setting = _setting_with_empty_history_fallback(setting, raw_scores)
        result = rrf_rank(
            branch_ranks=branch_ranks,
            max_items=max_items,
            k_rrf=sample_setting.k,
            rank_fusion_weight=sample_setting.rank_fusion_weight,
            raw_compat_weight=sample_setting.raw_compat_weight,
            raw_scores=raw_scores,
        )
        predictions.append(result.predictions)
        candidates.append(result.candidates)
    return predictions, candidates


def _target_rank_metrics_from_cached_ranks(
    sample_branch_ranks: list[dict[str, dict[int, int]]],
    targets: list[int],
    top_k: list[int],
    setting: RRFSettings,
    sample_raw_branch_scores: list[dict[str, dict[int, float]]] | None = None,
    sample_effective_branch_weights: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    if sample_raw_branch_scores is not None and len(sample_raw_branch_scores) != len(sample_branch_ranks):
        raise ValueError("cached raw score evidence must align with cached branch ranks")
    if (
        sample_effective_branch_weights is not None
        and len(sample_effective_branch_weights) != len(sample_branch_ranks)
    ):
        raise ValueError("cached effective raw weights must align with cached branch ranks")
    if (sample_raw_branch_scores is None) != (sample_effective_branch_weights is None):
        raise ValueError("cached raw scores and effective branch weights must be provided together")
    target_ranks: list[int | None] = []
    for sample_index, (branch_ranks, target) in enumerate(zip(sample_branch_ranks, targets)):
        candidate_ids = sorted(
            {
                int(item_id)
                for item_ranks in branch_ranks.values()
                for item_id in item_ranks
            }
        )
        raw_scores = _raw_scores_for_setting(
            None if sample_raw_branch_scores is None else sample_raw_branch_scores[sample_index],
            (
                None
                if sample_effective_branch_weights is None
                else sample_effective_branch_weights[sample_index]
            ),
            setting,
            candidate_ids,
        )
        sample_setting = _setting_with_empty_history_fallback(setting, raw_scores)
        target_ranks.append(
            _target_rank_from_branch_ranks(
                branch_ranks,
                int(target),
                sample_setting,
                raw_scores=raw_scores,
            )
        )
    while len(target_ranks) < len(targets):
        target_ranks.append(None)
    return _metrics_from_target_ranks(target_ranks, top_k)


def _target_rank_from_branch_ranks(
    branch_ranks: dict[str, dict[int, int]],
    target: int,
    setting: RRFSettings,
    raw_scores: dict[int, float] | None = None,
) -> int | None:
    target = int(target)
    candidate_ids = {
        int(item_id)
        for item_ranks in branch_ranks.values()
        for item_id in item_ranks
    }
    if target not in candidate_ids:
        return None
    result = rrf_rank(
        branch_ranks=branch_ranks,
        max_items=len(candidate_ids),
        k_rrf=setting.k,
        rank_fusion_weight=setting.rank_fusion_weight,
        raw_compat_weight=setting.raw_compat_weight,
        raw_scores=raw_scores,
    )
    return result.candidates.index(target) + 1


def _metrics_from_target_ranks(target_ranks: list[int | None], top_k: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    denom = len(target_ranks)
    if denom <= 0:
        for k in top_k:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"NDCG@{k}"] = 0.0
        metrics["BeamCoverage"] = 0.0
        return metrics

    for k in top_k:
        hits = [rank for rank in target_ranks if rank is not None and rank <= k]
        metrics[f"Recall@{k}"] = len(hits) / denom
        metrics[f"NDCG@{k}"] = sum(1.0 / math.log2(rank + 1) for rank in hits) / denom
    metrics["BeamCoverage"] = sum(rank is not None for rank in target_ranks) / denom
    return metrics


def _tune_rrf_from_cached_ranks(
    sample_branch_ranks: list[dict[str, dict[int, int]]],
    targets: list[int],
    top_k: list[int],
    max_items: int,
    settings_grid: list[RRFSettings],
    metric: str,
    sample_raw_branch_scores: list[dict[str, dict[int, float]]] | None = None,
    sample_effective_branch_weights: list[dict[str, float]] | None = None,
    evidence_cache: _RRFEvidenceCache | None = None,
) -> tuple[RRFSettings, dict[str, float], list[list[int]], list[list[int]]]:
    if not settings_grid:
        raise ValueError("RRF settings grid must not be empty")
    if evidence_cache is None:
        branches = sorted(
            {
                branch
                for branch_ranks in sample_branch_ranks
                for branch in branch_ranks
            }
        )
        evidence_cache = _RRFEvidenceCache.from_sparse(
            sample_branch_ranks,
            targets,
            branches,
            sample_raw_branch_scores=sample_raw_branch_scores,
            sample_effective_branch_weights=sample_effective_branch_weights,
        )
    component_cache: dict[
        tuple,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}
    max_cached_components = 8

    def metrics_for(setting: RRFSettings) -> dict[str, float]:
        key = evidence_cache.component_key(setting)
        components = component_cache.get(key)
        if components is None:
            if len(component_cache) >= max_cached_components:
                component_cache.pop(next(iter(component_cache)))
            components = evidence_cache.components(setting)
            component_cache[key] = components
        return evidence_cache.target_metrics(setting, top_k, components=components)

    best_setting = settings_grid[0]
    best_metrics = metrics_for(best_setting)
    best_score = resolve_beam_metric_score(best_metrics, metric, label="RRF tune metric")

    for setting in settings_grid[1:]:
        metrics = metrics_for(setting)
        score = resolve_beam_metric_score(metrics, metric, label="RRF tune metric")
        if score > best_score:
            best_score = score
            best_setting = setting
            best_metrics = metrics
    best_predictions, best_candidates = _rank_cached_branch_ranks(
        sample_branch_ranks,
        best_setting,
        max_items,
        sample_raw_branch_scores=sample_raw_branch_scores,
        sample_effective_branch_weights=sample_effective_branch_weights,
    )
    best_metrics = compute_ranking_metrics(best_predictions, best_candidates, targets, top_k)
    return best_setting, best_metrics, best_predictions, best_candidates


def _raw_scores_for_setting(
    raw_branch_scores: dict[str, dict[int, float]] | None,
    effective_branch_weights: dict[str, float] | None,
    setting: RRFSettings,
    candidate_ids: Iterable[int],
) -> dict[int, float] | None:
    if setting.raw_compat_weight == 0.0:
        return None
    if raw_branch_scores is None or effective_branch_weights is None:
        raise ValueError("raw_compat_weight requires cached raw scores and effective branch weights")
    aggregated = aggregate_raw_compatibility_scores(
        raw_branch_scores,
        effective_branch_weights,
        candidate_ids,
    )
    if all(float(weight) == 0.0 for weight in effective_branch_weights.values()):
        return None
    return aggregated


def _setting_with_empty_history_fallback(
    setting: RRFSettings,
    raw_scores: dict[int, float] | None,
) -> RRFSettings:
    if setting.raw_compat_weight == 0.0 or raw_scores is not None:
        return setting
    rank_weight = float(setting.rank_fusion_weight)
    return replace(
        setting,
        rank_fusion_weight=rank_weight if rank_weight > 0.0 else 1.0,
        raw_compat_weight=0.0,
    )


def _truncate_separate_candidates(
    results: list[BeamResult], config: DotDict, beam_size: int,
) -> list[BeamResult]:
    """Cap the merged pre-raw ranking, keeping all cached evidence consistent."""
    if getattr(getattr(config, "model", None), "sid_artifact_mode", None) != "separate":
        return results
    limit = beam_size * 10
    truncated = []
    for result in results:
        if len(result.candidates) <= limit:
            truncated.append(result)
            continue
        candidates = result.candidates[:limit]
        retained = set(candidates)
        truncated.append(replace(
            result,
            candidates=candidates,
            predictions=[item for item in result.predictions if item in retained],
            scores=result.scores[:limit],
            candidate_ranks={item: rank for item, rank in result.candidate_ranks.items() if item in retained},
            branch_candidates={branch: [item for item in items if item in retained]
                               for branch, items in result.branch_candidates.items()},
            branch_ranks={branch: {item: rank for item, rank in ranks.items() if item in retained}
                          for branch, ranks in result.branch_ranks.items()},
            branch_hit_counts={item: count for item, count in result.branch_hit_counts.items() if item in retained},
        ))
    return truncated


def _score_raw_candidates(
    model,
    raw_catalog: RawEmbeddingCatalog,
    results: list[BeamResult],
    history_queries: dict[str, torch.Tensor],
    history_reliability: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, dict[int, float]]], list[dict[str, float]]]:
    """Cache sparse raw scores and effective weights for one evaluation batch."""
    scorer = getattr(model, "raw_compatibility_scorer", None)
    if scorer is None:
        raise ValueError("raw compatibility inference requires model.raw_compatibility_scorer")
    if not history_queries:
        raise ValueError("raw compatibility inference requires non-empty history queries")
    if not isinstance(history_reliability, Mapping):
        raise ValueError("raw compatibility inference requires history_reliability")
    unknown_branches = sorted(set(history_queries).difference(MPSID_INPUT_MODALITIES))
    if unknown_branches:
        raise ValueError(f"unknown raw history query branches: {unknown_branches!r}")
    missing_reliability = sorted(set(history_queries).difference(history_reliability))
    if missing_reliability:
        raise ValueError(
            "history_reliability is missing raw query branches: "
            f"{missing_reliability!r}"
        )

    modality_weights = scorer.modality_weights().detach().cpu()
    if tuple(modality_weights.shape) != (len(MPSID_INPUT_MODALITIES),):
        raise ValueError(
            "raw modality weights must follow canonical branch order with shape "
            f"({len(MPSID_INPUT_MODALITIES)},), got {tuple(modality_weights.shape)}"
        )
    effective_by_branch: dict[str, torch.Tensor] = {}
    for branch, query in history_queries.items():
        if query.dim() != 2 or query.shape[0] != len(results):
            raise ValueError(
                f"raw history query for {branch!r} must have shape [B, H] with B={len(results)}, "
                f"got {tuple(query.shape)}"
            )
        reliability = history_reliability[branch]
        if not isinstance(reliability, torch.Tensor) or not reliability.is_floating_point():
            raise TypeError(f"history_reliability[{branch!r}] must be a floating tensor")
        if tuple(reliability.shape) != (len(results),):
            raise ValueError(
                f"history_reliability[{branch!r}] must have shape ({len(results)},), "
                f"got {tuple(reliability.shape)}"
            )
        reliability_cpu = reliability.detach().float().cpu()
        if not torch.isfinite(reliability_cpu).all().item():
            raise ValueError(f"history_reliability[{branch!r}] must contain only finite values")
        if reliability_cpu.lt(0.0).any().item() or reliability_cpu.gt(1.0).any().item():
            raise ValueError(f"history_reliability[{branch!r}] must be in [0, 1]")
        branch_index = MPSID_INPUT_MODALITIES.index(branch)
        effective_by_branch[branch] = reliability_cpu * modality_weights[branch_index]

    batch_effective_weights = [
        {
            branch: float(effective_by_branch[branch][sample_index].item())
            for branch in history_queries
        }
        for sample_index in range(len(results))
    ]

    candidate_lists = [
        list(dict.fromkeys(int(item_id) for item_id in result.candidates))
        for result in results
    ]
    max_candidates = max((len(candidate_ids) for candidate_ids in candidate_lists), default=0)
    batch_evidence = [
        {branch: {} for branch in history_queries}
        for _result in results
    ]
    if max_candidates == 0:
        return batch_evidence, batch_effective_weights

    item_ids = torch.zeros((len(results), max_candidates), dtype=torch.long)
    candidate_mask = torch.zeros_like(item_ids, dtype=torch.bool)
    for sample_index, candidate_ids in enumerate(candidate_lists):
        if candidate_ids:
            item_ids[sample_index, : len(candidate_ids)] = torch.tensor(candidate_ids, dtype=torch.long)
            candidate_mask[sample_index, : len(candidate_ids)] = True

    flat_item_ids = item_ids.reshape(-1)
    for branch, query in history_queries.items():
        candidate_raw, availability = raw_catalog.gather(branch, flat_item_ids)
        candidate_raw = candidate_raw.reshape(
            len(results),
            max_candidates,
            candidate_raw.shape[-1],
        ).to(
            query.device,
            non_blocking=query.device.type == "cuda",
        )
        available = availability.reshape(len(results), max_candidates) & candidate_mask
        scores = scorer.score_aligned_branch(branch, query, candidate_raw).detach().cpu().tolist()
        available_rows = available.tolist()
        for sample_index, candidate_ids in enumerate(candidate_lists):
            batch_evidence[sample_index][branch] = {
                item_id: float(scores[sample_index][column])
                for column, item_id in enumerate(candidate_ids)
                if available_rows[sample_index][column]
            }
    return batch_evidence, batch_effective_weights


def _add_fusion_diagnostics(
    metrics: dict[str, float],
    evidence_cache: _RRFEvidenceCache,
    setting: RRFSettings,
) -> None:
    components = evidence_cache.components(setting)
    selected_ranks = evidence_cache.target_ranks(setting, components=components)
    rank_only_setting = replace(
        setting,
        rank_fusion_weight=1.0,
        raw_compat_weight=0.0,
    )
    rank_only_ranks = evidence_cache.target_ranks(rank_only_setting, components=components)
    cutoff = 10
    selected_hits = selected_ranks.gt(0) & selected_ranks.le(cutoff)
    rank_only_hits = rank_only_ranks.gt(0) & rank_only_ranks.le(cutoff)
    rescue_count = int((selected_hits & ~rank_only_hits).sum().item())
    harm_count = int((rank_only_hits & ~selected_hits).sum().item())
    denom = max(int(selected_ranks.numel()), 1)
    metrics[f"raw_rescue@{cutoff}"] = rescue_count / denom
    metrics[f"raw_harm@{cutoff}"] = harm_count / denom
    metrics[f"raw_rescue_count@{cutoff}"] = float(rescue_count)
    metrics[f"raw_harm_count@{cutoff}"] = float(harm_count)

    rank_component, raw_rank_component, raw_available = components
    candidate_mask = evidence_cache.candidate_ids > 0
    _add_distribution_metrics(metrics, "rrf_rank_component", rank_component[candidate_mask])
    _add_distribution_metrics(
        metrics,
        "rrf_raw_rank_component",
        raw_rank_component[candidate_mask],
    )
    candidate_count = int(candidate_mask.sum().item())
    metrics["rrf_raw_available_candidate_rate"] = (
        float((raw_available & candidate_mask).sum().item()) / max(candidate_count, 1)
    )


def _add_distribution_metrics(
    metrics: dict[str, float],
    prefix: str,
    values: torch.Tensor,
) -> None:
    values = values.to(torch.float64).flatten()
    if values.numel() == 0:
        for suffix in ("mean", "std", "p25", "p50", "p75"):
            metrics[f"{prefix}_{suffix}"] = 0.0
        return
    metrics[f"{prefix}_mean"] = float(values.mean().item())
    metrics[f"{prefix}_std"] = float(values.std(unbiased=False).item())
    quantiles = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    for suffix, value in zip(("p25", "p50", "p75"), quantiles.tolist()):
        metrics[f"{prefix}_{suffix}"] = float(value)


def _add_trie_diagnostics(metrics: dict[str, float], trie: SIDTrie | dict[str, SIDTrie]) -> None:
    if isinstance(trie, dict):
        for branch, branch_trie in sorted(trie.items()):
            for key, value in branch_trie.collision_stats().items():
                metrics[f"{key}_{branch}"] = float(value)
        return
    for key, value in trie.collision_stats().items():
        metrics[key] = float(value)


def _add_beam_metric_aliases(metrics: dict[str, float], top_k: list[int]) -> None:
    """Expose method-facing beam_* names while keeping legacy Recall/NDCG keys."""
    for k in top_k:
        recall_key = f"Recall@{k}"
        ndcg_key = f"NDCG@{k}"
        if recall_key in metrics:
            metrics[f"beam_recall@{k}"] = float(metrics[recall_key])
        if ndcg_key in metrics:
            metrics[f"beam_ndcg@{k}"] = float(metrics[ndcg_key])
    if "BeamCoverage" in metrics:
        metrics["beam_coverage"] = float(metrics["BeamCoverage"])


def _add_branch_ranking_diagnostics(
    metrics: dict[str, float],
    sample_branch_candidates: list[dict[str, list[int]]],
    targets: list[int],
    top_k: list[int],
) -> None:
    """Add macro branch Top-K metrics; jaccard_all uses all-branch intersection/union."""
    if not sample_branch_candidates or len(sample_branch_candidates) != len(targets):
        return
    available_branches = {branch for candidates in sample_branch_candidates for branch in candidates}
    branches = sorted(available_branches)
    if len(branches) < 2:
        return

    denom = len(targets)
    for k in top_k:
        branch_hits = {branch: 0 for branch in branches}
        branch_ndcg = {branch: 0.0 for branch in branches}
        oracle_hits = 0
        jaccard_sum = 0.0
        for candidates_by_branch, target in zip(sample_branch_candidates, targets):
            topk_by_branch: dict[str, set[int]] = {}
            sample_oracle_hit = False
            for branch in branches:
                candidates = candidates_by_branch.get(branch, [])[:k]
                topk_by_branch[branch] = set(candidates)
                if target not in candidates:
                    continue
                rank = candidates.index(target) + 1
                branch_hits[branch] += 1
                branch_ndcg[branch] += 1.0 / math.log2(rank + 1)
                sample_oracle_hit = True
            oracle_hits += int(sample_oracle_hit)

            topk_sets = list(topk_by_branch.values())
            union = set().union(*topk_sets)
            if union:
                intersection = set.intersection(*topk_sets)
                jaccard_sum += len(intersection) / len(union)

        for branch in branches:
            suffix = _BRANCH_METRIC_SUFFIX.get(branch, branch)
            metrics[f"branch_hr_{suffix}@{k}"] = branch_hits[branch] / denom
            metrics[f"branch_ndcg_{suffix}@{k}"] = branch_ndcg[branch] / denom
        metrics[f"oracle_union@{k}"] = oracle_hits / denom
        metrics[f"jaccard_all@{k}"] = jaccard_sum / denom


@torch.inference_mode()
def evaluate_beam(
    model,
    dataloader: DataLoader,
    trie: SIDTrie | dict[str, SIDTrie],
    config: DotDict,
    device: torch.device,
    split: str = "valid",
    rrf_settings: RRFSettings | None = None,
    tune_rrf: bool = False,
    tune_metric: str | None = None,
    tune_seed: int | None = None,
    rrf_exhaustive: bool = False,
    raw_catalog: RawEmbeddingCatalog | None = None,
    beam_size: int | None = None,
    fixed_rrf_settings: list[RRFSettings] | None = None,
) -> BeamEvalOutput:
    evaluation_start = time.perf_counter()
    if fixed_rrf_settings is not None:
        if not fixed_rrf_settings or tune_rrf or not isinstance(trie, dict):
            raise ValueError("fixed RRF evaluation requires nonempty settings, multiple branches and no tuning")
        for setting in fixed_rrf_settings:
            _validate_rrf_settings(setting)
        rrf_settings = fixed_rrf_settings[0]
    if tune_rrf and split != "valid":
        raise ValueError("RRF tuning is validation-only; test must use checkpoint-selected settings")
    model.eval()
    top_k = get_top_k(config)
    max_k = max(top_k)
    if beam_size is None:
        beam_size = get_test_beam_size(config) if split == "test" else get_valid_beam_size(config)
    beam_size = int(beam_size)
    if beam_size <= 0:
        raise ValueError(f"beam_size must be positive, got {beam_size}")
    merge_branches = get_merge_branches(config)
    rrf_enabled = isinstance(trie, dict)
    rrf_branches = _active_rrf_branches(config, trie, merge_branches) if rrf_enabled else None
    active_rrf = get_rrf_settings(config, override=rrf_settings) if rrf_enabled else None
    settings_grid = None
    if tune_rrf and rrf_enabled and active_rrf is not None:
        settings_grid = get_rrf_search_grid(
            config,
            base_settings=active_rrf,
            seed=tune_seed,
            exhaustive=rrf_exhaustive,
        )
    settings_for_raw = settings_grid if settings_grid is not None else [active_rrf]
    if fixed_rrf_settings is not None:
        settings_for_raw = fixed_rrf_settings
    raw_scoring_required = any(
        setting is not None and setting.raw_compat_weight > 0.0
        for setting in settings_for_raw
    )
    if raw_scoring_required:
        if raw_catalog is None:
            raise ValueError("positive raw compatibility inference weight requires a RawEmbeddingCatalog")
        if not callable(getattr(model, "raw_history_summaries", None)):
            raise ValueError("positive raw compatibility inference weight requires model.raw_history_summaries")

    searcher = TrieBeamSearcher(
        trie=trie,
        beam_size=beam_size,
        max_items=max_k,
        merge_branches=merge_branches,
        rrf_k=60.0 if active_rrf is None else active_rrf.k,
        logits_chunk_size=get_beam_logits_chunk_size(config),
    )
    predictions: list[list[int]] = []
    candidates: list[list[int]] = []
    targets: list[int] = []
    target_presence_patterns: list[str] = []
    sample_branch_hit_maps: list[dict[str, int]] = []
    sample_branch_candidates: list[dict[str, list[int]]] = []
    branch_hits: dict[str, int] = {}
    sample_branch_ranks: list[dict[str, dict[int, int]]] = []
    sample_raw_branch_scores: list[dict[str, dict[int, float]]] = []
    sample_effective_branch_weights: list[dict[str, float]] = []
    max_batches = get_max_batches(config)
    single_branch_name = None if isinstance(trie, dict) else get_single_branch_name(config)

    progress: Iterable = tqdm(
        dataloader,
        desc=f"Beam eval {split}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch_idx, batch in enumerate(progress):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = _move_to_device(batch, device)
        with autocast_context(config, device):
            if raw_scoring_required:
                context = model.encode_batch_history(batch)
                results = searcher.search_batch(model, batch, context=context)
                results = _truncate_separate_candidates(results, config, beam_size)
                history_queries = model.raw_history_summaries(context)
                batch_raw_branch_scores, batch_effective_branch_weights = _score_raw_candidates(
                    model,
                    raw_catalog,
                    results,
                    history_queries,
                    batch["history_reliability"],
                )
            else:
                results = searcher.search_batch(model, batch)
                batch_raw_branch_scores = None
                batch_effective_branch_weights = None
        if batch_raw_branch_scores is not None:
            sample_raw_branch_scores.extend(batch_raw_branch_scores)
            sample_effective_branch_weights.extend(batch_effective_branch_weights)
        batch_targets = [int(x) for x in batch["target_item"].detach().cpu().tolist()]
        batch_presence_patterns = _target_presence_patterns(batch)
        predictions.extend(result.predictions for result in results)
        candidates.extend(result.candidates for result in results)
        for result, target in zip(results, batch_targets):
            sample_branch_candidates.append(
                {branch: list(branch_candidates) for branch, branch_candidates in result.branch_candidates.items()}
            )
            if rrf_enabled:
                sample_branch_ranks.append(result.branch_ranks)
            sample_hits: dict[str, int] = {}
            for branch, branch_candidates in result.branch_candidates.items():
                branch_hits.setdefault(branch, 0)
                hit = int(target in branch_candidates)
                branch_hits[branch] += hit
                sample_hits[branch] = hit
            if single_branch_name is not None:
                branch_hits.setdefault(single_branch_name, 0)
                hit = int(target in result.candidates)
                branch_hits[single_branch_name] += hit
                sample_hits[single_branch_name] = hit
            sample_branch_hit_maps.append(sample_hits)
        targets.extend(batch_targets)
        target_presence_patterns.extend(batch_presence_patterns)

    candidates_done = time.perf_counter()
    evidence_cache = None
    if (
        active_rrf is not None
        and len(sample_branch_ranks) == len(targets)
    ):
        evidence_cache = _RRFEvidenceCache.from_sparse(
            sample_branch_ranks,
            targets,
            rrf_branches or (),
            sample_raw_branch_scores=(
                sample_raw_branch_scores if raw_scoring_required else None
            ),
            sample_effective_branch_weights=(
                sample_effective_branch_weights if raw_scoring_required else None
            ),
            model_identity=id(model),
        )
        evidence_cache.assert_model(model)

    cache_done = time.perf_counter()
    if fixed_rrf_settings is not None:
        if evidence_cache is None:
            raise ValueError("fixed RRF evaluation requires cached branch evidence")
        fixed_metrics = []
        component_key = None
        components = None
        for setting in fixed_rrf_settings:
            key = evidence_cache.component_key(setting)
            if key != component_key:
                components = evidence_cache.components(setting)
                component_key = key
            row = evidence_cache.target_metrics(setting, top_k, components=components)
            _add_beam_metric_aliases(row, top_k)
            _add_rrf_metrics(row, setting, tuned=False, grid_size=1)
            row["n_eval"] = float(len(targets))
            fixed_metrics.append(row)
        return BeamEvalOutput(metrics=fixed_metrics[0], rrf_settings=active_rrf,
                              fixed_rrf_metrics=fixed_metrics)

    tuned = False
    grid_size = 1 if settings_grid is None else len(settings_grid)
    if (
        tune_rrf
        and rrf_enabled
        and active_rrf is not None
        and sample_branch_ranks
        and len(sample_branch_ranks) == len(targets)
    ):
        if settings_grid is None:
            raise RuntimeError("RRF tuning grid was not initialized")
        active_rrf, metrics, predictions, candidates = _tune_rrf_from_cached_ranks(
            sample_branch_ranks=sample_branch_ranks,
            targets=targets,
            top_k=top_k,
            max_items=max_k,
            settings_grid=settings_grid,
            metric=tune_metric or str(getattr(getattr(config, "rrf", None), "tune_metric", "NDCG@10")),
            sample_raw_branch_scores=(
                sample_raw_branch_scores if raw_scoring_required else None
            ),
            sample_effective_branch_weights=(
                sample_effective_branch_weights if raw_scoring_required else None
            ),
            evidence_cache=evidence_cache,
        )
        tuned = True
    elif active_rrf is not None and active_rrf.raw_compat_weight > 0.0:
        predictions, candidates = _rank_cached_branch_ranks(
            sample_branch_ranks,
            active_rrf,
            max_k,
            sample_raw_branch_scores=sample_raw_branch_scores,
            sample_effective_branch_weights=sample_effective_branch_weights,
        )
        metrics = compute_ranking_metrics(predictions, candidates, targets, top_k)
    else:
        metrics = compute_ranking_metrics(predictions, candidates, targets, top_k)

    ranking_done = time.perf_counter()
    if active_rrf is not None and evidence_cache is not None:
        _add_fusion_diagnostics(metrics, evidence_cache, active_rrf)

    for branch, hits in sorted(branch_hits.items()):
        coverage = hits / max(len(targets), 1)
        suffix = _BRANCH_METRIC_SUFFIX.get(branch, branch)
        metrics[f"beam_coverage_{suffix}"] = coverage
    _add_beam_metric_aliases(metrics, top_k)
    _add_branch_ranking_diagnostics(
        metrics,
        sample_branch_candidates,
        targets,
        top_k,
    )
    _add_ranking_bucket_metrics(
        metrics,
        predictions,
        candidates,
        targets,
        target_presence_patterns,
        sample_branch_hit_maps,
        top_k,
    )
    if rrf_branches is not None:
        _add_ranking_bucket_metrics(
            metrics,
            predictions,
            candidates,
            targets,
            _target_origin_patterns(sample_branch_hit_maps, rrf_branches),
            sample_branch_hit_maps,
            top_k,
        )
    if active_rrf is not None:
        _add_rrf_metrics(
            metrics,
            active_rrf,
            tuned,
            grid_size,
            exhaustive=rrf_exhaustive,
        )
    _add_trie_diagnostics(metrics, trie)
    diagnostics_done = time.perf_counter()
    metrics.update({
        "eval_beam_size": float(beam_size),
        "eval_candidate_seconds": candidates_done - evaluation_start,
        "eval_cache_seconds": cache_done - candidates_done,
        "eval_ranking_seconds": ranking_done - cache_done,
        "eval_diagnostics_seconds": diagnostics_done - ranking_done,
        "eval_total_seconds": diagnostics_done - evaluation_start,
    })
    metrics["n_eval"] = float(len(targets))
    recall_key = f"Recall@{max_k}"
    if metrics["BeamCoverage"] + 1.0e-12 < metrics[recall_key]:
        raise AssertionError(
            f"BeamCoverage={metrics['BeamCoverage']:.6f} is smaller than {recall_key}={metrics[recall_key]:.6f}"
        )
    return BeamEvalOutput(
        metrics=metrics,
        rrf_settings=active_rrf,
        rrf_tuned=tuned,
        rrf_search_exhaustive=bool(tuned and rrf_exhaustive),
    )


def _active_rrf_branches(
    config: DotDict,
    trie: SIDTrie | dict[str, SIDTrie],
    merge_branches: list[str] | None,
) -> list[str] | None:
    if merge_branches is not None:
        return merge_branches
    if isinstance(trie, dict):
        return [str(branch) for branch in trie]
    return [get_single_branch_name(config)]


def _target_presence_patterns(batch: dict) -> list[str]:
    presence = batch.get("target_presence")
    if not isinstance(presence, torch.Tensor):
        return []
    rows = presence.detach().cpu().bool().tolist()
    return [f"presence_{int(row[0])}{int(row[1])}" for row in rows]


def _target_origin_patterns(
    sample_branch_hit_maps: list[dict[str, int]],
    branches: Iterable[str],
) -> list[str]:
    ordered_branches = tuple(str(branch) for branch in branches)
    patterns = []
    for hit_map in sample_branch_hit_maps:
        parts = [
            f"{_BRANCH_METRIC_SUFFIX.get(branch, branch)}{int(hit_map.get(branch, 0))}"
            for branch in ordered_branches
        ]
        patterns.append("origin_" + "_".join(parts))
    return patterns


def _add_ranking_bucket_metrics(
    metrics: dict[str, float],
    predictions: list[list[int]],
    candidates: list[list[int]],
    targets: list[int],
    patterns: list[str],
    sample_branch_hit_maps: list[dict[str, int]],
    top_k: list[int],
) -> None:
    if not patterns or len(patterns) != len(targets):
        return
    for pattern in sorted(set(patterns)):
        indices = [idx for idx, value in enumerate(patterns) if value == pattern]
        if not indices:
            continue
        bucket_predictions = [predictions[idx] for idx in indices]
        bucket_candidates = [candidates[idx] for idx in indices]
        bucket_targets = [targets[idx] for idx in indices]
        bucket_metrics = compute_ranking_metrics(bucket_predictions, bucket_candidates, bucket_targets, top_k)
        metrics[f"{pattern}_n_eval"] = float(len(indices))
        for key, value in bucket_metrics.items():
            metrics[f"{pattern}_{key}"] = float(value)
        branch_hits: dict[str, int] = {}
        for idx in indices:
            for branch, hit in sample_branch_hit_maps[idx].items():
                branch_hits[branch] = branch_hits.get(branch, 0) + int(hit)
        for branch, hits in sorted(branch_hits.items()):
            suffix = _BRANCH_METRIC_SUFFIX.get(branch, branch)
            metrics[f"{pattern}_beam_coverage_{suffix}"] = hits / max(len(indices), 1)


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_move_to_device(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(child, device) for child in value)
    return value
