"""Rank-space candidate fusion and availability-aware raw-score aggregation."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class RRFItem:
    item_id: int
    score: float
    hit_count: int
    channel_ranks: dict[str, int]


@dataclass(frozen=True)
class RRFResult:
    predictions: list[int]
    candidates: list[int]
    scores: list[float]
    hit_counts: dict[int, int]
    ranked_items: list[RRFItem]


def aggregate_raw_compatibility_scores(
    branch_scores: Mapping[str, Mapping[int, float]],
    branch_weights: Mapping[str, float],
    candidate_ids: Iterable[int] | None = None,
) -> dict[int, float]:
    """Return the availability-aware weighted raw score for each candidate."""
    if set(branch_scores) != set(branch_weights):
        raise ValueError(
            "raw branch scores and weights must define the same branches; "
            f"got scores={sorted(branch_scores)!r}, weights={sorted(branch_weights)!r}"
        )

    weights: dict[str, float] = {}
    evidenced_items: set[int] = set()
    for branch in sorted(branch_scores):
        item_scores = branch_scores[branch]
        weight = float(branch_weights[branch])
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"raw branch weight for {branch!r} must be finite and non-negative")
        weights[branch] = weight
        for item_id, raw_score in item_scores.items():
            item_id = int(item_id)
            raw_score = float(raw_score)
            if not math.isfinite(raw_score):
                raise ValueError("raw branch scores must contain only finite values")
            evidenced_items.add(item_id)

    candidates = (
        sorted(evidenced_items)
        if candidate_ids is None
        else list(dict.fromkeys(int(item_id) for item_id in candidate_ids))
    )
    candidate_set = set(candidates)
    unexpected = sorted(evidenced_items.difference(candidate_set))
    if unexpected:
        raise ValueError(
            "raw compatibility evidence must be restricted to the beam candidate union; "
            f"unexpected item ids: {unexpected[:10]}"
        )
    if sum(weights.values()) == 0.0:
        return {item_id: 0.0 for item_id in candidates}

    numerator = {item_id: 0.0 for item_id in candidates}
    denominator = {item_id: 0.0 for item_id in candidates}
    for branch in sorted(branch_scores):
        item_scores = branch_scores[branch]
        weight = weights[branch]
        if weight == 0.0:
            continue
        for item_id, raw_score in item_scores.items():
            item_id = int(item_id)
            numerator[item_id] += weight * float(raw_score)
            denominator[item_id] += weight

    missing = [item_id for item_id in candidates if denominator[item_id] == 0.0]
    if missing:
        raise ValueError(
            "non-empty history candidate has no effective raw compatibility evidence; "
            f"missing item ids: {missing[:10]}"
        )

    return {
        item_id: numerator[item_id] / denominator[item_id]
        for item_id in candidates
    }


def competition_ranks(scores: Mapping[int, float]) -> dict[int, int]:
    """Return descending competition ranks, so equal scores share a rank."""
    score_by_item = {int(item_id): float(score) for item_id, score in scores.items()}
    if not all(math.isfinite(score) for score in score_by_item.values()):
        raise ValueError("rank scores must contain only finite values")

    ranked = sorted(score_by_item.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[int, int] = {}
    previous_score: float | None = None
    current_rank = 0
    for position, (item_id, score) in enumerate(ranked, start=1):
        if previous_score is None or score != previous_score:
            current_rank = position
            previous_score = score
        ranks[item_id] = current_rank
    return ranks


def rrf_rank(
    branch_ranks: Mapping[str, Mapping[int, int]],
    max_items: int,
    k_rrf: float = 60.0,
    *,
    rank_fusion_weight: float = 1.0,
    raw_compat_weight: float = 0.0,
    raw_scores: Mapping[int, float] | None = None,
) -> RRFResult:
    """Fuse branch ranks and one raw-compatibility rank list in rank space."""
    if max_items <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}")
    if k_rrf <= 0.0:
        raise ValueError(f"k_rrf must be positive, got {k_rrf}")
    if rank_fusion_weight < 0.0:
        raise ValueError(f"rank_fusion_weight must be non-negative, got {rank_fusion_weight}")
    if raw_compat_weight < 0.0:
        raise ValueError(f"raw_compat_weight must be non-negative, got {raw_compat_weight}")
    if rank_fusion_weight == 0.0 and raw_compat_weight == 0.0:
        raise ValueError("rank_fusion_weight and raw_compat_weight must not both be zero")

    ranks_by_item: dict[int, dict[str, int]] = {}
    branch_score_by_item: dict[int, float] = {}
    for branch, item_ranks in branch_ranks.items():
        for item_id, rank in item_ranks.items():
            item_id = int(item_id)
            rank = int(rank)
            if rank <= 0:
                raise ValueError(f"RRF rank must be positive for item {item_id}, got {rank}")
            ranks_by_item.setdefault(item_id, {})[branch] = rank
            branch_score_by_item[item_id] = (
                branch_score_by_item.get(item_id, 0.0) + 1.0 / (k_rrf + rank)
            )

    raw_ranks: dict[int, int] | None = None
    if raw_compat_weight > 0.0:
        if raw_scores is None:
            raise ValueError("raw_scores are required when raw_compat_weight is positive")
        raw_score_by_item = {int(item_id): float(score) for item_id, score in raw_scores.items()}
        missing = sorted(set(ranks_by_item).difference(raw_score_by_item))
        if missing:
            raise ValueError(
                "every beam candidate must have aggregated raw compatibility evidence; "
                f"missing item ids: {missing[:10]}"
            )
        unexpected = sorted(set(raw_score_by_item).difference(ranks_by_item))
        if unexpected:
            raise ValueError(
                "raw compatibility evidence must be restricted to the beam candidate union; "
                f"unexpected item ids: {unexpected[:10]}"
            )
        raw_ranks = competition_ranks(raw_score_by_item)

    ranked_items: list[RRFItem] = []
    hit_counts: dict[int, int] = {}
    for item_id, channel_ranks in ranks_by_item.items():
        hit_count = len(channel_ranks)
        hit_counts[item_id] = hit_count
        raw_rrf = 0.0 if raw_ranks is None else 1.0 / (k_rrf + raw_ranks[item_id])
        score = (
            rank_fusion_weight * branch_score_by_item[item_id]
            + raw_compat_weight * raw_rrf
        )
        ranked_items.append(
            RRFItem(
                item_id=item_id,
                score=score,
                hit_count=hit_count,
                channel_ranks=dict(channel_ranks),
            )
        )

    ranked_items.sort(key=lambda item: (-item.score, item.item_id))
    candidates = [item.item_id for item in ranked_items]
    scores = [item.score for item in ranked_items]
    return RRFResult(
        predictions=candidates[:max_items],
        candidates=candidates,
        scores=scores,
        hit_counts=hit_counts,
        ranked_items=ranked_items,
    )
