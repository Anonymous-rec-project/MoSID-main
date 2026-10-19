"""Recommendation evaluation metrics."""
from __future__ import annotations

import math


def recall_at_k(predictions: list[list[int]], targets: list[int], k: int) -> float:
    if not targets:
        return 0.0
    hits = 0
    for pred, target in zip(predictions, targets):
        hits += int(target in pred[:k])
    return hits / len(targets)


def ndcg_at_k(predictions: list[list[int]], targets: list[int], k: int) -> float:
    if not targets:
        return 0.0
    total = 0.0
    for pred, target in zip(predictions, targets):
        topk = pred[:k]
        if target not in topk:
            continue
        rank = topk.index(target) + 1
        total += 1.0 / math.log2(rank + 1)
    return total / len(targets)


def beam_coverage(candidates: list[list[int]], targets: list[int]) -> float:
    if not targets:
        return 0.0
    hits = 0
    for cand, target in zip(candidates, targets):
        hits += int(target in cand)
    return hits / len(targets)


def compute_ranking_metrics(
    predictions: list[list[int]],
    candidates: list[list[int]],
    targets: list[int],
    top_k: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in top_k:
        metrics[f"Recall@{k}"] = recall_at_k(predictions, targets, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(predictions, targets, k)
    metrics["BeamCoverage"] = beam_coverage(candidates, targets)
    return metrics
