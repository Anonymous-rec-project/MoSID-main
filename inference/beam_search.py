"""Trie-constrained beam search for MoSID."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from inference.rank_fusion import rrf_rank
from inference.trie import SIDTrie


@dataclass
class BeamResult:
    predictions: list[int]
    candidates: list[int]
    scores: list[float]
    candidate_ranks: dict[int, int] = field(default_factory=dict)
    branch_candidates: dict[str, list[int]] = field(default_factory=dict)
    branch_ranks: dict[str, dict[int, int]] = field(default_factory=dict)
    branch_hit_counts: dict[int, int] = field(default_factory=dict)


class TrieBeamSearcher:
    """Trie beam search for MoSID stage modes."""

    def __init__(
        self,
        trie: SIDTrie | dict[str, SIDTrie],
        beam_size: int,
        max_items: int,
        merge_branches: list[str] | None = None,
        rrf_k: float = 60.0,
        logits_chunk_size: int = 0,
        renormalize_allowed_tokens: bool = True,
        sparse_transitions: bool = False,
    ):
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if isinstance(trie, dict) and not trie:
            raise ValueError("parallel trie dict must not be empty")
        if int(logits_chunk_size) < 0:
            raise ValueError("logits_chunk_size must be non-negative")
        self.trie = trie
        self.beam_size = beam_size
        self.max_items = max_items
        self.merge_branches = self._resolve_merge_branches(merge_branches)
        self.rrf_k = float(rrf_k)
        self.logits_chunk_size = int(logits_chunk_size)
        self.renormalize_allowed_tokens = bool(renormalize_allowed_tokens)
        self.sparse_transitions = bool(sparse_transitions)
        # Evaluation-scoped cache: shared tries are copied to the device once,
        # then released with this searcher before training resumes.
        self._transition_cache: dict[
            tuple[int, torch.device, int],
            tuple[Tensor, ...],
        ] = {}
        if isinstance(trie, dict):
            trie_ids = [id(value) for value in trie.values()]
            # Do not retain every dense transition table for separate artifacts;
            # cache only trie objects shared by multiple unified branches.
            self._cacheable_trie_ids = {
                trie_id
                for trie_id in set(trie_ids)
                if trie_ids.count(trie_id) > 1
            }
        else:
            self._cacheable_trie_ids = {id(trie)}

    @torch.inference_mode()
    def search_batch(
        self,
        model,
        batch: dict[str, Tensor],
        context=None,
    ) -> list[BeamResult]:
        device = batch["attention_mask"].device
        batch_size = batch["attention_mask"].shape[0]
        if context is None:
            context = model.encode_batch_history(batch)

        if isinstance(self.trie, dict):
            if hasattr(context, "index_select"):
                context_by_branch = {branch: context for branch in self.trie}
            else:
                context_by_branch = context
            if not isinstance(context_by_branch, dict):
                raise TypeError("parallel trie search requires model.encode_batch_history() to return a dict")
            branch_results = {}

            for branch in self.merge_branches:
                trie = self.trie[branch]

                def logits_fn(sample_ids: Tensor, prefix_tokens: Tensor, active_branch: str = branch) -> Tensor:
                    return _logits_for_prefix_beams(
                        model,
                        context_by_branch[active_branch],
                        sample_ids,
                        prefix_tokens,
                        active_branch,
                        self.logits_chunk_size,
                    )

                branch_results[branch] = _search_with_trie(
                    trie=trie,
                    beam_size=self.beam_size,
                    max_items=self.max_items,
                    batch_size=batch_size,
                    device=device,
                    logits_fn=logits_fn,
                    transition_cache=(
                        self._transition_cache
                        if id(trie) in self._cacheable_trie_ids
                        else None
                    ),
                    renormalize_allowed_tokens=self.renormalize_allowed_tokens,
                    sparse_transitions=self.sparse_transitions,
                )
            merged = _merge_branch_results(
                {branch: branch_results[branch] for branch in self.merge_branches},
                self.max_items,
                self.rrf_k,
            )
            return merged

        context_by_branch = context
        if isinstance(context_by_branch, dict):
            raise TypeError("single-head trie search should receive tensor context, got dict")

        single_channel = None
        active_channels = getattr(model, "active_channels", None)
        if active_channels is not None and len(active_channels) == 1:
            single_channel = str(active_channels[0])

        def single_logits_fn(sample_ids: Tensor, prefix_tokens: Tensor) -> Tensor:
            return _logits_for_prefix_beams(
                model,
                context_by_branch,
                sample_ids,
                prefix_tokens,
                single_channel,
                self.logits_chunk_size,
            )

        return _search_with_trie(
            trie=self.trie,
            beam_size=self.beam_size,
            max_items=self.max_items,
            batch_size=batch_size,
            device=device,
            logits_fn=single_logits_fn,
            transition_cache=self._transition_cache,
            renormalize_allowed_tokens=self.renormalize_allowed_tokens,
            sparse_transitions=self.sparse_transitions,
        )

    def _resolve_merge_branches(self, merge_branches: list[str] | None) -> list[str]:
        if not isinstance(self.trie, dict):
            if merge_branches is not None:
                raise ValueError("inference.merge_branches is only valid for multi-branch trie search")
            return []

        if merge_branches is None:
            return list(self.trie)
        if not merge_branches:
            raise ValueError("merge_branches must not be empty")
        missing = [branch for branch in merge_branches if branch not in self.trie]
        if missing:
            raise ValueError(
                f"merge_branches contains branches without trie artifacts: {missing}; "
                f"available branches: {list(self.trie)}"
            )
        return list(merge_branches)


def _logits_for_prefix_beams(
    model,
    context,
    sample_ids: Tensor,
    prefix_tokens: Tensor,
    branch: str | None,
    chunk_size: int,
) -> Tensor:
    if chunk_size <= 0 or sample_ids.numel() <= chunk_size or model.training:
        return _logits_for_prefix_chunk(model, context, sample_ids, prefix_tokens, branch)

    logits_chunks = []
    for start in range(0, sample_ids.numel(), chunk_size):
        end = min(start + chunk_size, sample_ids.numel())
        logits_chunks.append(
            _logits_for_prefix_chunk(
                model,
                context,
                sample_ids[start:end],
                prefix_tokens[start:end],
                branch,
            )
        )
    return torch.cat(logits_chunks, dim=0)


def _logits_for_prefix_chunk(
    model,
    context,
    sample_ids: Tensor,
    prefix_tokens: Tensor,
    branch: str | None,
) -> Tensor:
    indexed_logits = getattr(model, "logits_for_indexed_prefix_context", None)
    if callable(indexed_logits):
        return indexed_logits(
            context,
            sample_ids,
            prefix_tokens,
            branch=branch,
        ).float()
    if branch is not None and hasattr(context, "index_select_branch"):
        context_beam = context.index_select_branch(0, sample_ids, branch)
    else:
        context_beam = context.index_select(0, sample_ids)
    if branch is None:
        return model.logits_for_prefix_context(context_beam, prefix_tokens).float()
    return model.logits_for_prefix_context(
        context_beam,
        prefix_tokens,
        branch=branch,
    ).float()


def _search_with_trie(
    trie: SIDTrie,
    beam_size: int,
    max_items: int,
    batch_size: int,
    device: torch.device,
    logits_fn,
    transition_cache: dict[tuple[int, torch.device, int], tuple[Tensor, ...]] | None = None,
    renormalize_allowed_tokens: bool = True,
    sparse_transitions: bool = False,
) -> list[BeamResult]:
    sample_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    node_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    scores = torch.zeros(batch_size, dtype=torch.float, device=device)
    prefix_tokens = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
    transition_tensors: tuple[Tensor, Tensor] | None = None
    compact_tensors: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
    transition_vocab_size = -1

    for _step in range(trie.depth):
        if sample_ids.numel() == 0:
            return _empty_results(batch_size)

        logits = logits_fn(sample_ids, prefix_tokens)
        if sparse_transitions:
            if logits.shape[-1] < trie.vocab_size:
                raise ValueError(
                    f"logits vocab_size={logits.shape[-1]} is smaller than "
                    f"trie vocab_size={trie.vocab_size}"
                )
            if compact_tensors is None:
                cache_key = (id(trie), torch.device(device), -1)
                cached = None if transition_cache is None else transition_cache.get(cache_key)
                if cached is None:
                    cached = trie.compact_tensors_for_device(device)
                    if transition_cache is not None:
                        transition_cache[cache_key] = cached
                compact_tensors = cached  # type: ignore[assignment]
            row_allowed = _sparse_allowed_rows(
                node_ids,
                logits.shape[-1],
                compact_tensors,
            )
        else:
            if transition_tensors is None or transition_vocab_size != logits.shape[-1]:
                cache_key = (id(trie), torch.device(device), int(logits.shape[-1]))
                cached = None if transition_cache is None else transition_cache.get(cache_key)
                if cached is None:
                    cached = trie.tensors_for_device(device, logits.shape[-1])
                    if transition_cache is not None:
                        transition_cache[cache_key] = cached
                transition_tensors = cached  # type: ignore[assignment]
                transition_vocab_size = logits.shape[-1]
            allowed_mask, _next_node = transition_tensors
            row_allowed = allowed_mask.index_select(0, node_ids)
        if renormalize_allowed_tokens:
            masked_logits = logits.masked_fill(~row_allowed, float("-inf"))
            log_probs = F.log_softmax(masked_logits, dim=-1)
        else:
            # Hugging Face beam search computes full-vocabulary log-probabilities
            # before its prefix constraint masks invalid continuations.
            log_probs = F.log_softmax(logits, dim=-1).masked_fill(
                ~row_allowed,
                float("-inf"),
            )
        total_scores = log_probs + scores.unsqueeze(1)

        row_has_next = row_allowed.any(dim=1)
        if not row_has_next.all().item():
            total_scores = total_scores.masked_fill(~row_has_next.unsqueeze(1), float("-inf"))

        flat_scores = total_scores.reshape(-1)
        finite = torch.isfinite(flat_scores)
        if not finite.any().item():
            return _empty_results(batch_size)

        valid_flat = finite.nonzero(as_tuple=True)[0]
        candidate_scores = flat_scores.index_select(0, valid_flat)
        vocab_size = logits.shape[-1]
        candidate_rows = torch.div(valid_flat, vocab_size, rounding_mode="floor")
        candidate_tokens = valid_flat.remainder(vocab_size)
        candidate_samples = sample_ids.index_select(0, candidate_rows)
        candidate_parent_nodes = node_ids.index_select(0, candidate_rows)
        if sparse_transitions:
            candidate_nodes = _sparse_next_nodes(
                candidate_parent_nodes,
                candidate_tokens,
                trie.vocab_size,
                compact_tensors,
            )
        else:
            _allowed_mask, next_node = transition_tensors
            candidate_nodes = next_node[candidate_parent_nodes, candidate_tokens]
        candidate_prefixes = torch.cat(
            [prefix_tokens.index_select(0, candidate_rows), candidate_tokens.unsqueeze(1)],
            dim=1,
        )

        sample_ids, node_ids, scores, prefix_tokens = _select_top_beams(
            beam_size=beam_size,
            batch_size=batch_size,
            candidate_samples=candidate_samples,
            candidate_nodes=candidate_nodes,
            candidate_scores=candidate_scores,
            candidate_prefixes=candidate_prefixes,
        )

    return _finalize_batch(
        trie=trie,
        max_items=max_items,
        batch_size=batch_size,
        sample_ids=sample_ids,
        node_ids=node_ids,
        scores=scores,
    )


def _sparse_allowed_rows(
    node_ids: Tensor,
    vocab_size: int,
    compact_tensors: tuple[Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    row_ptr, tokens, _next_nodes, _keys = compact_tensors
    starts = row_ptr.index_select(0, node_ids)
    ends = row_ptr.index_select(0, node_ids + 1)
    counts = ends - starts
    allowed = torch.zeros(
        node_ids.shape[0],
        vocab_size,
        dtype=torch.bool,
        device=node_ids.device,
    )
    total_edges = int(counts.sum().item())
    if total_edges == 0:
        return allowed
    rows = torch.repeat_interleave(
        torch.arange(node_ids.shape[0], device=node_ids.device),
        counts,
    )
    group_starts = counts.cumsum(dim=0) - counts
    edge_offsets = torch.arange(total_edges, device=node_ids.device) - torch.repeat_interleave(
        group_starts,
        counts,
    )
    edge_positions = starts.index_select(0, rows) + edge_offsets
    edge_tokens = tokens.index_select(0, edge_positions)
    allowed[rows, edge_tokens] = True
    return allowed


def _sparse_next_nodes(
    parent_nodes: Tensor,
    tokens: Tensor,
    trie_vocab_size: int,
    compact_tensors: tuple[Tensor, Tensor, Tensor, Tensor] | None,
) -> Tensor:
    if compact_tensors is None:
        raise AssertionError("compact trie transitions were not initialized")
    _row_ptr, _tokens, next_nodes, keys = compact_tensors
    candidate_keys = parent_nodes * int(trie_vocab_size) + tokens
    positions = torch.searchsorted(keys, candidate_keys)
    if positions.ge(keys.numel()).any().item():
        raise AssertionError("candidate beam transition is absent from the trie")
    matched = keys.index_select(0, positions)
    if not torch.equal(matched, candidate_keys):
        raise AssertionError("candidate beam transition is absent from the trie")
    return next_nodes.index_select(0, positions)


def _select_top_beams(
    beam_size: int,
    batch_size: int,
    candidate_samples: Tensor,
    candidate_nodes: Tensor,
    candidate_scores: Tensor,
    candidate_prefixes: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if candidate_scores.numel() == 0:
        device = candidate_scores.device
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float, device=device),
            torch.empty(0, candidate_prefixes.shape[1], dtype=torch.long, device=device),
        )

    # First establish the old per-sample stable score order, then group samples
    # with another stable sort. This preserves the flattened-order tie break.
    score_order = torch.argsort(candidate_scores, descending=True, stable=True)
    score_sorted_samples = candidate_samples.index_select(0, score_order)
    sample_order = torch.argsort(score_sorted_samples, stable=True)
    chosen = score_order.index_select(0, sample_order)
    grouped_samples = candidate_samples.index_select(0, chosen)

    counts = torch.bincount(grouped_samples, minlength=batch_size)
    group_starts = counts.cumsum(dim=0) - counts
    repeated_starts = torch.repeat_interleave(group_starts, counts)
    ranks = torch.arange(chosen.numel(), device=chosen.device) - repeated_starts
    chosen = chosen[ranks < beam_size]

    return (
        candidate_samples.index_select(0, chosen),
        candidate_nodes.index_select(0, chosen),
        candidate_scores.index_select(0, chosen),
        candidate_prefixes.index_select(0, chosen),
    )


def _finalize_batch(
    trie: SIDTrie,
    max_items: int,
    batch_size: int,
    sample_ids: Tensor,
    node_ids: Tensor,
    scores: Tensor,
) -> list[BeamResult]:
    sample_ids_cpu = sample_ids.detach().cpu().tolist()
    node_ids_cpu = node_ids.detach().cpu().tolist()
    scores_cpu = scores.detach().cpu().tolist()

    beams_by_sample: list[list[tuple[int, float]]] = [[] for _ in range(batch_size)]
    for sample_id, node_id, score in zip(sample_ids_cpu, node_ids_cpu, scores_cpu):
        beams_by_sample[sample_id].append((node_id, float(score)))

    return [
        _finalize_beams(trie, max_items, beams)
        for beams in beams_by_sample
    ]


def _finalize_beams(
    trie: SIDTrie,
    max_items: int,
    beams: list[tuple[int, float]],
) -> BeamResult:
    ranked_beams = sorted(beams, key=lambda beam: (-beam[1], beam[0]))
    best_by_item: dict[int, tuple[int, float]] = {}
    for beam_rank, (node_id, score) in enumerate(ranked_beams, start=1):
        for entry in trie.leaf_entries_for_node(node_id):
            item_id = int(entry.item_id)
            current = best_by_item.get(item_id)
            if current is None or beam_rank < current[0] or (beam_rank == current[0] and score > current[1]):
                best_by_item[item_id] = (beam_rank, float(score))

    ordered_items = sorted(
        best_by_item.items(),
        key=lambda item: (item[1][0], -item[1][1], item[0]),
    )
    candidates = [item_id for item_id, _meta in ordered_items]
    scores = [meta[1] for _item_id, meta in ordered_items]
    candidate_ranks = {item_id: meta[0] for item_id, meta in ordered_items}

    return BeamResult(
        predictions=candidates[:max_items],
        candidates=candidates,
        scores=scores,
        candidate_ranks=candidate_ranks,
    )


def _merge_branch_results(
    branch_results: dict[str, list[BeamResult]],
    max_items: int,
    rrf_k: float,
) -> list[BeamResult]:
    branches = list(branch_results)
    batch_size = len(next(iter(branch_results.values())))
    merged: list[BeamResult] = []

    for sample_idx in range(batch_size):
        branch_candidates = {
            branch: list(branch_results[branch][sample_idx].candidates)
            for branch in branches
        }
        branch_ranks = {
            branch: dict(branch_results[branch][sample_idx].candidate_ranks)
            for branch in branches
        }
        rrf_result = rrf_rank(
            branch_ranks=branch_ranks,
            max_items=max_items,
            k_rrf=rrf_k,
        )
        merged.append(
            BeamResult(
                predictions=rrf_result.predictions,
                candidates=rrf_result.candidates,
                scores=rrf_result.scores,
                candidate_ranks={item_id: idx for idx, item_id in enumerate(rrf_result.candidates, start=1)},
                branch_candidates=branch_candidates,
                branch_ranks=branch_ranks,
                branch_hit_counts=rrf_result.hit_counts,
            )
        )

    return merged


def _empty_results(batch_size: int) -> list[BeamResult]:
    return [
        BeamResult(predictions=[], candidates=[], scores=[])
        for _ in range(batch_size)
    ]
