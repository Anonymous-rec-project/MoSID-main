"""Static exact-LCP hard negatives for MoSID raw compatibility training."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class PrefixHardNegative:
    item_id: int
    lcp_depth: int


@dataclass(frozen=True)
class PrefixSamplingResult:
    negatives_by_sample: tuple[tuple[PrefixHardNegative, ...], ...]
    requested_by_lcp: tuple[int, ...]
    available_by_lcp: tuple[int, ...]


@dataclass(frozen=True)
class SharedRawCandidates:
    candidate_ids: Tensor
    selected_by_lcp: tuple[int, ...]
    duplicates_skipped: int
    cap_dropped: int


@dataclass(frozen=True)
class _PrefixNode:
    item_ids: Tensor
    child_ranges: dict[int, tuple[int, int]]


class PrefixHardNegativeSampler:
    """Build one CPU prefix index and deterministically sample exact-LCP items."""

    def __init__(
        self,
        sid_table: Tensor,
        counts_by_lcp: Sequence[int],
        seed: int,
        allow_missing: bool = False,
    ) -> None:
        self.sid_table = self._validate_sid_table(sid_table, allow_missing=allow_missing)
        self.sid_depth = int(self.sid_table.shape[1])
        self.num_items = int(self.sid_table.shape[0] - 1)
        self.counts_by_lcp = self._validate_counts(counts_by_lcp, self.sid_depth)
        self.seed = int(seed)
        self.epoch = 0
        self._nodes = self._build_nodes()

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self.epoch = epoch

    def sample(self, target_items: Tensor) -> PrefixSamplingResult:
        targets = self._validate_targets(target_items)
        requested = [len(targets) * count for count in self.counts_by_lcp]
        available = [0] * (self.sid_depth + 1)
        negatives_by_sample: list[tuple[PrefixHardNegative, ...]] = []

        for target_item in targets:
            sampled: list[PrefixHardNegative] = []
            target_codes = tuple(int(code) for code in self.sid_table[target_item].tolist())
            if not all(target_codes):
                negatives_by_sample.append(())
                continue
            for lcp_depth, count in enumerate(self.counts_by_lcp):
                if count == 0:
                    continue
                item_ids = self._sample_depth(target_item, target_codes, lcp_depth, count)
                available[lcp_depth] += len(item_ids)
                sampled.extend(
                    PrefixHardNegative(item_id=item_id, lcp_depth=lcp_depth)
                    for item_id in item_ids
                )
            negatives_by_sample.append(tuple(sampled))

        return PrefixSamplingResult(
            negatives_by_sample=tuple(negatives_by_sample),
            requested_by_lcp=tuple(requested),
            available_by_lcp=tuple(available),
        )

    def _sample_depth(
        self,
        target_item: int,
        target_codes: tuple[int, ...],
        lcp_depth: int,
        count: int,
    ) -> list[int]:
        prefix = target_codes[:lcp_depth]
        node = self._nodes[lcp_depth][prefix]
        if lcp_depth < self.sid_depth:
            child_start, child_end = node.child_ranges[target_codes[lcp_depth]]
        else:
            target_position = int(
                torch.searchsorted(
                    node.item_ids,
                    torch.tensor(target_item, dtype=node.item_ids.dtype),
                ).item()
            )
            child_start, child_end = target_position, target_position + 1

        excluded_count = child_end - child_start
        eligible_count = int(node.item_ids.numel()) - excluded_count
        selected_count = min(int(count), eligible_count)
        if selected_count == 0:
            return []

        start = (self._stable_offset(target_item, lcp_depth) + self.epoch) % eligible_count
        selected: list[int] = []
        for request_index in range(selected_count):
            virtual_offset = (start + request_index) % eligible_count
            actual_offset = (
                virtual_offset
                if virtual_offset < child_start
                else virtual_offset + excluded_count
            )
            selected.append(int(node.item_ids[actual_offset].item()))
        return selected

    def _build_nodes(self) -> tuple[dict[tuple[int, ...], _PrefixNode], ...]:
        child_buckets: list[dict[tuple[int, ...], dict[int, list[int]]]] = [
            {} for _depth in range(self.sid_depth)
        ]
        collision_buckets: dict[tuple[int, ...], list[int]] = {}

        for item_id in range(1, self.num_items + 1):
            codes = tuple(int(code) for code in self.sid_table[item_id].tolist())
            if not all(codes):
                continue
            for depth in range(self.sid_depth):
                prefix = codes[:depth]
                children = child_buckets[depth].setdefault(prefix, {})
                children.setdefault(codes[depth], []).append(item_id)
            collision_buckets.setdefault(codes, []).append(item_id)

        nodes: list[dict[tuple[int, ...], _PrefixNode]] = []
        for depth in range(self.sid_depth):
            depth_nodes: dict[tuple[int, ...], _PrefixNode] = {}
            for prefix, children in child_buckets[depth].items():
                flattened: list[int] = []
                child_ranges: dict[int, tuple[int, int]] = {}
                for child_code in sorted(children):
                    start = len(flattened)
                    flattened.extend(children[child_code])
                    child_ranges[child_code] = (start, len(flattened))
                depth_nodes[prefix] = _PrefixNode(
                    item_ids=torch.tensor(flattened, dtype=torch.long),
                    child_ranges=child_ranges,
                )
            nodes.append(depth_nodes)

        nodes.append(
            {
                prefix: _PrefixNode(
                    item_ids=torch.tensor(item_ids, dtype=torch.long),
                    child_ranges={},
                )
                for prefix, item_ids in collision_buckets.items()
            }
        )
        return tuple(nodes)

    def _stable_offset(self, target_item: int, lcp_depth: int) -> int:
        mask = (1 << 64) - 1
        value = (
            (self.seed & mask)
            ^ ((int(target_item) * 0x9E3779B97F4A7C15) & mask)
            ^ ((int(lcp_depth) * 0xBF58476D1CE4E5B9) & mask)
        )
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return int(value ^ (value >> 31))

    @staticmethod
    def _validate_sid_table(sid_table: Tensor, allow_missing: bool = False) -> Tensor:
        if not isinstance(sid_table, Tensor) or sid_table.device.type != "cpu":
            raise TypeError("sid_table must be a CPU tensor")
        if sid_table.dtype == torch.bool or sid_table.is_floating_point() or sid_table.is_complex():
            raise TypeError(f"sid_table must use an integer dtype, got {sid_table.dtype}")
        if sid_table.dim() != 2 or sid_table.shape[0] < 2 or sid_table.shape[1] < 1:
            raise ValueError(f"sid_table shape must be [N+1, K], got {tuple(sid_table.shape)}")
        sid_table = sid_table.long()
        if sid_table[0].ne(0).any().item():
            raise ValueError("sid_table row0 must be all zero")
        rows = sid_table[1:]
        if allow_missing:
            rows = rows[~rows.eq(0).all(dim=1)]
        if rows.le(0).any().item():
            raise ValueError("sid_table item rows must contain only positive codes")
        return sid_table

    @staticmethod
    def _validate_counts(counts_by_lcp: Sequence[int], sid_depth: int) -> tuple[int, ...]:
        counts = tuple(counts_by_lcp)
        if len(counts) != sid_depth + 1:
            raise ValueError(
                f"counts_by_lcp length must equal sid_depth + 1 ({sid_depth + 1}), "
                f"got {len(counts)}"
            )
        for depth, count in enumerate(counts):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"counts_by_lcp[{depth}] must be a non-negative integer, got {count!r}")
        return counts

    def _validate_targets(self, target_items: Tensor) -> list[int]:
        if not isinstance(target_items, Tensor) or target_items.device.type != "cpu":
            raise TypeError("target_items must be a CPU tensor")
        if target_items.dim() != 1 or target_items.numel() == 0:
            raise ValueError(f"target_items must be non-empty [B], got {tuple(target_items.shape)}")
        if target_items.dtype == torch.bool or target_items.is_floating_point() or target_items.is_complex():
            raise TypeError(f"target_items must use an integer dtype, got {target_items.dtype}")
        targets = [int(item_id) for item_id in target_items.tolist()]
        if min(targets) < 1 or max(targets) > self.num_items:
            raise IndexError(f"target_items must be in [1, {self.num_items}]")
        return targets


class SeparatePrefixHardNegativeSampler(PrefixHardNegativeSampler):
    """Share each LCP quota across modality-local indexes, rotating branch priority.

    LCP is exact in the source modality. Deeper quotas are filled first; an item
    is emitted once per target even if multiple modalities retrieve it.
    """

    def __init__(self, sid_tables: dict[str, Tensor], counts_by_lcp: Sequence[int], seed: int):
        if not sid_tables:
            raise ValueError("separate prefix sampling requires SID tables")
        tables = [self._validate_sid_table(table, allow_missing=True) for table in sid_tables.values()]
        if any(table.shape != tables[0].shape for table in tables):
            raise ValueError("separate prefix SID tables must have identical item counts and depths")
        self.sid_depth = int(tables[0].shape[1])
        self.num_items = int(tables[0].shape[0] - 1)
        self.counts_by_lcp = self._validate_counts(counts_by_lcp, self.sid_depth)
        self.seed = int(seed)
        self.epoch = 0
        self.samplers = [PrefixHardNegativeSampler(table, counts_by_lcp, seed, allow_missing=True)
                         for table in tables]

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        for sampler in self.samplers:
            sampler.set_epoch(epoch)

    def sample(self, target_items: Tensor) -> PrefixSamplingResult:
        targets = self._validate_targets(target_items)
        available = [0] * len(self.counts_by_lcp)
        rows = []
        budget = sum(self.counts_by_lcp)
        for target in targets:
            seen = {target}
            selected = []
            for depth in range(self.sid_depth, -1, -1):
                count = self.counts_by_lcp[depth]
                if count == 0:
                    continue
                pools = []
                start = (self._stable_offset(target, depth) + self.epoch) % len(self.samplers)
                for offset in range(len(self.samplers)):
                    sampler = self.samplers[(start + offset) % len(self.samplers)]
                    codes = tuple(sampler.sid_table[target].tolist())
                    # Up to budget candidates suffice even after cross-depth deduplication.
                    pools.append(sampler._sample_depth(target, codes, depth, budget) if all(codes) else [])
                taken = 0
                for index in range(max(map(len, pools), default=0)):
                    for pool in pools:
                        if index < len(pool) and pool[index] not in seen:
                            item = pool[index]
                            seen.add(item)
                            selected.append(PrefixHardNegative(item, depth))
                            taken += 1
                            if taken == count:
                                break
                    if taken == count:
                        break
                available[depth] += taken
            rows.append(tuple(selected))
        return PrefixSamplingResult(tuple(rows), tuple(len(targets) * c for c in self.counts_by_lcp),
                                    tuple(available))


def random_replacement_candidates(
    positive_ids: Tensor, negative_count: int, num_items: int, seed: int, epoch: int,
) -> Tensor:
    """Keep batch positives and uniformly replace the extra budget without replacement."""
    positives = [int(item) for item in positive_ids.tolist()]
    excluded = set(positives)
    pool = [item for item in range(1, num_items + 1) if item not in excluded]
    # Local RNG leaves training/dropout RNG streams untouched. Batch identity
    # and epoch vary samples while repeated evaluations remain reproducible.
    rng = random.Random(f"{seed}:{epoch}:" + ",".join(map(str, positives)))
    negatives = rng.sample(pool, negative_count)
    return torch.tensor(positives + negatives, dtype=torch.long)


def merge_shared_raw_candidates(
    positive_ids: Tensor,
    sampling: PrefixSamplingResult,
    effective_batch_size: int,
) -> SharedRawCandidates:
    """Keep all positives, then add unique negatives in sample-round-robin order."""
    if not isinstance(positive_ids, Tensor) or positive_ids.device.type != "cpu":
        raise TypeError("positive_ids must be a CPU tensor")
    if positive_ids.dim() != 1 or positive_ids.numel() == 0:
        raise ValueError(f"positive_ids must be non-empty [P], got {tuple(positive_ids.shape)}")
    if (
        isinstance(effective_batch_size, bool)
        or not isinstance(effective_batch_size, int)
        or effective_batch_size <= 0
    ):
        raise ValueError(
            f"effective_batch_size must be a positive integer, got {effective_batch_size!r}"
        )
    candidates = [int(item_id) for item_id in positive_ids.tolist()]
    seen = set(candidates)
    selected_by_lcp = [0] * len(sampling.requested_by_lcp)
    duplicates_skipped = 0
    cap_dropped = 0
    negative_count = 0
    max_requests = max((len(items) for items in sampling.negatives_by_sample), default=0)

    for request_index in range(max_requests):
        for sample_negatives in sampling.negatives_by_sample:
            if request_index >= len(sample_negatives):
                continue
            negative = sample_negatives[request_index]
            if negative.item_id in seen:
                duplicates_skipped += 1
                continue
            if negative_count >= effective_batch_size:
                cap_dropped += 1
                continue
            seen.add(negative.item_id)
            candidates.append(negative.item_id)
            selected_by_lcp[negative.lcp_depth] += 1
            negative_count += 1

    return SharedRawCandidates(
        candidate_ids=torch.tensor(candidates, dtype=torch.long),
        selected_by_lcp=tuple(selected_by_lcp),
        duplicates_skipped=duplicates_skipped,
        cap_dropped=cap_dropped,
    )
