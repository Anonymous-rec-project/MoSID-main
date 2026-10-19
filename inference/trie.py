"""SID prefix trie."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from utils.data import sid_codes_filename


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    item_ids: list[int] = field(default_factory=list)
    item_entries: list["TrieLeafEntry"] = field(default_factory=list)
    node_id: int = -1


@dataclass(frozen=True)
class TrieLeafEntry:
    item_id: int


class SIDTrie:
    """Build prefix constraints from raw SID artifacts.

    token_offset=0 keeps artifact-domain tokens (1..V).
    token_offset=-1 converts valid tokens to output-domain codes (0..V-1).
    """

    def __init__(
        self,
        sid_codes: Tensor,
        token_offset: int = 0,
    ):
        if sid_codes.dim() != 2:
            raise ValueError(f"sid_codes must be a 2D tensor, got {tuple(sid_codes.shape)}")
        if int(token_offset) not in {0, -1}:
            raise ValueError(f"token_offset must be 0 or -1, got {token_offset}")
        self.token_offset = int(token_offset)
        self.observed_artifact_sid_codes = sid_codes.long().cpu()
        self.artifact_sid_codes = self.observed_artifact_sid_codes
        self.sid_codes = self._to_trie_domain(self.artifact_sid_codes)
        self.depth = int(sid_codes.shape[1])
        indexed_codes = self._indexed_codes()
        if indexed_codes.numel() == 0:
            raise ValueError("SIDTrie has no valid SID code")
        self.vocab_size = int(indexed_codes.max().item()) + 1
        self.root = TrieNode(node_id=0)
        self.nodes: list[TrieNode] = [self.root]
        self.node_items: dict[int, list[int]] = {}
        self.node_entries: dict[int, list[TrieLeafEntry]] = {}
        self._build()
        self.leaf_size_by_item = self._build_leaf_size_by_item()
        self._build_transition_tensors()

    def to_trie_domain(self, artifact_sid: Tensor) -> Tensor:
        """Convert artifact-domain SID (`1..V`, zero as padding) to this trie's token domain."""
        return self._to_trie_domain(artifact_sid.long().cpu())

    def to_artifact_domain(self, trie_sid: Tensor) -> Tensor:
        """Convert this trie's token domain back to artifact-domain SID for diagnostics."""
        trie_sid = trie_sid.long().cpu()
        out = torch.zeros_like(trie_sid)
        valid = trie_sid >= 0
        out[valid] = trie_sid[valid] - self.token_offset
        return out

    def items_for_node(self, node_id: int) -> list[int]:
        """Return item ids for a complete SID leaf node."""
        return self.node_items.get(int(node_id), [])

    def leaf_entries_for_node(self, node_id: int) -> list[TrieLeafEntry]:
        """Return item entries for a complete SID leaf."""
        return list(self.node_entries.get(int(node_id), []))

    def node_id_for_sid(self, sid: Tensor) -> int:
        """Return the leaf node id for a trie-domain SID, or -1 when absent."""
        if sid.numel() != self.depth:
            raise ValueError(f"sid depth should be {self.depth}, got {sid.numel()}")
        node = self.root
        for token in [int(value) for value in sid.long().view(-1).tolist()]:
            child = node.children.get(token)
            if child is None:
                return -1
            node = child
        return int(node.node_id)

    def collision_stats(self) -> dict[str, float]:
        """Return leaf collision diagnostics."""
        leaf_sizes = [len(items) for items in self.node_items.values()]
        if not leaf_sizes:
            return {
                "sid_unique_ratio": 0.0,
                "collision_count": 0.0,
                "avg_items_per_sid": 0.0,
                "max_items_per_sid": 0.0,
            }
        unique_count = sum(1 for size in leaf_sizes if size == 1)
        collision_count = sum(1 for size in leaf_sizes if size > 1)
        return {
            "sid_unique_ratio": unique_count / len(leaf_sizes),
            "collision_count": float(collision_count),
            "avg_items_per_sid": sum(leaf_sizes) / len(leaf_sizes),
            "max_items_per_sid": float(max(leaf_sizes)),
        }

    def tensors_for_device(self, device: torch.device, vocab_size: int) -> tuple[Tensor, Tensor]:
        """Return transition tensors aligned to the logits vocabulary."""
        if vocab_size < self.vocab_size:
            raise ValueError(f"logits vocab_size={vocab_size} is smaller than trie vocab_size={self.vocab_size}")
        allowed, next_node = self._dense_transition_tensors(vocab_size)
        return allowed.to(device), next_node.to(device)

    @property
    def allowed_mask(self) -> Tensor:
        """Materialize the legacy dense allowed mask on demand."""
        return self._dense_transition_tensors(self.vocab_size)[0]

    @property
    def next_node(self) -> Tensor:
        """Materialize the legacy dense next-node table on demand."""
        return self._dense_transition_tensors(self.vocab_size)[1]

    def _dense_transition_tensors(self, vocab_size: int) -> tuple[Tensor, Tensor]:
        n_nodes = len(self.nodes)
        allowed = torch.zeros(n_nodes, vocab_size, dtype=torch.bool)
        next_node = torch.full((n_nodes, vocab_size), -1, dtype=torch.long)
        if self.transition_tokens.numel() > 0:
            rows = torch.repeat_interleave(
                torch.arange(n_nodes, dtype=torch.long),
                self.transition_row_ptr[1:] - self.transition_row_ptr[:-1],
            )
            allowed[rows, self.transition_tokens] = True
            next_node[rows, self.transition_tokens] = self.transition_next_nodes
        return allowed, next_node

    def compact_tensors_for_device(
        self,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return O(edges) CSR transitions for memory-bounded beam search."""
        return (
            self.transition_row_ptr.to(device),
            self.transition_tokens.to(device),
            self.transition_next_nodes.to(device),
            self.transition_keys.to(device),
        )

    def _build(self) -> None:
        indexed_count = 0
        for item_id in range(1, self.observed_artifact_sid_codes.shape[0]):
            artifact_sid_tensor = self.observed_artifact_sid_codes[item_id]
            if not self._is_complete_sid(artifact_sid_tensor):
                continue
            sid_tensor = self._to_trie_domain(artifact_sid_tensor)
            self._insert_entry(sid_tensor, TrieLeafEntry(item_id=item_id))
            indexed_count += 1

        if indexed_count == 0:
            raise ValueError("SIDTrie did not index any valid item")

    def _insert_entry(self, sid_tensor: Tensor, entry: TrieLeafEntry) -> None:
        node = self._insert_sid(sid_tensor)
        if entry.item_id not in node.item_ids:
            node.item_ids.append(entry.item_id)
        if not any(existing.item_id == entry.item_id for existing in node.item_entries):
            node.item_entries.append(entry)
        self.node_items[node.node_id] = node.item_ids
        self.node_entries[node.node_id] = node.item_entries

    def _insert_sid(self, sid_tensor: Tensor) -> TrieNode:
        sid = tuple(int(x) for x in sid_tensor.tolist())
        node = self.root
        for token in sid:
            if token < 0:
                raise ValueError("Trie cannot index negative tokens")
            if self.token_offset == 0 and token == 0:
                raise ValueError("Trie cannot index token 0")
            if token not in node.children:
                node.children[token] = TrieNode(node_id=len(self.nodes))
                self.nodes.append(node.children[token])
            node = node.children[token]
        return node

    def _build_transition_tensors(self) -> None:
        row_ptr = [0]
        tokens: list[int] = []
        next_nodes: list[int] = []
        keys: list[int] = []
        for node in self.nodes:
            for token, child in sorted(node.children.items()):
                tokens.append(int(token))
                next_nodes.append(int(child.node_id))
                keys.append(int(node.node_id) * self.vocab_size + int(token))
            row_ptr.append(len(tokens))
        self.transition_row_ptr = torch.tensor(row_ptr, dtype=torch.long)
        self.transition_tokens = torch.tensor(tokens, dtype=torch.long)
        self.transition_next_nodes = torch.tensor(next_nodes, dtype=torch.long)
        self.transition_keys = torch.tensor(keys, dtype=torch.long)

    def _build_leaf_size_by_item(self) -> Tensor:
        leaf_size = torch.zeros(self.sid_codes.shape[0], dtype=torch.long)
        for items in self.node_items.values():
            size = len(items)
            for item_id in items:
                leaf_size[int(item_id)] = size
        return leaf_size

    def _to_trie_domain(self, artifact_sid: Tensor) -> Tensor:
        out = torch.zeros_like(artifact_sid)
        valid = artifact_sid > 0
        out[valid] = artifact_sid[valid] + self.token_offset
        if valid.any().item() and int(out[valid].min().item()) < 0:
            raise ValueError("token_offset produced negative valid trie tokens")
        return out

    def _indexed_codes(self) -> Tensor:
        return self.sid_codes[self.artifact_sid_codes > 0].reshape(-1)

    @staticmethod
    def _is_complete_sid(sid: Tensor) -> bool:
        return bool(sid.gt(0).all().item())


def load_sid_trie_from_artifacts(
    data_dir: str | Path,
    channel: str,
    *,
    token_offset: int = -1,
) -> SIDTrie:
    data_dir = Path(data_dir)
    sid_path = data_dir / sid_codes_filename(channel)
    sid_codes = torch.load(sid_path, map_location="cpu", weights_only=True)
    if not isinstance(sid_codes, Tensor):
        raise TypeError(f"{sid_path.name} must contain a tensor")
    return SIDTrie(sid_codes, token_offset=token_offset)
