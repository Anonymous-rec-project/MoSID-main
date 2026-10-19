"""MoSID Stage 2 sequence dataset with stable SID batch schema."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from configs.config import get_mod_presence_filename
from utils.data import (
    MPSID_INPUT_MODALITIES,
    mosid_sid_artifact_channel,
    mosid_uses_unified_block_views,
    sid_codes_filename,
)
from utils.io import load_pkl


DEFAULT_CHANNELS = ("text", "vision", "collab")
CONTENT_PRESENCE_INDEX = {"text": 0, "vision": 1}


def compute_history_modality_reliability(
    item_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    mod_presence: torch.Tensor,
    use_history_reliability: bool = True,
) -> dict[str, torch.Tensor]:
    """Return collab/text/vision reliability for one padded history."""
    valid_items = item_tensor[attention_mask.bool()]
    if valid_items.numel() == 0:
        return {
            branch: torch.zeros((), dtype=torch.float32, device=mod_presence.device)
            for branch in MPSID_INPUT_MODALITIES
        }

    if not use_history_reliability:
        return {
            branch: torch.ones((), dtype=torch.float32, device=mod_presence.device)
            for branch in MPSID_INPUT_MODALITIES
        }
    observed_rates = mod_presence.index_select(0, valid_items.long()).float().mean(dim=0)
    return {
        "collab": observed_rates.new_ones(()),
        "text": observed_rates[CONTENT_PRESENCE_INDEX["text"]].sqrt(),
        "vision": observed_rates[CONTENT_PRESENCE_INDEX["vision"]].sqrt(),
    }


class MoSIDSequenceDataset(Dataset):
    """Load sequence-to-label samples and configured SID artifacts for MoSID."""

    def __init__(self, config: DotDict, split: str):
        super().__init__()
        if split not in ("train", "valid", "test"):
            raise ValueError(f"unknown split: {split}")

        self.config = config
        self.split = split
        self.max_len = int(config.data.max_len)
        if self.max_len <= 0:
            raise ValueError(f"data.max_len must be positive, got {self.max_len}")
        self.data_dir = Path(config.data._data_dir)
        self.channels = tuple(str(channel) for channel in getattr(config.model, "active_channels", DEFAULT_CHANNELS))
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError(f"model.active_channels must be a non-empty unique channel list, got {self.channels!r}")
        self.use_unified_block_views = mosid_uses_unified_block_views(config)

        self.n_items = self._load_n_items()
        self.mod_presence = self._load_mod_presence()
        if self.mod_presence.shape[0] != self.n_items + 1:
            raise ValueError(
                f"mod_presence rows {self.mod_presence.shape[0]} do not match mappings N+1={self.n_items + 1}"
            )

        self.sid_tables = self._load_sid_tables()
        self.sid_depths = {channel: int(sid.shape[1]) for channel, sid in self.sid_tables.items()}
        availability_by_table: dict[int, torch.Tensor] = {}
        self.sid_available = {}
        for channel, sid in self.sid_tables.items():
            table_key = id(sid)
            if table_key not in availability_by_table:
                availability_by_table[table_key] = sid.gt(0).all(dim=-1)
            self.sid_available[channel] = availability_by_table[table_key]
        self.samples = self._load_samples(split)
        self.unique_items = len({item for seq, target in self.samples for item in [*seq, target]})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item_seq, target_item = self.samples[idx]
        item_tensor, attention_mask = self._pad_item_seq(item_seq)
        target_tensor = torch.tensor(target_item, dtype=torch.long)

        history_sid = {}
        history_available = {}
        target_sid = {}
        target_available = {}
        for channel, sid in self.sid_tables.items():
            history_sid[channel] = sid[item_tensor]
            history_available[channel] = self._history_available(channel, item_tensor, attention_mask)
            target_sid[channel] = sid[target_item].clone()
            target_available[channel] = self._channel_target_available(channel, target_item)

        return {
            "item_seq": item_tensor,
            "attention_mask": attention_mask,
            "history_sid": history_sid,
            "history_available": history_available,
            "history_presence": self.mod_presence[item_tensor].clone(),
            "history_reliability": compute_history_modality_reliability(
                item_tensor,
                attention_mask,
                self.mod_presence,
                use_history_reliability=bool(getattr(
                    getattr(self.config, "loss", None), "raw_use_history_reliability", True,
                )),
            ),
            "target_item": target_tensor,
            "target_sid": target_sid,
            "target_available": target_available,
            "target_presence": self.mod_presence[target_item].clone(),
        }

    def summary(self) -> dict[str, Any]:
        summary = {
            "split": self.split,
            "samples": len(self.samples),
            "max_len": self.max_len,
            "n_items": self.n_items,
            "n_items_with_padding": self.n_items + 1,
            "sid_depths": dict(self.sid_depths),
            "unique_items": self.unique_items,
        }
        return summary

    def _load_n_items(self) -> int:
        mappings = load_pkl(self.data_dir / "mappings.pkl")
        if not isinstance(mappings, dict) or "item2id" not in mappings:
            raise KeyError("mappings.pkl must contain item2id")
        item2id = mappings["item2id"]
        if not isinstance(item2id, dict) or not item2id:
            raise ValueError("mappings['item2id'] must be a non-empty dict")
        mapped_ids = sorted(int(value) for value in item2id.values())
        expected = list(range(1, len(mapped_ids) + 1))
        if mapped_ids != expected:
            raise ValueError("mapped_id in mappings.pkl must be contiguous 1..N")
        return len(mapped_ids)

    def _load_mod_presence(self) -> torch.Tensor:
        path = self.data_dir / get_mod_presence_filename(self.config)
        presence = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(presence, torch.Tensor):
            raise TypeError(f"{path.name} must be a tensor")
        if presence.dtype != torch.bool:
            raise TypeError(f"{path.name} dtype must be bool, got {presence.dtype}")
        if presence.ndim != 2 or presence.shape[1] != 2:
            raise ValueError(f"{path.name} shape must be [N+1, 2], got {tuple(presence.shape)}")
        if presence[0].any().item():
            raise ValueError(f"{path.name} row0 must be all False")
        return presence

    def _load_sid_tables(self) -> dict[str, torch.Tensor]:
        sid_tables: dict[str, torch.Tensor] = {}
        loaded_by_path: dict[Path, torch.Tensor] = {}
        for channel in self.channels:
            artifact_channel = mosid_sid_artifact_channel(self.config, channel)
            path = self.data_dir / sid_codes_filename(artifact_channel)
            sid = loaded_by_path.get(path)
            if sid is None:
                sid = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(sid, torch.Tensor):
                    loaded_by_path[path] = sid
            validated = self._validate_sid_table(channel, sid, path)
            sid_tables[channel] = validated
            if self.use_unified_block_views:
                loaded_by_path[path] = validated
        return sid_tables

    def _validate_sid_table(self, channel: str, sid: Any, path: Path) -> torch.Tensor:
        if not isinstance(sid, torch.Tensor):
            raise TypeError(f"{path.name} must be a tensor")
        if sid.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            raise TypeError(f"{path.name} dtype must be integer, got {sid.dtype}")
        if sid.ndim != 2:
            raise ValueError(f"{path.name} shape must be [N+1, K], got {tuple(sid.shape)}")
        if sid.shape[0] != self.n_items + 1:
            raise ValueError(f"{path.name} rows {sid.shape[0]} do not match mappings N+1={self.n_items + 1}")
        if sid.shape[1] < 1:
            raise ValueError(f"{path.name} must contain at least one SID layer")
        if sid[0].ne(0).any().item():
            raise ValueError(f"{path.name} row0 must be all zero")

        sid = sid.long()
        if sid.lt(0).any().item():
            raise ValueError(f"{channel} SID codes must be non-negative")
        if self.use_unified_block_views:
            valid_mask = torch.ones(self.n_items + 1, dtype=torch.bool)
            valid_mask[0] = False
        elif channel == "collab":
            valid_mask = torch.ones(self.n_items + 1, dtype=torch.bool)
            valid_mask[0] = False
        elif channel in CONTENT_PRESENCE_INDEX:
            valid_mask = self.mod_presence[:, CONTENT_PRESENCE_INDEX[channel]].clone()
            valid_mask[0] = False
            # Presence describes observed raw features, not SID availability.
            # Stage 1 pre-completion may export complete codes for missing features;
            # sid_available below determines their history/target eligibility.
        else:
            valid_mask = torch.ones(self.n_items + 1, dtype=torch.bool)
            valid_mask[0] = False

        if sid[valid_mask].le(0).any().item():
            raise ValueError(f"{channel} SID valid rows must use codes in 1..V")
        partial_zero_rows = sid.ne(0).any(dim=1) & sid.eq(0).any(dim=1)
        if partial_zero_rows.any().item():
            raise ValueError(f"{channel} SID rows must be either all zero or all valid codes")
        return sid

    def _pad_item_seq(self, item_seq: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        item_seq = item_seq[-self.max_len:]
        pad_len = self.max_len - len(item_seq)
        padded_items = [0] * pad_len + item_seq
        item_tensor = torch.tensor(padded_items, dtype=torch.long)
        return item_tensor, item_tensor != 0

    def _history_available(
        self,
        channel: str,
        item_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_unified_block_views:
            return attention_mask & self.sid_available[channel][item_tensor]
        if channel == "collab":
            return attention_mask.clone()
        return attention_mask & self.sid_available[channel][item_tensor]

    def _channel_target_available(self, channel: str, target_item: int) -> torch.Tensor:
        return torch.tensor(self._has_complete_target_sid(channel, target_item), dtype=torch.bool)

    def _load_samples(self, split: str) -> list[tuple[list[int], int]]:
        data = load_pkl(self.data_dir / f"{split}.pkl")
        if not isinstance(data, dict):
            raise TypeError(f"{split}.pkl should be a dict")
        if "item_seqs" not in data or "labels" not in data:
            raise KeyError(f"{split}.pkl must contain item_seqs and labels")

        item_seqs = data["item_seqs"]
        labels = data["labels"]
        if len(item_seqs) != len(labels):
            raise ValueError(f"{split}.pkl item_seqs/labels length mismatch: {len(item_seqs)} vs {len(labels)}")

        samples: list[tuple[list[int], int]] = []
        for idx, (item_seq, label) in enumerate(zip(item_seqs, labels)):
            if not hasattr(label, "__len__") or len(label) < 1:
                raise ValueError(f"{split}.labels[{idx}] should contain at least target item")
            target_item = int(label[0])
            self._validate_item_id(target_item, f"{split}.labels[{idx}][0]")

            seq_values = [int(item) for item in item_seq]
            for pos, item in enumerate(seq_values):
                self._validate_item_id(item, f"{split}.item_seqs[{idx}][{pos}]")
            samples.append((seq_values, target_item))

        if not samples:
            raise ValueError(f"{split} has no samples")
        return samples

    def _has_complete_target_sid(self, channel: str, item_id: int) -> bool:
        return bool(self.sid_available[channel][int(item_id)].item())

    def _validate_item_id(self, item_id: int, field: str) -> None:
        if item_id < 1 or item_id > self.n_items:
            raise ValueError(f"{field} item_id out of range: {item_id}")
