"""Read-only CPU catalog for normalized raw item embeddings."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from configs.config import DotDict, get_mod_presence_filename
from utils.data import MPSID_INPUT_MODALITIES, build_mpsid_modality_mask, embedding_filename
from utils.io import load_pkl


class RawEmbeddingCatalog:
    """Load frozen collab/text/vision raw embeddings and gather them by item id."""

    def __init__(self, data_dir: str | Path, mod_presence_filename: str = "mod_presence.pt") -> None:
        self._data_dir = Path(data_dir)
        self._num_items = self._load_num_items()
        n_total = self._num_items + 1
        mod_presence = self._load_mod_presence(mod_presence_filename, n_total)
        self._availability = build_mpsid_modality_mask(mod_presence, n_total)

        self._embeddings: dict[str, Tensor] = {}
        self._dimensions: dict[str, int] = {}
        for index, branch in enumerate(MPSID_INPUT_MODALITIES):
            embedding = self._load_embedding(branch, n_total)
            self._dimensions[branch] = int(embedding.shape[1])
            self._embeddings[branch] = self._normalize_observed(
                embedding,
                self._availability[:, index],
            )

    @classmethod
    def from_config(cls, config: DotDict) -> "RawEmbeddingCatalog":
        return cls(
            data_dir=config.data._data_dir,
            mod_presence_filename=get_mod_presence_filename(config),
        )

    @property
    def num_items(self) -> int:
        return self._num_items

    @property
    def dimensions(self) -> dict[str, int]:
        return dict(self._dimensions)

    def dimension(self, branch: str) -> int:
        self._validate_branch(branch)
        return self._dimensions[branch]

    def gather(
        self,
        branch: str,
        item_ids: Tensor | Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        """Return ``(normalized_embeddings, availability)`` for one CPU id vector."""
        self._validate_branch(branch)
        ids = torch.as_tensor(item_ids)
        if ids.device.type != "cpu":
            raise ValueError("RawEmbeddingCatalog item_ids must be on CPU")
        if ids.ndim != 1:
            raise ValueError(f"item_ids shape must be [C], got {tuple(ids.shape)}")
        invalid_dtype = ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex()
        if ids.numel() > 0 and invalid_dtype:
            raise TypeError(f"item_ids must use an integer dtype, got {ids.dtype}")
        ids = ids.long()
        if ids.numel() > 0:
            min_id = int(ids.min().item())
            max_id = int(ids.max().item())
            if min_id < 0 or max_id > self._num_items:
                raise IndexError(
                    f"item_ids must be in [0, {self._num_items}], got min={min_id}, max={max_id}"
                )

        branch_index = MPSID_INPUT_MODALITIES.index(branch)
        return (
            self._embeddings[branch].index_select(0, ids),
            self._availability[:, branch_index].index_select(0, ids),
        )

    def _load_num_items(self) -> int:
        mappings = load_pkl(self._data_dir / "mappings.pkl")
        if not isinstance(mappings, dict) or "item2id" not in mappings:
            raise KeyError("mappings.pkl must contain item2id")
        item2id = mappings["item2id"]
        if not isinstance(item2id, dict) or not item2id:
            raise ValueError("mappings['item2id'] must be a non-empty dict")
        mapped_ids = sorted(int(value) for value in item2id.values())
        if mapped_ids != list(range(1, len(mapped_ids) + 1)):
            raise ValueError("mapped_id in mappings.pkl must be contiguous 1..N")
        return len(mapped_ids)

    def _load_mod_presence(self, filename: str, n_total: int) -> Tensor:
        path = self._data_dir / filename
        presence = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(presence, Tensor):
            raise TypeError(f"{path.name} must contain a tensor")
        if presence.dtype != torch.bool:
            raise TypeError(f"{path.name} dtype must be bool, got {presence.dtype}")
        if presence.ndim != 2 or presence.shape != (n_total, 2):
            raise ValueError(f"{path.name} shape must be ({n_total}, 2), got {tuple(presence.shape)}")
        if presence[0].any().item():
            raise ValueError(f"{path.name} row0 must be all False")
        return presence

    def _load_embedding(self, branch: str, n_total: int) -> Tensor:
        path = self._data_dir / embedding_filename(branch)
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(embedding, Tensor):
            raise TypeError(f"{path.name} must contain a tensor")
        if embedding.ndim != 2:
            raise ValueError(f"{path.name} shape must be [N+1, D], got {tuple(embedding.shape)}")
        if embedding.shape[0] != n_total:
            raise ValueError(
                f"{path.name} rows {embedding.shape[0]} do not match mappings N+1={n_total}"
            )
        if embedding.shape[1] < 1:
            raise ValueError(f"{path.name} embedding dimension must be positive")
        if not embedding.is_floating_point():
            embedding = embedding.float()
        return embedding.detach()

    @staticmethod
    def _normalize_observed(embedding: Tensor, observed_mask: Tensor) -> Tensor:
        normalized = torch.zeros_like(embedding)
        if observed_mask.any().item():
            normalized[observed_mask] = F.normalize(embedding[observed_mask], dim=-1)
        return normalized

    @staticmethod
    def _validate_branch(branch: str) -> None:
        if branch not in MPSID_INPUT_MODALITIES:
            raise ValueError(
                f"unknown raw embedding branch {branch!r}; expected one of {MPSID_INPUT_MODALITIES!r}"
            )
