"""Item embedding datasets for residual quantization."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from configs.config import DotDict, get_mod_presence_filename
from utils.data import (
    MPSID_INPUT_MODALITIES,
    UNIFIED_SID_CHANNEL,
    build_mpsid_modality_mask,
    build_rqvae_valid_mask,
    embedding_filename,
)
from utils.io import load_pkl


class RQVAEDataset(Dataset):
    """Load normalized item embeddings and their original item IDs."""

    def __init__(self, config: DotDict):
        super().__init__()
        self._delegate: Dataset | None = None
        data_dir = config.data._data_dir
        modality = config.data.modality
        pre_complete = str(getattr(config.data, "pre_complete", "no")).lower()
        pre_complete_top_k = int(getattr(config.data, "pre_complete_top_k", 10))
        if pre_complete not in {"global_mean", "no", "seq_neighbor"}:
            raise ValueError(
                "data.pre_complete currently only supports "
                f"['global_mean', 'no', 'seq_neighbor'], got {pre_complete!r}"
            )
        if pre_complete_top_k <= 0:
            raise ValueError(
                f"data.pre_complete_top_k must be positive, got {pre_complete_top_k}"
            )
        self._sequence_neighbor_counts: dict[int, dict[int, int]] | None = None
        self._sequence_neighbor_n_total: int | None = None

        if modality == UNIFIED_SID_CHANNEL:
            if pre_complete != "no":
                raise ValueError(
                    f"data.pre_complete={pre_complete!r} is not implemented for unified RQ-VAE"
                )
            self._delegate = MPSIDRQVAEDataset(config)
            return


        if modality == "text":
            raw_emb = torch.load(
                os.path.join(data_dir, embedding_filename("text")),
                map_location="cpu", weights_only=True
            )  # (N+1, 768)
            mod_presence = self._load_mod_presence(data_dir, config)
            if pre_complete == "global_mean":
                raw_emb = self._complete_with_global_mean(raw_emb, mod_presence[:, 0], "text")
                valid_mask = self._all_non_padding_items(raw_emb.shape[0])
            elif pre_complete == "seq_neighbor":
                raw_emb = self._complete_with_sequence_neighbors(
                    raw_emb,
                    mod_presence[:, 0],
                    "text",
                    data_dir,
                    pre_complete_top_k,
                )
                valid_mask = self._all_non_padding_items(raw_emb.shape[0])
            else:
                valid_mask = build_rqvae_valid_mask(
                    modality, mod_presence, raw_emb.shape[0]
                )  # (N+1, )

        elif modality == "vision":
            raw_emb = torch.load(
                os.path.join(data_dir, embedding_filename("vision")),
                map_location="cpu", weights_only=True
            )  # (N+1, 768)
            mod_presence = self._load_mod_presence(data_dir, config)
            if pre_complete == "global_mean":
                raw_emb = self._complete_with_global_mean(raw_emb, mod_presence[:, 1], "vision")
                valid_mask = self._all_non_padding_items(raw_emb.shape[0])
            elif pre_complete == "seq_neighbor":
                raw_emb = self._complete_with_sequence_neighbors(
                    raw_emb,
                    mod_presence[:, 1],
                    "vision",
                    data_dir,
                    pre_complete_top_k,
                )
                valid_mask = self._all_non_padding_items(raw_emb.shape[0])
            else:
                valid_mask = build_rqvae_valid_mask(
                    modality, mod_presence, raw_emb.shape[0]
                )  # (N+1, )

        elif modality == "collab":
            raw_emb = torch.load(
                os.path.join(data_dir, embedding_filename("collab")),
                map_location="cpu", weights_only=True
            )  # (N+1, D_cf)

            valid_mask = torch.ones(raw_emb.shape[0], dtype=torch.bool)  # (N+1, )
            valid_mask[0] = False

        elif modality == "fused":
            txt_emb = torch.load(
                os.path.join(data_dir, embedding_filename("text")),
                map_location="cpu", weights_only=True
            )  # (N+1, 768)
            img_emb = torch.load(
                os.path.join(data_dir, embedding_filename("vision")),
                map_location="cpu", weights_only=True
            )  # (N+1, 768)
            mod_presence = self._load_mod_presence(data_dir, config)
            if pre_complete == "global_mean":
                txt_emb = self._complete_with_global_mean(txt_emb, mod_presence[:, 0], "text")
                img_emb = self._complete_with_global_mean(img_emb, mod_presence[:, 1], "vision")
                raw_emb = torch.add(txt_emb, img_emb)
                valid_mask = self._all_non_padding_items(raw_emb.shape[0])
            elif pre_complete == "seq_neighbor":
                txt_emb = self._complete_with_sequence_neighbors(
                    txt_emb,
                    mod_presence[:, 0],
                    "text",
                    data_dir,
                    pre_complete_top_k,
                )
                img_emb = self._complete_with_sequence_neighbors(
                    img_emb,
                    mod_presence[:, 1],
                    "vision",
                    data_dir,
                    pre_complete_top_k,
                )
                raw_emb = torch.add(txt_emb, img_emb)
                valid_mask = self._all_non_padding_items(raw_emb.shape[0])
            else:
                raw_emb = torch.add(txt_emb, img_emb)  # (N+1, 768)
                valid_mask = build_rqvae_valid_mask(
                    modality, mod_presence, raw_emb.shape[0]
                )  # (N+1, )

        else:
            raise ValueError(
                f"Unknown modality: '{modality}' (expected text / vision / collab / fused / unified)"
            )


        self.valid_mask = valid_mask
        self.valid_item_ids = valid_mask.nonzero(as_tuple=True)[0]  # (N_valid, )

        self.embeddings = raw_emb[self.valid_item_ids]  # (N_valid, D)

    def __len__(self) -> int:
        if self._delegate is not None:
            return len(self._delegate)
        return len(self.valid_item_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._delegate is not None:
            return self._delegate[idx]
        emb = self.embeddings[idx]  # (D, )

        emb = F.normalize(emb, dim=-1)
        item_id = self.valid_item_ids[idx].item()
        return {"embedding": emb, "item_id": item_id}

    def get_batch(self, indices: slice | torch.Tensor | list[int]) -> dict[str, torch.Tensor]:
        """Return an ordered unified batch without materializing per-item dictionaries."""
        if self._delegate is None or not hasattr(self._delegate, "get_batch"):
            raise RuntimeError("get_batch is only available for the unified MP-SID dataset")
        return self._delegate.get_batch(indices)

    def iter_batches(self, batch_size: int):
        """Iterate unified rows in dataset order with the same boundaries as DataLoader."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        for start in range(0, len(self), batch_size):
            yield self.get_batch(slice(start, min(start + batch_size, len(self))))

    def get_catalog_metadata(self) -> dict[str, torch.Tensor]:
        """Return unified item ids and availability without gathering embeddings."""
        if self._delegate is None or not hasattr(self._delegate, "get_catalog_metadata"):
            raise RuntimeError("catalog metadata is only available for the unified MP-SID dataset")
        return self._delegate.get_catalog_metadata()


    def _load_mod_presence(
        self, data_dir: str, config: DotDict
    ) -> torch.Tensor:
        """Load the [N+1, 2] Boolean modality-presence tensor."""

        filename = get_mod_presence_filename(config)
        path = os.path.join(data_dir, filename)
        mod_presence = torch.load(path, map_location="cpu", weights_only=True)  # (N + 1, 2) bool
        return mod_presence

    @staticmethod
    def _complete_with_global_mean(
        embedding: torch.Tensor,
        observed_mask: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError(f"{modality} embedding should be 2D, got shape {tuple(embedding.shape)}")
        if observed_mask.ndim != 1 or observed_mask.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"{modality} availability shape should be ({embedding.shape[0]},), "
                f"got {tuple(observed_mask.shape)}"
            )

        observed = observed_mask.to(dtype=torch.bool).clone()
        observed[0] = False
        if not observed.any().item():
            raise ValueError(f"cannot compute {modality} global mean without observed items")

        missing = ~observed
        missing[0] = False
        completed = embedding.clone()
        if missing.any().item():
            completed[missing] = embedding[observed].mean(dim=0)
        return completed

    def _complete_with_sequence_neighbors(
        self,
        embedding: torch.Tensor,
        observed_mask: torch.Tensor,
        modality: str,
        data_dir: str,
        top_k: int,
    ) -> torch.Tensor:
        """Fill missing rows from observed adjacent-transition neighbors in train.pkl.

        Neighbor embeddings are averaged with transition counts as weights. Items
        without an observed sequence neighbor fall back to the observed global mean.
        """
        if embedding.ndim != 2:
            raise ValueError(f"{modality} embedding should be 2D, got shape {tuple(embedding.shape)}")
        if observed_mask.ndim != 1 or observed_mask.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"{modality} availability shape should be ({embedding.shape[0]},), "
                f"got {tuple(observed_mask.shape)}"
            )

        observed = observed_mask.to(dtype=torch.bool).clone()
        observed[0] = False
        if not observed.any().item():
            raise ValueError(f"cannot complete {modality} without observed items")

        completed = embedding.clone()
        missing_item_ids = (~observed).nonzero(as_tuple=True)[0]
        missing_item_ids = missing_item_ids[missing_item_ids != 0]
        if missing_item_ids.numel() == 0:
            return completed

        neighbor_counts = self._get_sequence_neighbor_counts(
            data_dir,
            embedding.shape[0],
        )
        fallback = embedding[observed].mean(dim=0)
        for item_id_tensor in missing_item_ids:
            item_id = int(item_id_tensor.item())
            candidates = [
                (neighbor_id, count)
                for neighbor_id, count in neighbor_counts.get(item_id, {}).items()
                if observed[neighbor_id].item()
            ]
            candidates.sort(key=lambda pair: (-pair[1], pair[0]))
            selected = candidates[:top_k]
            if not selected:
                completed[item_id] = fallback
                continue

            neighbor_ids = torch.tensor(
                [neighbor_id for neighbor_id, _ in selected],
                dtype=torch.long,
                device=embedding.device,
            )
            weights = embedding.new_tensor([count for _, count in selected])
            neighbor_embeddings = embedding.index_select(0, neighbor_ids)
            completed[item_id] = (
                neighbor_embeddings * weights.unsqueeze(-1)
            ).sum(dim=0) / weights.sum()
        return completed

    def _get_sequence_neighbor_counts(
        self,
        data_dir: str,
        n_total: int,
    ) -> dict[int, dict[int, int]]:
        """Build one undirected adjacent-transition count graph from train.pkl."""
        if self._sequence_neighbor_counts is not None:
            if self._sequence_neighbor_n_total != int(n_total):
                raise ValueError("content embedding row counts should match for seq_neighbor")
            return self._sequence_neighbor_counts

        train_path = Path(data_dir) / "train.pkl"
        train_data = load_pkl(train_path)
        if not isinstance(train_data, dict):
            raise TypeError("train.pkl should be a dict")
        if "item_seqs" not in train_data or "labels" not in train_data:
            raise KeyError("train.pkl must contain item_seqs and labels")

        item_seqs = train_data["item_seqs"]
        labels = train_data["labels"]
        if len(item_seqs) != len(labels):
            raise ValueError(
                "train.pkl item_seqs/labels length mismatch: "
                f"{len(item_seqs)} vs {len(labels)}"
            )
        if len(item_seqs) == 0:
            raise ValueError("train.pkl has no samples")

        neighbor_counts: dict[int, dict[int, int]] = {}
        max_item_id = int(n_total) - 1
        for row, (item_seq, label) in enumerate(zip(item_seqs, labels)):
            if not hasattr(label, "__len__") or len(label) < 1:
                raise ValueError(f"train.labels[{row}] should contain at least target item")
            trajectory = [int(item_id) for item_id in item_seq]
            trajectory.append(int(label[0]))
            for position, item_id in enumerate(trajectory):
                if not 1 <= item_id <= max_item_id:
                    raise ValueError(
                        f"train trajectory item_id out of range at row {row}, "
                        f"position {position}: {item_id}"
                    )
            for left, right in zip(trajectory, trajectory[1:]):
                if left == right:
                    continue
                left_counts = neighbor_counts.setdefault(left, {})
                right_counts = neighbor_counts.setdefault(right, {})
                left_counts[right] = left_counts.get(right, 0) + 1
                right_counts[left] = right_counts.get(left, 0) + 1

        self._sequence_neighbor_counts = neighbor_counts
        self._sequence_neighbor_n_total = int(n_total)
        return neighbor_counts

    @staticmethod
    def _all_non_padding_items(n_total: int) -> torch.Tensor:
        valid_mask = torch.ones(int(n_total), dtype=torch.bool)
        valid_mask[0] = False
        return valid_mask


class MPSIDRQVAEDataset(Dataset):
    """Load normalized collaborative, text, and visual embeddings. Missing content blocks are zeroed according to modality presence before normalization; projection is performed by the model."""

    def __init__(self, config: DotDict):
        super().__init__()
        self.config = config
        self.data_dir = config.data._data_dir
        self.modality_order = MPSID_INPUT_MODALITIES

        collab = self._load_embedding("collab")
        text = self._load_embedding("text")
        vision = self._load_embedding("vision")
        self._validate_embedding_shapes(config, collab, text, vision)

        n_total = collab.shape[0]
        self.mod_presence = self._load_mod_presence(self.data_dir, config)
        self._validate_mod_presence(self.mod_presence, n_total)
        self.valid_mask = build_rqvae_valid_mask(UNIFIED_SID_CHANNEL, self.mod_presence, n_total)
        self.modality_mask = build_mpsid_modality_mask(self.mod_presence, n_total)
        self.valid_item_ids = self.valid_mask.nonzero(as_tuple=True)[0]

        self.collab_embeddings = self._normalize_observed(collab, self.modality_mask[:, 0])
        self.text_embeddings = self._normalize_observed(text, self.modality_mask[:, 1])
        self.vision_embeddings = self._normalize_observed(vision, self.modality_mask[:, 2])

    def __len__(self) -> int:
        return len(self.valid_item_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item_id = self.valid_item_ids[idx]
        return {
            "item_id": int(item_id.item()),
            "collab_embedding": self.collab_embeddings[item_id],
            "text_embedding": self.text_embeddings[item_id],
            "vision_embedding": self.vision_embeddings[item_id],
            "modality_mask": self.modality_mask[item_id].clone(),
        }

    def get_batch(self, indices: slice | torch.Tensor | list[int]) -> dict[str, torch.Tensor]:
        """Vectorized equivalent of default-collating ``self[indices]`` in order."""
        item_ids = self.valid_item_ids[indices]
        if item_ids.dim() == 0:
            item_ids = item_ids.unsqueeze(0)
        item_ids = item_ids.long()
        return {
            "item_id": item_ids.clone(),
            "collab_embedding": self.collab_embeddings.index_select(0, item_ids),
            "text_embedding": self.text_embeddings.index_select(0, item_ids),
            "vision_embedding": self.vision_embeddings.index_select(0, item_ids),
            "modality_mask": self.modality_mask.index_select(0, item_ids),
        }

    def iter_batches(self, batch_size: int):
        """Iterate vectorized batches in stable catalog order."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        for start in range(0, len(self), batch_size):
            yield self.get_batch(slice(start, min(start + batch_size, len(self))))

    def get_catalog_metadata(self) -> dict[str, torch.Tensor]:
        """Return ordered catalog ids and masks without materializing embedding batches."""
        item_ids = self.valid_item_ids.long()
        return {
            "item_id": item_ids.clone(),
            "modality_mask": self.modality_mask.index_select(0, item_ids),
        }

    def _load_embedding(self, modality: str) -> torch.Tensor:
        path = os.path.join(self.data_dir, embedding_filename(modality))
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"{path} must contain a tensor")
        if embedding.ndim != 2:
            raise ValueError(f"{path} shape must be [N+1, D], got {tuple(embedding.shape)}")
        if not embedding.is_floating_point():
            embedding = embedding.float()
        return embedding

    def _validate_embedding_shapes(
        self,
        config: DotDict,
        collab: torch.Tensor,
        text: torch.Tensor,
        vision: torch.Tensor,
    ) -> None:
        n_total = collab.shape[0]
        for name, tensor in (("text", text), ("vision", vision)):
            if tensor.shape[0] != n_total:
                raise ValueError(
                    f"{name} embedding rows {tensor.shape[0]} do not match collab rows {n_total}"
                )

        expected_dims = {
            "collab": int(config.model.collab_input_dim),
            "text": int(config.model.text_input_dim),
            "vision": int(config.model.visual_input_dim),
        }
        actual_dims = {
            "collab": int(collab.shape[1]),
            "text": int(text.shape[1]),
            "vision": int(vision.shape[1]),
        }
        for name, expected in expected_dims.items():
            actual = actual_dims[name]
            if actual != expected:
                raise ValueError(f"{name} embedding dim should be {expected}, got {actual}")

    def _load_mod_presence(self, data_dir: str, config: DotDict) -> torch.Tensor:
        filename = get_mod_presence_filename(config)
        path = os.path.join(data_dir, filename)
        mod_presence = torch.load(path, map_location="cpu", weights_only=True)
        return mod_presence

    @staticmethod
    def _validate_mod_presence(mod_presence: torch.Tensor, n_total: int) -> None:
        if not isinstance(mod_presence, torch.Tensor):
            raise TypeError("mod_presence must be a tensor")
        if mod_presence.dtype != torch.bool:
            raise TypeError(f"mod_presence dtype must be bool, got {mod_presence.dtype}")
        if mod_presence.ndim != 2 or mod_presence.shape != (n_total, 2):
            raise ValueError(f"mod_presence shape must be ({n_total}, 2), got {tuple(mod_presence.shape)}")
        if mod_presence[0].any().item():
            raise ValueError("mod_presence row0 must be all False")

    @staticmethod
    def _normalize_observed(embedding: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(embedding)
        observed = observed_mask.bool()
        if observed.any().item():
            normalized[observed] = F.normalize(embedding[observed], dim=-1)
        return normalized
