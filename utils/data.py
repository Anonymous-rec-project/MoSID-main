"""Modality availability, artifact names, and valid-item masks."""
from __future__ import annotations

import os

import torch

from configs.config import get_mod_presence_filename, DotDict


UNIFIED_SID_CHANNEL = "unified"
MPSID_INPUT_MODALITIES = ("collab", "text", "vision")
MOSID_SEPARATE_SID_ARTIFACT_MODE = "separate"
MOSID_UNIFIED_SID_ARTIFACT_MODE = "unified"
MPSID_BLOCK_INDEX_BY_CHANNEL = {
    modality: index
    for index, modality in enumerate(MPSID_INPUT_MODALITIES)
}


def load_mod_presence(data_dir: str, config: DotDict) -> torch.Tensor:
    """Load the [N+1, 2] Boolean modality-presence tensor."""
    filename = get_mod_presence_filename(config)  # str
    path = os.path.join(data_dir, filename)       # str
    return torch.load(path, map_location="cpu", weights_only=True)


def embedding_filename(modality: str) -> str:
    base_names = {
        "text": "txt_emb",
        "vision": "img_emb",
        "collab": "collab_emb",
    }
    if modality not in base_names:
        raise ValueError(f"Unknown modality: '{modality}'")
    return f"{base_names[modality]}.pt"


def sid_codes_filename(branch: str) -> str:
    return f"sid_codes_{branch}.pt"


def codebook_weights_filename(branch: str) -> str:
    return f"codebook_weights_{branch}.pt"


def mosid_sid_artifact_mode(config: DotDict) -> str:
    model = getattr(config, "model", None)
    mode = getattr(model, "sid_artifact_mode", MOSID_SEPARATE_SID_ARTIFACT_MODE) if model is not None else None
    return str(mode or MOSID_SEPARATE_SID_ARTIFACT_MODE).lower()


def mosid_uses_unified_block_views(config: DotDict) -> bool:
    return mosid_sid_artifact_mode(config) == MOSID_UNIFIED_SID_ARTIFACT_MODE


def mosid_sid_artifact_channel(config: DotDict, branch: str) -> str:
    if mosid_uses_unified_block_views(config):
        return UNIFIED_SID_CHANNEL
    return str(branch)


def build_valid_mask(
    modality: str,
    mod_presence: torch.Tensor | None,
    n_total: int,
) -> torch.Tensor:
    """Build a Boolean item-validity mask; padding index zero is always excluded."""

    if modality == "text":
        valid_mask = mod_presence[:, 0].clone()             # (N+1,) bool
    elif modality == "vision":
        valid_mask = mod_presence[:, 1].clone()             # (N+1,) bool
    elif modality in {"collab", UNIFIED_SID_CHANNEL}:
        valid_mask = torch.ones(n_total, dtype=torch.bool)  # (N+1,) bool
    elif modality == "fused":
        valid_mask = mod_presence[:, 0] | mod_presence[:, 1]  # (N+1,) bool
    else:
        raise ValueError(f"Unknown modality: '{modality}'")

    valid_mask[0] = False
    return valid_mask


def build_mpsid_modality_mask(
    mod_presence: torch.Tensor,
    n_total: int,
) -> torch.Tensor:
    """Build availability masks in collab/text/vision order; padding row zero is excluded."""
    if mod_presence is None:
        raise ValueError("MP-SID requires mod_presence for text/vision availability")
    if mod_presence.ndim != 2 or mod_presence.shape[1] != 2:
        raise ValueError(f"mod_presence shape must be [N+1, 2], got {tuple(mod_presence.shape)}")
    if mod_presence.shape[0] != n_total:
        raise ValueError(f"mod_presence rows {mod_presence.shape[0]} do not match n_total={n_total}")

    mask = torch.zeros(n_total, len(MPSID_INPUT_MODALITIES), dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1] = mod_presence[:, 0].bool()
    mask[:, 2] = mod_presence[:, 1].bool()
    mask[0] = False
    return mask


def build_rqvae_valid_mask(
    modality: str,
    mod_presence: torch.Tensor | None,
    n_total: int,
) -> torch.Tensor:
    return build_valid_mask(modality, mod_presence, n_total)
