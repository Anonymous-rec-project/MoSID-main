"""SID code embedding utilities for MoSID decoder-only generation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor


CodebookWeights = Mapping[str, Sequence[Tensor]] | None


class SIDCodeEmbedding(nn.Module):
    """Layer-specific SID embeddings initialized from Stage 1 codebooks.

    Stage 1 artifacts store valid SID codes in artifact domain ``1..V`` and
    keep ``0`` for padding/missing rows. Decoder generation operates in output
    domain ``0..V-1``; this module embeds those output-domain codes.
    """

    def __init__(
        self,
        channels: Sequence[str],
        sid_depth: int,
        codebook_size: int,
        sid_embed_dim: int,
        codebook_weights: CodebookWeights = None,
    ):
        super().__init__()
        self.channels = tuple(str(channel) for channel in channels)
        self.sid_depth = int(sid_depth)
        self.codebook_size = int(codebook_size)
        self.sid_embed_dim = int(sid_embed_dim)

        if not self.channels:
            raise ValueError("channels must be non-empty")
        if self.sid_depth <= 0:
            raise ValueError(f"sid_depth must be positive, got {sid_depth}")
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size}")
        if self.sid_embed_dim <= 0:
            raise ValueError(f"sid_embed_dim must be positive, got {sid_embed_dim}")

        self.embeddings = nn.ModuleDict()
        for channel in self.channels:
            layer_weights = None if codebook_weights is None else codebook_weights.get(channel)
            self.embeddings[channel] = self._build_channel_embeddings(channel, layer_weights)

    def forward(self, channel: str, layer: int, output_codes: Tensor, ignore_index: int = -100) -> Tensor:
        return self.embed_output_codes(channel, layer, output_codes, ignore_index=ignore_index)

    def embed_output_codes(
        self,
        channel: str,
        layer: int,
        output_codes: Tensor,
        ignore_index: int = -100,
        *,
        codes_validated: bool = False,
    ) -> Tensor:
        """Embed codes, skipping range checks only after validated SID conversion."""
        embedding = self._layer_embedding(channel, layer)
        codes = output_codes.long()
        valid = codes != int(ignore_index)
        if not codes_validated and (valid & ((codes < 0) | (codes >= self.codebook_size))).any().item():
            valid_codes = codes[valid]
            min_code = int(valid_codes.min().item())
            max_code = int(valid_codes.max().item())
            raise ValueError(
                f"{channel} layer {layer} output codes must be in [0, {self.codebook_size - 1}], "
                f"got min={min_code}, max={max_code}"
            )

        safe_codes = codes.clamp_min(0)
        embedded = embedding(safe_codes)
        return embedded.masked_fill(~valid.unsqueeze(-1), 0.0)

    def layer_weight(self, channel: str, layer: int) -> Tensor:
        return self._layer_embedding(channel, layer).weight

    def _build_channel_embeddings(
        self,
        channel: str,
        layer_weights: Sequence[Tensor] | None,
    ) -> nn.ModuleList:
        if layer_weights is not None and len(layer_weights) != self.sid_depth:
            raise ValueError(
                f"{channel} codebook_weights should contain {self.sid_depth} layers, got {len(layer_weights)}"
            )

        layers = nn.ModuleList()
        for layer in range(self.sid_depth):
            weight = None if layer_weights is None else layer_weights[layer]
            if weight is None:
                embedding = nn.Embedding(self.codebook_size, self.sid_embed_dim)
                nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
                embedding.weight.requires_grad_(False)
            else:
                weight = self._validate_codebook_weight(channel, layer, weight)
                embedding = nn.Embedding.from_pretrained(weight, freeze=True)
            layers.append(embedding)
        return layers

    def _validate_codebook_weight(self, channel: str, layer: int, weight: Tensor) -> Tensor:
        if not isinstance(weight, Tensor):
            raise TypeError(f"{channel} codebook layer {layer} must be a tensor")
        expected = (self.codebook_size, self.sid_embed_dim)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"{channel} codebook layer {layer} shape should be {expected}, got {tuple(weight.shape)}"
            )
        if not torch.isfinite(weight.float()).all().item():
            raise ValueError(f"{channel} codebook layer {layer} contains NaN/Inf")
        return weight.detach().float().clone()

    def _layer_embedding(self, channel: str, layer: int) -> nn.Embedding:
        if channel not in self.embeddings:
            raise ValueError(f"unknown SID channel {channel!r}; expected one of {self.channels!r}")
        layer = int(layer)
        if layer < 0 or layer >= self.sid_depth:
            raise ValueError(f"layer must be in [0, {self.sid_depth - 1}], got {layer}")
        return self.embeddings[channel][layer]


def artifact_sid_to_output_codes(
    artifact_sid: Tensor,
    codebook_size: int,
    ignore_index: int = -100,
) -> Tensor:
    """Convert artifact-domain SID codes ``1..V`` to output-domain ``0..V-1``."""
    sid = artifact_sid.long()
    valid = sid > 0
    # One boundary check for the complete SID tensor; per-layer embedding can
    # reuse this guarantee. Non-positive codes retain the ignored-code mapping.
    if (valid & (sid > int(codebook_size))).any().item():
        valid_sid = sid[valid]
        min_code = int(valid_sid.min().item())
        max_code = int(valid_sid.max().item())
        raise ValueError(
            f"artifact SID codes must be in [1, {int(codebook_size)}], "
            f"got min={min_code}, max={max_code}"
        )
    return torch.where(valid, sid - 1, int(ignore_index))


def load_codebook_weights(path: str | bytes | "os.PathLike[str]") -> list[Tensor]:
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(weights, list) or not weights:
        raise TypeError(f"{path} must contain a non-empty list of codebook tensors")
    return weights
