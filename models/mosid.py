"""MoSID Stage 2 P2/P8 encoder-decoder backbone."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from configs.config import DotDict, raw_compat_is_enabled
from models.history_encoder import (
    CausalSIDHistoryEncoder,
    HistoryEncoderOutput,
    SIDHistoryMemoryEncoder,
)
from models.sid_embedding import CodebookWeights, SIDCodeEmbedding, load_codebook_weights
from models.raw_compatibility import RawCompatibilityScorer
from models.sid_t5_decoder import SIDCrossAttentionDecoder
from utils.data import (
    MPSID_BLOCK_INDEX_BY_CHANNEL,
    MPSID_INPUT_MODALITIES,
    UNIFIED_SID_CHANNEL,
    codebook_weights_filename,
    mosid_uses_unified_block_views,
)


@dataclass
class MoSIDHistoryContext:
    token_hidden_states: dict[str, Tensor]
    token_attention_mask: dict[str, Tensor]
    attention_mask: Tensor
    memory: dict[str, Tensor]
    memory_key_padding_mask: dict[str, Tensor]

    def index_select(self, dim: int, dim0_indices: Tensor) -> "MoSIDHistoryContext":
        if dim != 0:
            raise ValueError(f"MoSIDHistoryContext only supports dim=0 index_select, got dim={dim}")
        return MoSIDHistoryContext(
            token_hidden_states={
                channel: value.index_select(0, dim0_indices)
                for channel, value in self.token_hidden_states.items()
            },
            token_attention_mask={
                channel: value.index_select(0, dim0_indices)
                for channel, value in self.token_attention_mask.items()
            },
            attention_mask=self.attention_mask.index_select(0, dim0_indices),
            memory={
                channel: value.index_select(0, dim0_indices)
                for channel, value in self.memory.items()
            },
            memory_key_padding_mask={
                channel: value.index_select(0, dim0_indices)
                for channel, value in self.memory_key_padding_mask.items()
            },
        )

    def index_select_branch(self, dim: int, dim0_indices: Tensor, channel: str) -> "MoSIDHistoryContext":
        if dim != 0:
            raise ValueError(f"MoSIDHistoryContext only supports dim=0 index_select, got dim={dim}")
        channel = str(channel)
        return MoSIDHistoryContext(
            token_hidden_states=self._index_select_channel_map(self.token_hidden_states, dim0_indices, channel),
            token_attention_mask=self._index_select_channel_map(self.token_attention_mask, dim0_indices, channel),
            attention_mask=self.attention_mask.index_select(0, dim0_indices),
            memory=self._index_select_channel_map(self.memory, dim0_indices, channel),
            memory_key_padding_mask=self._index_select_channel_map(
                self.memory_key_padding_mask,
                dim0_indices,
                channel,
            ),
        )

    @staticmethod
    def _index_select_channel_map(
        values: dict[str, Tensor],
        dim0_indices: Tensor,
        channel: str,
    ) -> dict[str, Tensor]:
        if channel not in values:
            return {}
        return {channel: values[channel].index_select(0, dim0_indices)}


@dataclass
class MoSIDOutput:
    logits: dict[str, Tensor]
    hidden_states: dict[str, Tensor]
    target_sid: dict[str, Tensor]
    target_available: dict[str, Tensor]
    target_item: Tensor | None
    context: MoSIDHistoryContext
    loss_ignore_index: int = -100


class MoSIDModel(nn.Module):
    """Observed-only multi-channel SID generation backbone used by P2/P8."""

    def __init__(
        self,
        max_len: int,
        channels: Sequence[str],
        active_channels: Sequence[str],
        sid_depth: int | None = None,
        codebook_size: int | None = None,
        sid_embed_dim: int = 64,
        hidden_dim: int = 64,
        n_heads: int = 4,
        history_layers: int = 2,
        decoder_layers: int = 1,
        ff_dim: int = 256,
        dropout: float = 0.1,
        encoder_mode: str = "causal",
        codebook_weights: CodebookWeights = None,
        raw_dimensions: Mapping[str, int] | None = None,
        ff_activation: str = "relu",
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.channels = tuple(str(channel) for channel in channels)
        self.active_channels = tuple(str(channel) for channel in active_channels)
        self.sid_depth, self.codebook_size = self._resolve_sid_shape(
            sid_depth,
            codebook_size,
            codebook_weights,
        )
        self.sid_embed_dim = int(sid_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.history_layers = int(history_layers)
        self.decoder_layers = int(decoder_layers)
        self.ff_dim = int(ff_dim)
        self.ff_activation = str(ff_activation).lower()
        self.dropout = float(dropout)
        self.encoder_mode = str(encoder_mode).lower()
        self.loss_ignore_index = -100
        self._validate_init()

        self.t5_decoder = self._build_t5_decoder(codebook_weights)
        self.history_encoders = nn.ModuleDict(
            {
                channel: self._build_history_encoder(
                    channel=channel,
                    channels=(channel,),
                )
                for channel in self.active_channels
            }
        )
        if raw_dimensions is not None:
            # Keep data-order/dropout RNG streams comparable with raw-disabled runs.
            with torch.random.fork_rng(devices=[]):
                self.raw_compatibility_scorer = RawCompatibilityScorer(
                    raw_dimensions=raw_dimensions,
                    hidden_dim=self.hidden_dim,
                )

    @classmethod
    def from_config(
        cls,
        config: DotDict,
        codebook_weights: CodebookWeights = None,
        raw_dimensions: Mapping[str, int] | None = None,
        enable_raw_compatibility: bool | None = None,
    ) -> "MoSIDModel":
        raw_enabled = (
            raw_compat_is_enabled(config)
            if enable_raw_compatibility is None
            else bool(enable_raw_compatibility)
        )
        if raw_enabled and raw_dimensions is None:
            raise ValueError("raw compatibility is enabled but raw_dimensions were not provided")
        if not raw_enabled and raw_dimensions is not None:
            raise ValueError("raw_dimensions were provided while raw compatibility is disabled")
        weights = codebook_weights or cls._load_active_codebooks(config)
        return cls(
            max_len=int(config.data.max_len),
            channels=tuple(config.model.channels),
            active_channels=tuple(config.model.active_channels),
            sid_embed_dim=int(config.model.sid_embed_dim),
            hidden_dim=int(config.model.hidden_dim),
            n_heads=int(config.model.n_heads),
            history_layers=int(config.model.history_layers),
            decoder_layers=int(config.model.decoder_layers),
            ff_dim=int(config.model.ff_dim),
            ff_activation=str(getattr(config.model, "ff_activation", "relu")),
            dropout=float(config.model.dropout),
            encoder_mode=str(getattr(config.model, "encoder_mode", "causal")),
            codebook_weights=weights,
            raw_dimensions=raw_dimensions,
        )

    def forward(self, batch: Mapping[str, object]) -> MoSIDOutput:
        context = self.encode_batch_history(batch)
        device = next(self.parameters()).device
        logits: dict[str, Tensor] = {}
        hidden_states: dict[str, Tensor] = {}
        target_sid: dict[str, Tensor] = {}
        target_available: dict[str, Tensor] = {}

        for channel in self.active_channels:
            target_artifact_sid = self._target_sid(batch, channel).to(device)
            decoder_output = self._decode_target(context, target_artifact_sid, channel)
            logits[channel] = decoder_output.logits
            hidden_states[channel] = decoder_output.hidden_states
            target_sid[channel] = decoder_output.target_sid
            target_available[channel] = self._target_available(
                batch,
                channel,
                device=device,
                batch_size=decoder_output.target_sid.shape[0],
            ).to(decoder_output.target_sid.device)

        return MoSIDOutput(
            logits=logits,
            hidden_states=hidden_states,
            target_sid=target_sid,
            target_available=target_available,
            target_item=self._optional_tensor(batch, "target_item"),
            context=context,
            loss_ignore_index=self.loss_ignore_index,
        )

    def encode_batch_history(self, batch: Mapping[str, object]) -> MoSIDHistoryContext:
        device = next(self.parameters()).device
        attention_mask = self._attention_mask(batch, device)
        token_hidden_states: dict[str, Tensor] = {}
        token_attention_mask: dict[str, Tensor] = {}
        memory: dict[str, Tensor] = {}
        memory_key_padding_mask: dict[str, Tensor] = {}

        for channel in self.active_channels:
            channel_attention_mask = self._history_attention_mask(batch, channel, attention_mask, device)
            history_output = self._encode_history_channel(
                batch,
                channel,
                channel_attention_mask,
                device=device,
            )
            token_hidden_states[channel] = history_output.token_hidden_states
            token_attention_mask[channel] = history_output.token_attention_mask
            memory[channel], memory_key_padding_mask[channel] = self._history_memory(history_output)

        return MoSIDHistoryContext(
            token_hidden_states=token_hidden_states,
            token_attention_mask=token_attention_mask,
            attention_mask=attention_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def raw_history_summaries(
        self,
        context: MoSIDHistoryContext,
        decoder_hidden_states: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Summarize each encoded branch memory for raw compatibility scoring."""
        if not hasattr(self, "raw_compatibility_scorer"):
            raise ValueError("raw history summaries require an initialized raw compatibility scorer")
        if not isinstance(context, MoSIDHistoryContext):
            raise TypeError(f"context should be MoSIDHistoryContext, got {type(context).__name__}")

        if decoder_hidden_states is not None:
            return self._teacher_forced_raw_summaries(context, decoder_hidden_states)

        summaries: dict[str, Tensor] = {}
        for channel in self.active_channels:
            memory, memory_key_padding_mask = self._branch_memory(context, channel)
            summaries[channel] = self.t5_decoder.history_summary(
                memory,
                memory_key_padding_mask,
                channel=channel,
            )
        return summaries

    def _teacher_forced_raw_summaries(
        self,
        context: MoSIDHistoryContext,
        decoder_hidden_states: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        summaries: dict[str, Tensor] = {}
        for channel in self.active_channels:
            if channel not in decoder_hidden_states:
                raise KeyError(f"decoder hidden states are missing branch {channel!r}")
            hidden = decoder_hidden_states[channel]
            memory, _ = self._branch_memory(context, channel)
            if hidden.dim() != 3 or hidden.shape[0] != memory.shape[0] or hidden.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"decoder hidden states for {channel!r} must have shape "
                    f"[B, K, {self.hidden_dim}], got {tuple(hidden.shape)}"
                )
            if hidden.shape[1] < 1:
                raise ValueError(f"decoder hidden states for {channel!r} must contain BOS")
            summaries[channel] = hidden[:, 0, :]
        return summaries

    def _encode_history_channel(
        self,
        batch: Mapping[str, object],
        channel: str,
        attention_mask: Tensor,
        device: torch.device,
    ) -> HistoryEncoderOutput:
        history_sid = self._history_sid(batch, channel).to(device)
        available = self._history_available(batch, channel, attention_mask).to(device)
        encoder = self.history_encoders[channel]
        return encoder(
            history_sid,
            attention_mask,
            channel=channel,
            available_mask=available,
        )

    @staticmethod
    def _history_memory(history_output: HistoryEncoderOutput) -> tuple[Tensor, Tensor]:
        token_hidden = history_output.token_hidden_states
        token_mask = history_output.token_attention_mask
        batch_size, seq_len, sid_depth, hidden_dim = token_hidden.shape
        memory = token_hidden.reshape(batch_size, seq_len * sid_depth, hidden_dim)
        memory_key_padding_mask = ~token_mask.reshape(batch_size, seq_len * sid_depth).bool()
        return memory, memory_key_padding_mask

    def logits_for_prefix_context(
        self,
        context: MoSIDHistoryContext,
        prefix_sid: Tensor,
        channel: str = "collab",
        branch: str | None = None,
    ) -> Tensor:
        if branch is not None:
            channel = branch
        elif channel == "collab" and "collab" not in self.active_channels and len(self.active_channels) == 1:
            channel = self.active_channels[0]
        if not isinstance(context, MoSIDHistoryContext):
            raise TypeError(f"context should be MoSIDHistoryContext, got {type(context).__name__}")
        if channel not in self.active_channels:
            raise ValueError(f"channel {channel!r} is not active in MoSID backbone: {self.active_channels!r}")
        memory, memory_key_padding_mask = self._branch_memory(context, channel)
        return self.t5_decoder.logits_for_prefix_context(
            memory,
            memory_key_padding_mask,
            prefix_sid,
            channel=channel,
        )

    def logits_for_indexed_prefix_context(
        self,
        context: MoSIDHistoryContext,
        dim0_indices: Tensor,
        prefix_sid: Tensor,
        channel: str = "collab",
        branch: str | None = None,
    ) -> Tensor:
        """Decode indexed beams while selecting only tensors consumed by the branch."""
        if branch is not None:
            channel = branch
        elif channel == "collab" and "collab" not in self.active_channels and len(self.active_channels) == 1:
            channel = self.active_channels[0]
        if not isinstance(context, MoSIDHistoryContext):
            raise TypeError(f"context should be MoSIDHistoryContext, got {type(context).__name__}")
        if channel not in self.active_channels:
            raise ValueError(f"channel {channel!r} is not active in MoSID backbone: {self.active_channels!r}")

        memory, memory_key_padding_mask = self._branch_memory(context, channel)
        return self.t5_decoder.logits_for_prefix_context(
            memory.index_select(0, dim0_indices),
            memory_key_padding_mask.index_select(0, dim0_indices),
            prefix_sid,
            channel=channel,
        )

    def _validate_init(self) -> None:
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channels must be a non-empty unique sequence, got {self.channels!r}")
        if not self.active_channels:
            raise ValueError("active_channels must be non-empty")
        if len(set(self.active_channels)) != len(self.active_channels):
            raise ValueError(f"active_channels must be unique, got {self.active_channels!r}")
        unknown = [channel for channel in self.active_channels if channel not in self.channels]
        if unknown:
            raise ValueError(f"active_channels contains unknown channels: {unknown!r}")
        for name, value in (
            ("max_len", self.max_len),
            ("sid_depth", self.sid_depth),
            ("codebook_size", self.codebook_size),
            ("sid_embed_dim", self.sid_embed_dim),
            ("hidden_dim", self.hidden_dim),
            ("n_heads", self.n_heads),
            ("history_layers", self.history_layers),
            ("decoder_layers", self.decoder_layers),
            ("ff_dim", self.ff_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by n_heads, got {self.hidden_dim} and {self.n_heads}")
        if self.ff_activation not in {"gelu", "leaky_relu", "relu", "swiglu"}:
            raise ValueError(
                "ff_activation must be 'relu', 'gelu', 'leaky_relu', or 'swiglu', "
                f"got {self.ff_activation!r}"
            )
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.encoder_mode not in {"causal", "bidirection"}:
            raise ValueError(
                "encoder_mode must be 'causal' or 'bidirection', "
                f"got {self.encoder_mode!r}"
            )
    def _build_t5_decoder(
        self,
        codebook_weights: CodebookWeights,
    ) -> SIDCrossAttentionDecoder:
        return SIDCrossAttentionDecoder(
            channels=self.active_channels,
            sid_depth=self.sid_depth,
            codebook_size=self.codebook_size,
            sid_embed_dim=self.sid_embed_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            decoder_layers=self.decoder_layers,
            ff_dim=self.ff_dim,
            ff_activation=self.ff_activation,
            dropout=self.dropout,
            codebook_weights=codebook_weights,
            loss_ignore_index=self.loss_ignore_index,
        )

    def _decode_target(
        self,
        context: MoSIDHistoryContext,
        target_artifact_sid: Tensor,
        channel: str,
    ):
        memory, memory_key_padding_mask = self._branch_memory(context, channel)
        return self.t5_decoder(
            memory,
            memory_key_padding_mask,
            target_artifact_sid,
            channel=channel,
        )

    def _branch_memory(self, context: MoSIDHistoryContext, channel: str) -> tuple[Tensor, Tensor]:
        if channel not in context.memory:
            raise ValueError(f"context memory for channel {channel!r} is required for decoder cross-attention")
        if channel not in context.memory_key_padding_mask:
            raise ValueError(
                f"context memory_key_padding_mask for channel {channel!r} is required for decoder cross-attention"
            )
        return context.memory[channel], context.memory_key_padding_mask[channel]

    def _sid_embedding_for(self, channel: str):
        return self.t5_decoder.sid_embedding

    def _unique_sid_embeddings(self):
        seen: set[int] = set()
        for module in self.modules():
            if isinstance(module, SIDCodeEmbedding):
                identifier = id(module)
                if identifier in seen:
                    continue
                seen.add(identifier)
                yield module

    def _build_history_encoder(
        self,
        channel: str,
        channels: Sequence[str],
    ) -> CausalSIDHistoryEncoder:
        encoder_cls = SIDHistoryMemoryEncoder if self.encoder_mode == "bidirection" else CausalSIDHistoryEncoder
        return encoder_cls(
            channel=channel,
            channels=channels,
            max_len=self.max_len,
            sid_depth=self.sid_depth,
            codebook_size=self.codebook_size,
            sid_embed_dim=self.sid_embed_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            num_layers=self.history_layers,
            ff_dim=self.ff_dim,
            ff_activation=self.ff_activation,
            dropout=self.dropout,
            sid_embedding=self._sid_embedding_for(channel),
        )

    @staticmethod
    def _resolve_sid_shape(
        sid_depth: int | None,
        codebook_size: int | None,
        codebook_weights: CodebookWeights,
    ) -> tuple[int, int]:
        if sid_depth is not None and codebook_size is not None:
            return int(sid_depth), int(codebook_size)
        if not codebook_weights:
            raise ValueError("sid_depth/codebook_size must be provided when codebook_weights are unavailable")
        first_channel, first_weights = next(iter(codebook_weights.items()))
        inferred_depth = len(first_weights)
        if inferred_depth <= 0:
            raise ValueError(f"{first_channel} codebook_weights must be non-empty")
        inferred_size = int(first_weights[0].shape[0])
        return int(sid_depth or inferred_depth), int(codebook_size or inferred_size)

    @staticmethod
    def _load_active_codebooks(config: DotDict) -> dict[str, list[Tensor]]:
        data_dir = Path(config.data._data_dir)
        if mosid_uses_unified_block_views(config):
            return MoSIDModel._load_unified_block_view_codebooks(config, data_dir)
        weights: dict[str, list[Tensor]] = {}
        for channel in config.model.active_channels:
            path = data_dir / codebook_weights_filename(channel)
            weights[channel] = load_codebook_weights(path)
        return weights

    @staticmethod
    def _load_unified_block_view_codebooks(config: DotDict, data_dir: Path) -> dict[str, list[Tensor]]:
        sid_embed_dim = int(config.model.sid_embed_dim)
        if sid_embed_dim <= 0:
            raise ValueError(f"model.sid_embed_dim must be positive, got {sid_embed_dim}")
        path = data_dir / codebook_weights_filename(UNIFIED_SID_CHANNEL)
        unified_weights = load_codebook_weights(path)
        expected_dim = len(MPSID_INPUT_MODALITIES) * sid_embed_dim
        weights: dict[str, list[Tensor]] = {}
        for channel in config.model.active_channels:
            channel = str(channel)
            block_index = MPSID_BLOCK_INDEX_BY_CHANNEL.get(channel)
            if block_index is None:
                raise ValueError(
                    "model.sid_artifact_mode='unified' requires active channels "
                    f"from {tuple(MPSID_BLOCK_INDEX_BY_CHANNEL)!r}, got {channel!r}"
                )
            start = block_index * sid_embed_dim
            end = start + sid_embed_dim
            channel_weights: list[Tensor] = []
            for layer_idx, layer_weight in enumerate(unified_weights):
                if layer_weight.dim() != 2 or int(layer_weight.shape[1]) != expected_dim:
                    raise ValueError(
                        f"{path.name} layer {layer_idx} shape must be [V, {expected_dim}] "
                        "for unified block views "
                        f"(sid_embed_dim={sid_embed_dim}, modalities={MPSID_INPUT_MODALITIES!r}), "
                        f"got {tuple(layer_weight.shape)}"
                    )
                channel_weights.append(layer_weight[:, start:end].detach().clone())
            weights[channel] = channel_weights
        return weights

    @staticmethod
    def _nested_tensor(batch: Mapping[str, object], outer: str, inner: str) -> Tensor:
        value = batch.get(outer)
        if not isinstance(value, Mapping) or inner not in value:
            raise KeyError(f"batch must contain {outer}[{inner!r}]")
        tensor = value[inner]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"batch {outer}[{inner!r}] must be a tensor")
        return tensor

    def _history_sid(self, batch: Mapping[str, object], channel: str) -> Tensor:
        sid = self._nested_tensor(batch, "history_sid", channel).long()
        if sid.dim() != 3 or sid.shape[1] != self.max_len or sid.shape[2] != self.sid_depth:
            raise ValueError(
                f"history_sid[{channel!r}] shape should be (B, {self.max_len}, {self.sid_depth}), "
                f"got {tuple(sid.shape)}"
            )
        return sid

    def _target_sid(self, batch: Mapping[str, object], channel: str) -> Tensor:
        sid = self._nested_tensor(batch, "target_sid", channel).long()
        if sid.dim() != 2 or sid.shape[1] != self.sid_depth:
            raise ValueError(
                f"target_sid[{channel!r}] shape should be (B, {self.sid_depth}), got {tuple(sid.shape)}"
            )
        return sid

    def _target_available(
        self,
        batch: Mapping[str, object],
        channel: str,
        device: torch.device,
        batch_size: int,
    ) -> Tensor:
        value = batch.get("target_available")
        if isinstance(value, Mapping) and channel in value:
            available = value[channel]
            if not isinstance(available, Tensor):
                raise TypeError(f"batch target_available[{channel!r}] must be a tensor")
            if tuple(available.shape) != (batch_size,):
                raise ValueError(
                    f"target_available[{channel!r}] shape should be ({batch_size},), "
                    f"got {tuple(available.shape)}"
                )
            return available.bool().to(device)
        return torch.ones(batch_size, dtype=torch.bool, device=device)

    def _history_available(self, batch: Mapping[str, object], channel: str, attention_mask: Tensor) -> Tensor:
        value = batch.get("history_available")
        if isinstance(value, Mapping) and channel in value:
            available = value[channel]
            if not isinstance(available, Tensor):
                raise TypeError(f"batch history_available[{channel!r}] must be a tensor")
            return available.bool()
        if channel == "collab":
            return attention_mask.detach().cpu() if attention_mask.is_cuda else attention_mask
        sid = self._history_sid(batch, channel)
        return sid.gt(0).all(dim=-1)

    def _attention_mask(self, batch: Mapping[str, object], device: torch.device) -> Tensor:
        value = batch.get("attention_mask")
        if not isinstance(value, Tensor):
            raise KeyError("batch must contain tensor attention_mask")
        mask = value.bool().to(device)
        if mask.dim() != 2 or mask.shape[1] != self.max_len:
            raise ValueError(f"attention_mask shape should be (B, {self.max_len}), got {tuple(mask.shape)}")
        return mask

    def _history_attention_mask(
        self,
        batch: Mapping[str, object],
        channel: str,
        fallback: Tensor,
        device: torch.device,
    ) -> Tensor:
        value = batch.get("history_attention_mask")
        if isinstance(value, Mapping) and channel in value:
            mask = value[channel]
            if not isinstance(mask, Tensor):
                raise TypeError(f"batch history_attention_mask[{channel!r}] must be a tensor")
            mask = mask.bool().to(device)
            if mask.dim() != 2 or mask.shape[1] != self.max_len:
                raise ValueError(
                    f"history_attention_mask[{channel!r}] shape should be (B, {self.max_len}), "
                    f"got {tuple(mask.shape)}"
                )
            return mask
        return fallback

    @staticmethod
    def _optional_tensor(batch: Mapping[str, object], key: str) -> Tensor | None:
        value = batch.get(key)
        if value is None:
            return None
        if not isinstance(value, Tensor):
            raise TypeError(f"batch {key!r} must be a tensor when provided")
        return value
