"""Causal SID history encoder for the MoSID Stage 2 backbone."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from models.feedforward import resolve_ff_activation
from models.sid_embedding import CodebookWeights, SIDCodeEmbedding, artifact_sid_to_output_codes


@dataclass
class HistoryEncoderOutput:
    token_hidden_states: Tensor
    token_attention_mask: Tensor
    attention_mask: Tensor


@dataclass
class HistoryTokenEmbeddingOutput:
    token_embeddings: Tensor
    token_attention_mask: Tensor
    attention_mask: Tensor
    available_mask: Tensor


class CausalSIDHistoryEncoder(nn.Module):
    """Encode one channel's historical SID tokens with causal self-attention."""

    def __init__(
        self,
        channel: str,
        max_len: int,
        sid_depth: int,
        codebook_size: int,
        sid_embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
        sid_embedding: SIDCodeEmbedding | None = None,
        codebook_weights: CodebookWeights = None,
        channels: Sequence[str] | None = None,
        ff_activation: str = "relu",
    ):
        super().__init__()
        if channels is None:
            channels = (str(channel),)
        self.channels = tuple(str(name) for name in channels)
        if not self.channels:
            raise ValueError("channels must be non-empty")
        self.channel = str(channel or self.channels[0])
        if self.channel not in self.channels:
            raise ValueError(f"default channel {self.channel!r} not in channels {self.channels!r}")
        self.max_len = int(max_len)
        self.sid_depth = int(sid_depth)
        self.codebook_size = int(codebook_size)
        self.sid_embed_dim = int(sid_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.num_layers = int(num_layers)
        self.ff_dim = int(ff_dim)
        self.ff_activation = str(ff_activation).lower()
        self.dropout_prob = float(dropout)
        self.loss_ignore_index = -100
        self._validate_init()

        self.sid_embedding = sid_embedding or SIDCodeEmbedding(
            channels=self.channels,
            sid_depth=self.sid_depth,
            codebook_size=self.codebook_size,
            sid_embed_dim=self.sid_embed_dim,
            codebook_weights=codebook_weights,
        )
        self.sid_input_proj = self._linear_or_identity(self.sid_embed_dim, self.hidden_dim)
        self.miss_embeddings = nn.ParameterDict(
            {
                channel_name: nn.Parameter(torch.empty(self.sid_embed_dim))
                for channel_name in self.channels
                if channel_name in {"text", "vision"}
            }
        )
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.absolute_slot_bias = nn.Parameter(torch.empty(self.max_len))
        self.relative_slot_bias = nn.Parameter(torch.empty(2 * self.max_len - 1))
        self.layer_pair_bias = nn.Parameter(torch.empty(self.sid_depth, self.sid_depth))
        self.input_dropout = nn.Dropout(self.dropout_prob)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout_prob,
            activation=resolve_ff_activation(self.ff_activation),
            batch_first=True,
            norm_first=True,
        )
        if self.ff_activation == "swiglu":
            # Pack gate/value projections; linear2 still consumes ff_dim features.
            layer.linear1 = nn.Linear(self.hidden_dim, 2 * self.ff_dim)
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self._reset_parameters()

    def forward(
        self,
        sid_seq: Tensor,
        attention_mask: Tensor,
        channel: str | None = None,
        available_mask: Tensor | None = None,
    ) -> HistoryEncoderOutput:
        token_output = self.build_token_embeddings(
            sid_seq,
            attention_mask,
            channel=channel,
            available_mask=available_mask,
        )
        return self.encode_token_embeddings(
            token_output.token_embeddings,
            token_output.token_attention_mask,
            token_output.attention_mask,
        )

    def build_token_embeddings(
        self,
        sid_seq: Tensor,
        attention_mask: Tensor,
        channel: str | None = None,
        available_mask: Tensor | None = None,
    ) -> HistoryTokenEmbeddingOutput:
        active_channel = str(channel or self.channel)
        if active_channel not in self.channels:
            raise ValueError(f"unknown history channel {active_channel!r}; expected one of {self.channels!r}")
        self._validate_inputs(sid_seq, attention_mask)
        sid_seq = sid_seq.long()
        attention_mask = attention_mask.bool()
        available_mask = self._resolve_available_mask(active_channel, sid_seq, attention_mask, available_mask)
        token_attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, self.sid_depth)
        self._validate_valid_history_tokens(sid_seq, token_attention_mask, attention_mask, active_channel, available_mask)
        token_emb = self._embed_history_tokens(
            sid_seq,
            token_attention_mask,
            active_channel,
            available_mask,
        )
        return HistoryTokenEmbeddingOutput(
            token_embeddings=token_emb,
            token_attention_mask=token_attention_mask,
            attention_mask=attention_mask,
            available_mask=available_mask,
        )

    def encode_token_embeddings(
        self,
        token_emb: Tensor,
        token_attention_mask: Tensor,
        attention_mask: Tensor,
    ) -> HistoryEncoderOutput:
        self._validate_token_embedding_inputs(token_emb, token_attention_mask, attention_mask)
        token_attention_mask = token_attention_mask.bool()
        attention_mask = attention_mask.bool()
        batch_size, seq_len, _, _ = token_emb.shape
        flat_emb = token_emb.reshape(batch_size, seq_len * self.sid_depth, self.hidden_dim)
        flat_mask = token_attention_mask.reshape(batch_size, seq_len * self.sid_depth)
        padding_mask = ~flat_mask
        if padding_mask[:, 0].any().item():
            padding_mask = padding_mask.clone()
            padding_mask[:, 0] = False

        flat_emb = self.input_dropout(flat_emb)
        attention_mask_2d = self._position_attention_bias(
            seq_len=seq_len,
            device=flat_emb.device,
            dtype=flat_emb.dtype,
        )
        encoder_padding_mask = self._float_key_padding_mask(padding_mask, flat_emb.dtype)
        encoded = self.encoder(flat_emb, mask=attention_mask_2d, src_key_padding_mask=encoder_padding_mask)
        encoded = encoded.masked_fill(~flat_mask.unsqueeze(-1), 0.0)

        return HistoryEncoderOutput(
            token_hidden_states=encoded.reshape(batch_size, seq_len, self.sid_depth, self.hidden_dim),
            token_attention_mask=token_attention_mask,
            attention_mask=attention_mask,
        )

    def _validate_token_embedding_inputs(
        self,
        token_emb: Tensor,
        token_attention_mask: Tensor,
        attention_mask: Tensor,
    ) -> None:
        expected_token_shape = token_emb.shape[:3]
        if token_emb.dim() != 4 or token_emb.shape[2] != self.sid_depth or token_emb.shape[3] != self.hidden_dim:
            raise ValueError(
                f"token_emb shape should be (B, L, {self.sid_depth}, {self.hidden_dim}), "
                f"got {tuple(token_emb.shape)}"
            )
        if token_attention_mask.dim() != 3 or tuple(token_attention_mask.shape) != tuple(expected_token_shape):
            raise ValueError(
                f"token_attention_mask shape should be {tuple(expected_token_shape)}, "
                f"got {tuple(token_attention_mask.shape)}"
            )
        if attention_mask.dim() != 2 or tuple(attention_mask.shape) != tuple(token_emb.shape[:2]):
            raise ValueError(
                f"attention_mask shape should be {tuple(token_emb.shape[:2])}, got {tuple(attention_mask.shape)}"
            )

    def _encoder_attention_mask(self, size: int, device: torch.device) -> Tensor | None:
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def _position_attention_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        slots = torch.arange(seq_len, device=device, dtype=torch.long).repeat_interleave(self.sid_depth)
        query_slots = slots.view(-1, 1)
        key_slots = slots.view(1, -1)
        relative_slots = key_slots - query_slots + self.max_len - 1

        bias = self.absolute_slot_bias[slots].view(1, -1).expand(slots.numel(), -1)
        bias = bias + self.relative_slot_bias[relative_slots]
        layers = torch.arange(self.sid_depth, device=device, dtype=torch.long).repeat(seq_len)
        bias = bias + self.layer_pair_bias[layers.view(-1, 1), layers.view(1, -1)]

        causal_mask = self._encoder_attention_mask(slots.numel(), device)
        if causal_mask is not None:
            bias = bias.masked_fill(causal_mask, float("-inf"))
        return bias.to(dtype=dtype)

    @staticmethod
    def _float_key_padding_mask(
        padding_mask: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        mask = torch.zeros(padding_mask.shape, dtype=dtype, device=padding_mask.device)
        return mask.masked_fill(padding_mask, float("-inf"))

    def _embed_history_tokens(
        self,
        sid_seq: Tensor,
        token_attention_mask: Tensor,
        channel: str,
        available_mask: Tensor,
    ) -> Tensor:
        output_codes = artifact_sid_to_output_codes(
            sid_seq,
            codebook_size=self.codebook_size,
            ignore_index=self.loss_ignore_index,
        )
        pieces = []
        observed_token_mask = token_attention_mask & available_mask.unsqueeze(-1)
        missing_token_mask = token_attention_mask & ~available_mask.unsqueeze(-1)
        for layer in range(self.sid_depth):
            layer_codes = output_codes[:, :, layer]
            emb = self.sid_embedding.embed_output_codes(
                channel,
                layer,
                layer_codes,
                ignore_index=self.loss_ignore_index,
                codes_validated=True,  # artifact_sid_to_output_codes checked the full tensor.
            )
            if channel in self.miss_embeddings:
                miss = self._missing_embedding(
                    channel,
                    emb,
                )
                emb = torch.where(missing_token_mask[:, :, layer].unsqueeze(-1), miss, emb)
            emb = emb.masked_fill(~observed_token_mask[:, :, layer].unsqueeze(-1) & ~missing_token_mask[:, :, layer].unsqueeze(-1), 0.0)
            emb = self.sid_input_proj(emb)
            pieces.append(emb)
        token_emb = torch.stack(pieces, dim=2)
        token_emb = self.input_norm(token_emb)
        return token_emb.masked_fill(~token_attention_mask.unsqueeze(-1), 0.0)

    def _missing_embedding(
        self,
        channel: str,
        reference: Tensor,
    ) -> Tensor:
        base = self.miss_embeddings[channel].view(1, 1, -1).expand_as(reference)
        return base

    def _validate_init(self) -> None:
        for name, value in (
            ("max_len", self.max_len),
            ("sid_depth", self.sid_depth),
            ("codebook_size", self.codebook_size),
            ("sid_embed_dim", self.sid_embed_dim),
            ("hidden_dim", self.hidden_dim),
            ("n_heads", self.n_heads),
            ("num_layers", self.num_layers),
            ("ff_dim", self.ff_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by n_heads, got {self.hidden_dim} and {self.n_heads}")
        if self.dropout_prob < 0.0 or self.dropout_prob >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout_prob}")
    def _validate_inputs(self, sid_seq: Tensor, attention_mask: Tensor) -> None:
        if sid_seq.dim() != 3 or sid_seq.shape[-1] != self.sid_depth:
            raise ValueError(f"sid_seq shape should be (B, L, {self.sid_depth}), got {tuple(sid_seq.shape)}")
        if sid_seq.shape[1] > self.max_len:
            raise ValueError(f"sid_seq length {sid_seq.shape[1]} exceeds max_len={self.max_len}")
        if attention_mask.dim() != 2 or tuple(attention_mask.shape) != tuple(sid_seq.shape[:2]):
            raise ValueError(
                f"attention_mask shape should be {tuple(sid_seq.shape[:2])}, got {tuple(attention_mask.shape)}"
            )

    def _validate_valid_history_tokens(
        self,
        sid_seq: Tensor,
        token_attention_mask: Tensor,
        attention_mask: Tensor,
        channel: str,
        available_mask: Tensor,
    ) -> None:
        valid_rows = attention_mask.bool()
        observed_rows = valid_rows & available_mask.bool()
        if observed_rows.any().item() and sid_seq[observed_rows].le(0).any().item():
            raise ValueError(f"{channel} observed history rows must contain positive SID codes")
        missing_rows = valid_rows & ~available_mask.bool()
        if channel == "collab" and missing_rows.any().item():
            raise ValueError("collab history rows marked valid must be available")
        if channel in {"text", "vision"} and missing_rows.any().item() and sid_seq[missing_rows].ne(0).any().item():
            raise ValueError(f"{channel} missing history rows must contain all-zero SID codes")
        invalid_rows = ~valid_rows
        if invalid_rows.any().item() and sid_seq[invalid_rows].ne(0).any().item():
            raise ValueError(f"{channel} history padding rows must contain all-zero SID codes")
        observed_token_mask = token_attention_mask & available_mask.unsqueeze(-1)
        if observed_token_mask.any().item():
            valid_codes = sid_seq[observed_token_mask]
            if int(valid_codes.max().item()) > self.codebook_size:
                raise ValueError(
                    f"{channel} history SID code exceeds codebook_size={self.codebook_size}: "
                    f"max={int(valid_codes.max().item())}"
                )

    def _resolve_available_mask(
        self,
        channel: str,
        sid_seq: Tensor,
        attention_mask: Tensor,
        available_mask: Tensor | None,
    ) -> Tensor:
        if available_mask is None:
            if channel == "collab":
                return attention_mask.bool()
            return attention_mask.bool() & sid_seq.gt(0).all(dim=-1)
        if available_mask.dim() != 2 or tuple(available_mask.shape) != tuple(attention_mask.shape):
            raise ValueError(
                f"available_mask shape should be {tuple(attention_mask.shape)}, got {tuple(available_mask.shape)}"
            )
        available = available_mask.bool()
        if (available & ~attention_mask.bool()).any().item():
            raise ValueError(f"{channel} available_mask cannot be true on padding positions")
        return available

    @staticmethod
    def _linear_or_identity(in_dim: int, out_dim: int) -> nn.Module:
        if int(in_dim) == int(out_dim):
            return nn.Identity()
        return nn.Linear(int(in_dim), int(out_dim), bias=False)

    def _reset_parameters(self) -> None:
        for miss in self.miss_embeddings.values():
            nn.init.normal_(miss, mean=0.0, std=0.02)
        nn.init.zeros_(self.absolute_slot_bias)
        nn.init.zeros_(self.relative_slot_bias)
        nn.init.zeros_(self.layer_pair_bias)
        if isinstance(self.sid_input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.sid_input_proj.weight)


class SIDHistoryMemoryEncoder(CausalSIDHistoryEncoder):
    """Encode historical SID tokens with bidirectional self-attention."""

    def _encoder_attention_mask(self, size: int, device: torch.device) -> Tensor | None:
        return None
