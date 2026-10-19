"""T5-style SID decoder with cross-attention over encoded history memory."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from models.feedforward import resolve_ff_activation
from models.sid_embedding import CodebookWeights, SIDCodeEmbedding, artifact_sid_to_output_codes


@dataclass
class SIDEncoderDecoderOutput:
    logits: Tensor
    hidden_states: Tensor
    target_sid: Tensor
    loss_ignore_index: int = -100


class SIDCrossAttentionDecoder(nn.Module):
    """Autoregressive SID decoder that cross-attends full history memory."""

    def __init__(
        self,
        channels: Sequence[str],
        sid_depth: int,
        codebook_size: int,
        sid_embed_dim: int,
        hidden_dim: int,
        n_heads: int,
        decoder_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
        codebook_weights: CodebookWeights = None,
        loss_ignore_index: int = -100,
        ff_activation: str = "relu",
    ):
        super().__init__()
        self.channels = tuple(str(channel) for channel in channels)
        self.sid_depth = int(sid_depth)
        self.codebook_size = int(codebook_size)
        self.sid_embed_dim = int(sid_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.decoder_layers = int(decoder_layers)
        self.ff_dim = int(ff_dim)
        self.ff_activation = str(ff_activation).lower()
        self.dropout_prob = float(dropout)
        self.loss_ignore_index = int(loss_ignore_index)
        self._validate_init()

        self.sid_embedding = SIDCodeEmbedding(
            channels=self.channels,
            sid_depth=self.sid_depth,
            codebook_size=self.codebook_size,
            sid_embed_dim=self.sid_embed_dim,
            codebook_weights=codebook_weights,
        )
        self.sid_input_proj = self._linear_or_identity(self.sid_embed_dim, self.hidden_dim)
        self.bos_tokens = nn.ParameterDict(
            {channel: nn.Parameter(torch.empty(self.hidden_dim)) for channel in self.channels}
        )
        self.depth_pair_bias = nn.Parameter(torch.empty(self.sid_depth, self.sid_depth))
        self.input_dropout = nn.Dropout(self.dropout_prob)

        layer = nn.TransformerDecoderLayer(
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
        self.decoder = nn.TransformerDecoder(layer, num_layers=self.decoder_layers)
        self.output_heads = nn.ModuleDict()
        for channel in self.channels:
            self.output_heads[channel] = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.codebook_size, bias=False) for _ in range(self.sid_depth)]
            )
        self._reset_parameters()

    def forward(
        self,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        target_sid: Tensor,
        channel: str = "collab",
    ) -> SIDEncoderDecoderOutput:
        self._validate_channel(channel)
        memory_key_padding_mask = self._validate_memory(memory, memory_key_padding_mask)
        if target_sid.dim() != 2 or target_sid.shape[1] != self.sid_depth:
            raise ValueError(f"target_sid shape should be (B, {self.sid_depth}), got {tuple(target_sid.shape)}")
        if target_sid.shape[0] != memory.shape[0]:
            raise ValueError(f"target batch {target_sid.shape[0]} != memory batch {memory.shape[0]}")

        target_output = artifact_sid_to_output_codes(
            target_sid,
            codebook_size=self.codebook_size,
            ignore_index=self.loss_ignore_index,
        )
        prefix = target_output[:, :-1]
        hidden = self._decode_prefix(
            memory,
            memory_key_padding_mask,
            prefix,
            channel,
            target_length=self.sid_depth,
            codes_validated=True,  # target_output passed artifact SID range validation.
        )
        logits = self._project_logits(channel, hidden).float()
        return SIDEncoderDecoderOutput(
            logits=logits,
            hidden_states=hidden,
            target_sid=target_output,
            loss_ignore_index=self.loss_ignore_index,
        )

    def logits_for_prefix_context(
        self,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        prefix_sid: Tensor,
        channel: str = "collab",
    ) -> Tensor:
        """Return logits for the next SID level from output-domain prefix codes."""
        self._validate_channel(channel)
        memory_key_padding_mask = self._validate_memory(memory, memory_key_padding_mask)
        if prefix_sid.dim() != 2:
            raise ValueError(f"prefix_sid shape should be (B, step), got {tuple(prefix_sid.shape)}")
        if prefix_sid.shape[0] != memory.shape[0]:
            raise ValueError(f"prefix batch {prefix_sid.shape[0]} != memory batch {memory.shape[0]}")
        step = int(prefix_sid.shape[1])
        if step >= self.sid_depth:
            raise ValueError(f"prefix length {step} reached SID depth {self.sid_depth}")

        self._validate_output_prefix(channel, prefix_sid)
        hidden = self._decode_prefix(
            memory,
            memory_key_padding_mask,
            prefix_sid.long(),
            channel,
            target_length=step + 1,
        )
        return self._project_layer_logits(channel, step, hidden[:, step, :]).float()

    def history_summary(
        self,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        channel: str = "collab",
    ) -> Tensor:
        """Decode only the branch BOS token against encoded history memory."""
        self._validate_channel(channel)
        memory_key_padding_mask = self._validate_memory(memory, memory_key_padding_mask)
        empty_prefix = torch.empty(
            (memory.shape[0], 0),
            dtype=torch.long,
            device=memory.device,
        )
        hidden = self._decode_prefix(
            memory,
            memory_key_padding_mask,
            empty_prefix,
            channel,
            target_length=1,
        )
        return hidden[:, 0, :]

    def _decode_prefix(
        self,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        prefix_output_codes: Tensor,
        channel: str,
        target_length: int,
        codes_validated: bool = False,
    ) -> Tensor:
        batch_size, prefix_len = prefix_output_codes.shape
        if target_length < 1 or target_length > self.sid_depth:
            raise ValueError(f"target_length must be in [1, {self.sid_depth}], got {target_length}")
        if prefix_len != target_length - 1:
            raise ValueError(
                f"prefix length {prefix_len} is incompatible with target_length {target_length}"
            )

        pieces = [self.bos_tokens[channel].view(1, 1, -1).expand(batch_size, 1, -1)]
        for layer in range(prefix_len):
            code_embedding = self.sid_embedding.embed_output_codes(
                channel,
                layer,
                prefix_output_codes[:, layer],
                ignore_index=self.loss_ignore_index,
                codes_validated=codes_validated,
            )
            pieces.append(self.sid_input_proj(code_embedding).unsqueeze(1))

        sequence = torch.cat(pieces, dim=1)
        sequence = self.input_dropout(sequence)
        causal_mask = torch.triu(
            torch.ones(target_length, target_length, dtype=torch.bool, device=sequence.device),
            diagonal=1,
        )
        target_mask = self.depth_pair_bias[:target_length, :target_length].to(dtype=sequence.dtype)
        target_mask = target_mask.masked_fill(causal_mask, float("-inf"))
        # Disable causal-mask autodetection so a zero-initialized learned bias remains in the autograd graph.
        return self.decoder(
            sequence,
            memory,
            tgt_mask=target_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=False,
        )

    def _project_logits(self, channel: str, hidden: Tensor) -> Tensor:
        logits = [
            self._project_layer_logits(channel, layer, hidden[:, layer, :])
            for layer in range(self.sid_depth)
        ]
        return torch.stack(logits, dim=1)

    def _project_layer_logits(self, channel: str, layer: int, hidden: Tensor) -> Tensor:
        return self.output_heads[channel][layer](hidden)

    def _validate_output_prefix(self, channel: str, prefix_sid: Tensor) -> None:
        for layer in range(prefix_sid.shape[1]):
            self.sid_embedding.embed_output_codes(
                channel,
                layer,
                prefix_sid[:, layer],
                ignore_index=self.loss_ignore_index,
            )

    def _validate_memory(self, memory: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        if memory.dim() != 3 or memory.shape[2] != self.hidden_dim:
            raise ValueError(f"memory shape should be (B, M, {self.hidden_dim}), got {tuple(memory.shape)}")
        if memory_key_padding_mask.dim() != 2 or tuple(memory_key_padding_mask.shape) != tuple(memory.shape[:2]):
            raise ValueError(
                "memory_key_padding_mask shape should match memory[:2], "
                f"got {tuple(memory_key_padding_mask.shape)} vs {tuple(memory.shape[:2])}"
            )
        mask = memory_key_padding_mask.bool().to(memory.device)
        all_masked = mask.all(dim=1)
        if all_masked.any().item():
            mask = mask.clone()
            mask[all_masked, 0] = False
        return mask

    def _validate_init(self) -> None:
        if not self.channels:
            raise ValueError("channels must be non-empty")
        for name, value in (
            ("sid_depth", self.sid_depth),
            ("codebook_size", self.codebook_size),
            ("sid_embed_dim", self.sid_embed_dim),
            ("hidden_dim", self.hidden_dim),
            ("n_heads", self.n_heads),
            ("decoder_layers", self.decoder_layers),
            ("ff_dim", self.ff_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by n_heads, got {self.hidden_dim} and {self.n_heads}")
        if self.dropout_prob < 0.0 or self.dropout_prob >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout_prob}")

    def _validate_channel(self, channel: str) -> None:
        if channel not in self.channels:
            raise ValueError(f"unknown SID channel {channel!r}; expected one of {self.channels!r}")

    @staticmethod
    def _linear_or_identity(in_dim: int, out_dim: int) -> nn.Module:
        if int(in_dim) == int(out_dim):
            return nn.Identity()
        return nn.Linear(int(in_dim), int(out_dim), bias=False)

    def _reset_parameters(self) -> None:
        for token in self.bos_tokens.values():
            nn.init.normal_(token, mean=0.0, std=0.02)
        nn.init.zeros_(self.depth_pair_bias)
        if isinstance(self.sid_input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.sid_input_proj.weight)
        for heads in self.output_heads.values():
            for head in heads:
                nn.init.xavier_uniform_(head.weight)
