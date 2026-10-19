"""Block-structured MP-SID RQ-VAE skeleton for unified SID tokenization."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.rqvae import (
    RQMLP,
    _decoder_dims,
    _encoder_dims,
    _log_if_available,
    _normalize_sk_epsilons,
    _synchronize_if_cuda,
)
from utils.data import MPSID_INPUT_MODALITIES


@dataclass
class MPSIDRQVAEOutput:
    loss: Tensor
    recon_loss: Tensor
    quantize_loss: Tensor
    quantized: Tensor
    latent: Tensor
    semantic_codes: Tensor
    all_ids: list[Tensor]
    reconstructions: dict[str, Tensor]
    raw_blocks: dict[str, Tensor]
    latent_blocks: dict[str, Tensor]
    modality_mask: Tensor
    recon_loss_by_modality: dict[str, Tensor]
    rec_loss_observed: Tensor | None = None
    cross_modal_pred_loss: Tensor | None = None
    cross_modal_pred_weighted_loss: Tensor | None = None
    pred_loss_text: Tensor | None = None
    pred_loss_vision: Tensor | None = None
    cross_modal_pred_skipped: bool = False
    cross_modal_pred_full_count: int = 0
    predictive_view_codes: dict[str, Tensor] | None = None


def _encoder_dims_for_mode(input_dim: int, output_dim: int, enc_dec_mode: str) -> list[int]:
    if enc_dec_mode == "linear":
        return [input_dim, output_dim]
    return _encoder_dims(input_dim, output_dim, hidden_dims=None, enc_dec_type="MPL")


def _decoder_dims_for_mode(input_dim: int, output_dim: int, enc_dec_mode: str) -> list[int]:
    if enc_dec_mode == "linear":
        return [input_dim, output_dim]
    return _decoder_dims(input_dim, output_dim, hidden_dims=None, enc_dec_type="MPL")


class _RawBlockDecoder(nn.Module):
    def __init__(
        self,
        dims: list[int],
        dropout_prob: float,
        use_batch_norm: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.net = RQMLP(
            dims,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)


class PartialModalityQuantizeLayer(nn.Module):
    """Quantize with availability-aware distance over observed latent blocks."""

    def __init__(
        self,
        codebook_size: int,
        n_modalities: int,
        block_dim: int,
        commitment_weight: float = 0.25,
        assignment_mode: str = "argmin",
        sk_epsilon: float = 0.0,
        sk_iters: int = 50,
        use_sinkhorn_train: bool = True,
        use_sinkhorn_eval: bool = False,
    ) -> None:
        super().__init__()
        self.n_modalities = int(n_modalities)
        self.block_dim = int(block_dim)
        if self.n_modalities <= 0:
            raise ValueError(f"n_modalities should be positive, got {self.n_modalities}")
        if self.block_dim <= 0:
            raise ValueError(f"block_dim should be positive, got {self.block_dim}")
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.assignment_mode = str(assignment_mode).lower()
        if self.assignment_mode not in {"argmin", "sinkhorn"}:
            raise ValueError(f"model.assignment_mode should be 'argmin' or 'sinkhorn', got {assignment_mode!r}")
        self.sk_epsilon = float(sk_epsilon)
        if self.sk_epsilon < 0.0:
            raise ValueError(f"model.sk_epsilons values must be non-negative, got {self.sk_epsilon}")
        self.sk_iters = max(1, int(sk_iters))
        self.use_sinkhorn_train = bool(use_sinkhorn_train)
        self.use_sinkhorn_eval = bool(use_sinkhorn_eval)
        self.codebook = nn.Embedding(
            self.codebook_size,
            self.n_modalities * self.block_dim,
        )
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def get_codebook(self) -> Tensor:
        return self.codebook.weight

    @staticmethod
    def _center_distance_for_constraint(distances: Tensor) -> Tensor:
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) * 0.5
        amplitude = (max_distance - middle).clamp_min(1e-5)
        return (distances - middle) / amplitude

    @staticmethod
    @torch.no_grad()
    def _sinkhorn_algorithm(distances: Tensor, epsilon: float, sinkhorn_iterations: int) -> Tensor:
        logits = -distances.double() / max(float(epsilon), 1e-12)
        logits = logits - logits.max(dim=1, keepdim=True).values
        q = torch.exp(logits)

        total = q.sum().clamp_min(1e-12)
        q = q / total
        batch_size = q.shape[0]
        codebook_size = q.shape[1]

        for _ in range(int(sinkhorn_iterations)):
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q = q / batch_size
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
            q = q / codebook_size

        return (q * batch_size).to(dtype=distances.dtype)

    def _should_use_sinkhorn(self) -> bool:
        if self.assignment_mode != "sinkhorn" or self.sk_epsilon <= 0.0:
            return False
        if self.training:
            return self.use_sinkhorn_train
        return self.use_sinkhorn_eval

    def _assign_codes_with_sinkhorn(self, dist: Tensor, use_sinkhorn: bool) -> Tensor:
        argmin_ids = dist.detach().argmin(dim=-1)
        if not use_sinkhorn:
            return argmin_ids

        centered = self._center_distance_for_constraint(dist.detach())
        assignment = self._sinkhorn_algorithm(centered, self.sk_epsilon, self.sk_iters)
        if torch.isnan(assignment).any() or torch.isinf(assignment).any():
            return argmin_ids
        return assignment.argmax(dim=-1).long()

    def _assign_codes(self, dist: Tensor) -> Tensor:
        return self._assign_codes_with_sinkhorn(dist, self._should_use_sinkhorn())

    @staticmethod
    def compute_partial_distance(
        x: Tensor,
        centers: Tensor,
        modality_mask: Tensor,
        n_modalities: int,
        block_dim: int,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"x should be a 2D Tensor, got shape {tuple(x.shape)}")
        if centers.dim() != 2:
            raise ValueError(f"centers should be a 2D Tensor, got shape {tuple(centers.shape)}")
        expected_dim = int(n_modalities) * int(block_dim)
        if x.shape[1] != expected_dim:
            raise ValueError(f"x last dim should be {expected_dim}, got {x.shape[1]}")
        if centers.shape[1] != expected_dim:
            raise ValueError(f"centers last dim should be {expected_dim}, got {centers.shape[1]}")
        if tuple(modality_mask.shape) != (x.shape[0], int(n_modalities)):
            raise ValueError(
                f"modality_mask shape should be ({x.shape[0]}, {int(n_modalities)}), "
                f"got {tuple(modality_mask.shape)}"
            )

        mask = modality_mask.to(device=x.device, dtype=torch.bool)
        observed_counts = mask.sum(dim=-1)
        if (observed_counts == 0).any().item():
            raise ValueError("partial-modality distance requires at least one observed block per row")

        x_blocks = x.reshape(x.shape[0], int(n_modalities), int(block_dim))
        center_blocks = centers.to(device=x.device, dtype=x.dtype).reshape(
            centers.shape[0],
            int(n_modalities),
            int(block_dim),
        )
        mask_float = mask.to(dtype=x.dtype)
        dist = x.new_zeros((x.shape[0], centers.shape[0]))
        for modality_idx in range(int(n_modalities)):
            x_block = x_blocks[:, modality_idx, :]
            center_block = center_blocks[:, modality_idx, :]
            block_dist = (
                x_block.pow(2).sum(dim=-1, keepdim=True)
                + center_block.pow(2).sum(dim=-1)
                - 2.0 * x_block @ center_block.t()
            ) / float(block_dim)
            dist = dist + block_dist * mask_float[:, modality_idx].unsqueeze(-1)
        return dist / observed_counts.to(dtype=x.dtype).unsqueeze(-1)

    def partial_distance(self, x: Tensor, centers: Tensor, modality_mask: Tensor) -> Tensor:
        return self.compute_partial_distance(
            x,
            centers,
            modality_mask,
            self.n_modalities,
            self.block_dim,
        )

    def assign_argmin(self, x: Tensor, modality_mask: Tensor) -> Tensor:
        dist = self.partial_distance(x, self.get_codebook(), modality_mask)
        return dist.argmin(dim=-1)

    def forward(
        self,
        x: Tensor,
        modality_mask: Tensor,
        compute_loss: bool = True,
        soft_indices: bool = False,
        soft_indices_temperature: float = 0.1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cb = self.get_codebook()
        dist = self.partial_distance(x, cb, modality_mask)
        ids = self._assign_codes(dist)
        hard_emb = cb[ids]

        if soft_indices and torch.is_grad_enabled():
            temperature = float(soft_indices_temperature)
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(
                    "model.soft_indices_temperature must be finite and positive, "
                    f"got {soft_indices_temperature}"
                )
            soft_assignment = F.softmax(-dist.float() / temperature, dim=-1)
            soft_assignment = soft_assignment.to(dtype=cb.dtype)
            hard_assignment = F.one_hot(ids, num_classes=self.codebook_size).to(
                device=soft_assignment.device,
                dtype=soft_assignment.dtype,
            )
            assignment = soft_assignment + (hard_assignment - soft_assignment).detach()
            emb = assignment @ cb
        elif not self.training:
            emb = hard_emb
        else:
            emb = x + (hard_emb - x).detach()

        q_loss = (
            self._compute_loss(x, hard_emb, modality_mask)
            if compute_loss
            else x.new_zeros(x.shape[0])
        )
        return emb, ids, q_loss

    def _compute_loss(self, x: Tensor, emb: Tensor, modality_mask: Tensor) -> Tensor:
        codebook_loss = self._row_partial_distance(x.detach(), emb, modality_mask)
        commit_loss = self._row_partial_distance(x, emb.detach(), modality_mask)
        return codebook_loss + self.commitment_weight * commit_loss

    def _row_partial_distance(self, x: Tensor, emb: Tensor, modality_mask: Tensor) -> Tensor:
        mask = modality_mask.to(device=x.device, dtype=torch.bool)
        observed_counts = mask.sum(dim=-1)
        if (observed_counts == 0).any().item():
            raise ValueError("partial-modality loss requires at least one observed block per row")
        x_blocks = x.reshape(x.shape[0], self.n_modalities, self.block_dim)
        emb_blocks = emb.reshape(emb.shape[0], self.n_modalities, self.block_dim)
        block_loss = (x_blocks - emb_blocks).pow(2).sum(dim=-1) / float(self.block_dim)
        mask_float = mask.to(dtype=x.dtype)
        return (block_loss * mask_float).sum(dim=-1) / observed_counts.to(dtype=x.dtype)


def _global_observed_block_means(
    data: Tensor,
    modality_mask: Tensor,
    n_modalities: int,
    block_dim: int,
) -> Tensor:
    blocks = data.reshape(data.shape[0], int(n_modalities), int(block_dim))
    mask = modality_mask.to(device=data.device, dtype=torch.bool)
    means = data.new_zeros((int(n_modalities), int(block_dim)))
    for modality_idx in range(int(n_modalities)):
        observed = mask[:, modality_idx]
        if observed.any().item():
            means[modality_idx] = blocks[observed, modality_idx, :].mean(dim=0)
    return means


def _initial_masked_centers(
    data: Tensor,
    modality_mask: Tensor,
    n_clusters: int,
    init_idx: int,
    random_state: int | None,
    global_means: Tensor,
    n_modalities: int,
    block_dim: int,
    center_init: str,
) -> Tensor:
    n_samples = int(data.shape[0])
    center_init = str(center_init).lower()
    if center_init not in {"global_mean", "full_exist"}:
        raise ValueError(f"unknown kmeans center init mode: {center_init!r}")
    if center_init == "full_exist":
        full_rows = modality_mask.to(device=data.device, dtype=torch.bool).all(dim=-1)
        full_indices = full_rows.nonzero(as_tuple=True)[0]
        if int(full_indices.numel()) < int(n_clusters):
            raise ValueError(
                "model.kmeans_center_init='full_exist' requires at least "
                f"{int(n_clusters)} rows with all modality blocks observed, got {int(full_indices.numel())}"
            )
        if random_state is None:
            order = torch.randperm(int(full_indices.numel()), device=data.device)[:n_clusters]
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_state) + int(init_idx))
            order = torch.randperm(int(full_indices.numel()), generator=generator, device="cpu")[:n_clusters]
            order = order.to(device=data.device)
        return data.index_select(0, full_indices.index_select(0, order)).clone()

    if random_state is None:
        init_indices = torch.randperm(n_samples, device="cpu")[:n_clusters].to(device=data.device)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_state) + int(init_idx))
        init_indices = torch.randperm(n_samples, generator=generator, device="cpu")[:n_clusters].to(device=data.device)

    centers = data.index_select(0, init_indices).clone()
    center_mask = modality_mask.index_select(0, init_indices).to(device=data.device, dtype=torch.bool)
    center_blocks = centers.reshape(n_clusters, int(n_modalities), int(block_dim))
    center_blocks = torch.where(
        center_mask.unsqueeze(-1),
        center_blocks,
        global_means.unsqueeze(0),
    )
    return center_blocks.reshape(n_clusters, int(n_modalities) * int(block_dim))


def _assignment_from_partial_distance(
    dist: Tensor,
    n_clusters: int,
    *,
    use_sinkhorn: bool,
    epsilon: float,
    sinkhorn_iterations: int,
) -> tuple[Tensor, Tensor]:
    labels = dist.argmin(dim=-1)
    if not use_sinkhorn:
        assignment = F.one_hot(labels, num_classes=int(n_clusters)).to(device=dist.device, dtype=dist.dtype)
        return assignment, labels

    centered = PartialModalityQuantizeLayer._center_distance_for_constraint(dist)
    assignment = PartialModalityQuantizeLayer._sinkhorn_algorithm(centered, epsilon, sinkhorn_iterations)
    if torch.isnan(assignment).any() or torch.isinf(assignment).any():
        assignment = F.one_hot(labels, num_classes=int(n_clusters)).to(device=dist.device, dtype=dist.dtype)
        return assignment, labels
    return assignment, assignment.argmax(dim=-1).long()


def _masked_center_update(
    data: Tensor,
    modality_mask: Tensor,
    assignment: Tensor,
    previous_centers: Tensor,
    n_modalities: int,
    block_dim: int,
) -> Tensor:
    data_blocks = data.reshape(data.shape[0], int(n_modalities), int(block_dim))
    next_blocks = previous_centers.reshape(previous_centers.shape[0], int(n_modalities), int(block_dim)).clone()
    mask = modality_mask.to(device=data.device, dtype=data.dtype)
    for modality_idx in range(int(n_modalities)):
        weights = assignment * mask[:, modality_idx].unsqueeze(-1)
        denom = weights.sum(dim=0)
        active_clusters = denom > 1.0e-12
        if active_clusters.any().item():
            weighted_sum = weights.t() @ data_blocks[:, modality_idx, :]
            next_blocks[active_clusters, modality_idx, :] = (
                weighted_sum[active_clusters] / denom[active_clusters].unsqueeze(-1)
            )
    return next_blocks.reshape(previous_centers.shape[0], int(n_modalities) * int(block_dim))


@torch.no_grad()
def _fit_masked_sinkhorn_balanced_kmeans(
    data: Tensor,
    modality_mask: Tensor,
    n_clusters: int,
    *,
    n_modalities: int,
    block_dim: int,
    n_init: int,
    max_iter: int,
    random_state: int | None,
    epsilon: float,
    sinkhorn_iterations: int,
    use_sinkhorn: bool,
    center_init: str = "global_mean",
    tolerance: float = 1.0e-4,
) -> tuple[Tensor, Tensor]:
    """Fit masked residual K-means on the input tensor's device."""
    n_samples = int(data.shape[0])
    n_clusters = int(n_clusters)
    if n_clusters <= 0:
        raise ValueError(f"n_clusters should be positive, got {n_clusters}")
    if n_samples < n_clusters:
        raise ValueError(f"masked K-means requires n_samples >= n_clusters, got {n_samples} < {n_clusters}")

    work = data.detach().float()
    mask = modality_mask.to(device=work.device, dtype=torch.bool)
    expected_mask = (n_samples, int(n_modalities))
    if tuple(mask.shape) != expected_mask:
        raise ValueError(f"modality_mask shape should be {expected_mask}, got {tuple(mask.shape)}")
    if (mask.sum(dim=-1) == 0).any().item():
        raise ValueError("masked K-means requires at least one observed block per row")

    n_init = max(1, int(n_init))
    max_iter = max(1, int(max_iter))
    sinkhorn_iterations = max(1, int(sinkhorn_iterations))
    epsilon = max(float(epsilon), 1.0e-12)
    tolerance = max(0.0, float(tolerance))
    global_means = _global_observed_block_means(work, mask, n_modalities, block_dim)

    best_centers: Tensor | None = None
    best_labels: Tensor | None = None
    best_score = float("inf")

    for init_idx in range(n_init):
        centers = _initial_masked_centers(
            work,
            mask,
            n_clusters,
            init_idx,
            random_state,
            global_means,
            n_modalities,
            block_dim,
            center_init,
        )

        for _ in range(max_iter):
            dist = PartialModalityQuantizeLayer.compute_partial_distance(
                work,
                centers,
                mask,
                n_modalities,
                block_dim,
            )
            assignment, _ = _assignment_from_partial_distance(
                dist,
                n_clusters,
                use_sinkhorn=use_sinkhorn,
                epsilon=epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
            )
            next_centers = _masked_center_update(work, mask, assignment, centers, n_modalities, block_dim)
            if torch.isnan(next_centers).any() or torch.isinf(next_centers).any():
                break

            center_shift = (next_centers - centers).pow(2).sum().sqrt().item()
            centers = next_centers
            if center_shift <= tolerance:
                break

        final_dist = PartialModalityQuantizeLayer.compute_partial_distance(
            work,
            centers,
            mask,
            n_modalities,
            block_dim,
        )
        final_assignment, final_labels = _assignment_from_partial_distance(
            final_dist,
            n_clusters,
            use_sinkhorn=use_sinkhorn,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        final_score = (final_assignment * final_dist).sum().item()
        if final_score < best_score:
            best_score = final_score
            best_centers = centers.clone()
            best_labels = final_labels.clone()

    if best_centers is None or best_labels is None:
        raise RuntimeError("masked K-means failed to produce centers")
    return best_centers.to(dtype=data.dtype), best_labels


class MPSIDRqVae(nn.Module):
    """MP-SID block-structured tokenizer with availability-aware quantization."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.modalities = tuple(MPSID_INPUT_MODALITIES)
        self.raw_dims = {
            "collab": int(config.model.collab_input_dim),
            "text": int(config.model.text_input_dim),
            "vision": int(config.model.visual_input_dim),
        }
        self.latent_block_dim = int(config.model.latent_block_dim)
        self.codebook_size = int(config.model.codebook_size)
        self.n_layers = int(config.model.n_layers)
        self.use_bpd = getattr(config.model, "use_bpd", True)
        if not isinstance(self.use_bpd, bool):
            raise ValueError("model.use_bpd must be a boolean")
        self.quant_loss_weight = float(getattr(config.model, "quant_loss_weight", 1.0))
        self.cross_modal_pred_weight = max(0.0, float(getattr(config.model, "cross_modal_pred_weight", 1.0e-4)))
        self.soft_indices = bool(getattr(config.model, "soft_indices", False))
        self.soft_indices_temperature = float(getattr(config.model, "soft_indices_temperature", 0.1))
        if not math.isfinite(self.soft_indices_temperature) or self.soft_indices_temperature <= 0.0:
            raise ValueError(
                "model.soft_indices_temperature must be finite and positive, "
                f"got {self.soft_indices_temperature}"
            )

        self.enc_dec_mode = str(getattr(config.model, "enc_dec_mode", "linear")).lower()
        if self.enc_dec_mode not in {"linear", "mlp"}:
            raise ValueError(
                "model.enc_dec_mode should be 'linear' or 'mlp', "
                f"got {self.enc_dec_mode!r}"
            )
        self.dropout_prob = float(getattr(config.model, "dropout_prob", 0.0))
        self.use_batch_norm = bool(getattr(config.model, "use_batch_norm", False))
        self.activation = str(getattr(config.model, "activation", "leaky_relu"))

        self.raw_encoders = nn.ModuleDict(
            {
                modality: RQMLP(
                    _encoder_dims_for_mode(
                        self.raw_dims[modality],
                        self.latent_block_dim,
                        self.enc_dec_mode,
                    ),
                    dropout_prob=self.dropout_prob,
                    use_batch_norm=self.use_batch_norm,
                    activation=self.activation,
                )
                for modality in self.modalities
            }
        )

        self.raw_decoders = nn.ModuleDict(
            {
                modality: _RawBlockDecoder(
                    dims=_decoder_dims_for_mode(
                        self.latent_block_dim,
                        self.raw_dims[modality],
                        self.enc_dec_mode,
                    ),
                    dropout_prob=self.dropout_prob,
                    use_batch_norm=self.use_batch_norm,
                    activation=self.activation,
                )
                for modality in self.modalities
            }
        )

        assignment_mode = str(getattr(config.model, "assignment_mode", "argmin")).lower()
        sk_epsilons = _normalize_sk_epsilons(getattr(config.model, "sk_epsilons", None), self.n_layers)
        sk_iters = int(getattr(config.model, "sk_iters", 50))
        use_sinkhorn_train = bool(getattr(config.model, "use_sinkhorn_train", True))
        use_sinkhorn_eval = bool(getattr(config.model, "use_sinkhorn_eval", False))
        commitment_weight = float(config.model.commitment_weight)
        self.quantize_layers = nn.ModuleList(
            [
                PartialModalityQuantizeLayer(
                    self.codebook_size,
                    n_modalities=len(self.modalities),
                    block_dim=self.latent_block_dim,
                    commitment_weight=commitment_weight,
                    assignment_mode=assignment_mode,
                    sk_epsilon=sk_epsilons[layer_idx],
                    sk_iters=sk_iters,
                    use_sinkhorn_train=use_sinkhorn_train,
                    use_sinkhorn_eval=use_sinkhorn_eval,
                )
                for layer_idx in range(self.n_layers)
            ]
        )
        self._kmeans_initted = False

    def forward(self, batch: Mapping[str, Tensor]) -> MPSIDRQVAEOutput:
        inputs = self._batch_inputs(batch)
        modality_mask = inputs["modality_mask"]
        core = self._run_view(inputs, modality_mask)
        raw_blocks = core["raw_blocks"]
        latent_blocks = core["latent_blocks"]
        latent = core["latent"]
        quantized = core["quantized"]
        total_q_loss = core["total_q_loss"]
        all_ids = core["all_ids"]
        reconstructions = core["reconstructions"]
        recon_loss, recon_loss_by_modality = self._raw_recon_loss(reconstructions, inputs, modality_mask)
        quantize_loss = total_q_loss.mean() / max(self.n_layers, 1)
        pred_weight = self.cross_modal_pred_weight
        pred_info = self._cross_modal_predictive_loss(inputs, modality_mask, pred_weight)
        loss = (
            recon_loss
            + self.quant_loss_weight * quantize_loss
            + pred_info["weighted_loss"]
        )

        return MPSIDRQVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            quantize_loss=quantize_loss,
            quantized=quantized,
            latent=latent,
            semantic_codes=torch.stack(all_ids, dim=-1),
            all_ids=all_ids,
            reconstructions=reconstructions,
            raw_blocks=raw_blocks,
            latent_blocks=latent_blocks,
            modality_mask=modality_mask,
            recon_loss_by_modality=recon_loss_by_modality,
            rec_loss_observed=recon_loss,
            cross_modal_pred_loss=pred_info["loss"],
            cross_modal_pred_weighted_loss=pred_info["weighted_loss"],
            pred_loss_text=pred_info["pred_loss_text"],
            pred_loss_vision=pred_info["pred_loss_vision"],
            cross_modal_pred_skipped=bool(pred_info["skipped"]),
            cross_modal_pred_full_count=int(pred_info["full_count"]),
            predictive_view_codes=pred_info["view_codes"],
        )

    def _run_view(
        self,
        inputs: Mapping[str, Tensor],
        modality_mask: Tensor,
        decode_modalities: tuple[str, ...] | None = None,
        compute_q_loss: bool = True,
        decode_from_hard_codebook: bool = False,
        soft_indices: bool = False,
    ) -> dict[str, object]:
        raw_blocks = self._prepare_raw_blocks(inputs, modality_mask)
        latent_blocks = self._encode_latent(raw_blocks, modality_mask)
        latent = torch.cat([latent_blocks[modality] for modality in self.modalities], dim=-1)
        quantized, total_q_loss, all_ids = self._quantize(
            latent,
            modality_mask,
            compute_q_loss=compute_q_loss,
            soft_indices=soft_indices,
        )
        decoder_quantized = quantized
        if decode_from_hard_codebook:
            # A masked target block has no STE path back to its encoder. Re-look up
            # live codebooks so predictive reconstruction updates exported blocks.
            decoder_quantized = sum(
                layer.get_codebook()[ids]
                for layer, ids in zip(self.quantize_layers, all_ids)
            )
        quantized_blocks = self._split_blocks(decoder_quantized)
        requested_modalities = self.modalities if decode_modalities is None else tuple(decode_modalities)
        unknown_modalities = set(requested_modalities) - set(self.modalities)
        if unknown_modalities:
            raise ValueError(f"unknown reconstruction modalities: {sorted(unknown_modalities)}")
        # In training, unused decoder calls must be retained whenever they could
        # consume RNG or update running statistics. The unified config uses p=0
        # and no BatchNorm, so predictive views can decode only their loss targets.
        preserve_stateful_calls = self.training and (self.dropout_prob > 0.0 or self.use_batch_norm)
        active_modalities = self.modalities if preserve_stateful_calls else requested_modalities
        reconstructions = {
            modality: self.raw_decoders[modality](quantized_blocks[modality])
            for modality in active_modalities
        }
        return {
            "raw_blocks": raw_blocks,
            "latent_blocks": latent_blocks,
            "latent": latent,
            "quantized": quantized,
            "total_q_loss": total_q_loss,
            "all_ids": all_ids,
            "reconstructions": reconstructions,
        }

    def _cross_modal_predictive_loss(
        self,
        inputs: Mapping[str, Tensor],
        modality_mask: Tensor,
        pred_weight: float,
    ) -> dict[str, object]:
        zero = modality_mask.new_zeros((), dtype=next(self.parameters()).dtype)
        empty_info = {
            "loss": zero,
            "weighted_loss": zero,
            "pred_loss_text": zero,
            "pred_loss_vision": zero,
            "skipped": True,
            "full_count": 0,
            "view_codes": {},
        }
        if pred_weight <= 0.0:
            return empty_info

        full_rows = modality_mask.all(dim=-1)
        full_count = int(full_rows.sum().item())
        if full_count == 0:
            return empty_info

        full_inputs: dict[str, Tensor] = {
            modality: inputs[modality][full_rows]
            for modality in self.modalities
        }
        view_specs = (
            ("collab_only", (True, False, False), ("text", "vision")),
            ("collab_text", (True, True, False), ("vision",)),
            ("collab_vision", (True, False, True), ("text",)),
        )
        view_losses: list[Tensor] = []
        text_losses: list[Tensor] = []
        vision_losses: list[Tensor] = []
        view_codes: dict[str, Tensor] = {}
        for view_name, mask_values, targets in view_specs:
            view_mask = torch.tensor(mask_values, device=modality_mask.device, dtype=torch.bool)
            view_mask = view_mask.unsqueeze(0).expand(full_count, -1)
            view_core = self._run_view(
                full_inputs,
                view_mask,
                decode_modalities=targets,
                compute_q_loss=False,
                decode_from_hard_codebook=not self.soft_indices,
                soft_indices=self.soft_indices,
            )
            reconstructions = view_core["reconstructions"]
            target_losses: list[Tensor] = []
            for target in targets:
                diff = F.mse_loss(
                    reconstructions[target],
                    full_inputs[target],
                    reduction="mean",
                )
                # Match cosine-error weights across normalized raw modalities.
                diff = diff * (self.raw_dims[target] / self.raw_dims["collab"])
                target_losses.append(diff)
                if target == "text":
                    text_losses.append(diff)
                elif target == "vision":
                    vision_losses.append(diff)
            view_losses.append(torch.stack(target_losses).mean())
            view_codes[view_name] = torch.stack(view_core["all_ids"], dim=-1)

        pred_loss = torch.stack(view_losses).mean()
        pred_loss_text = torch.stack(text_losses).mean() if text_losses else zero
        pred_loss_vision = torch.stack(vision_losses).mean() if vision_losses else zero
        return {
            "loss": pred_loss,
            "weighted_loss": pred_loss * pred_weight,
            "pred_loss_text": pred_loss_text,
            "pred_loss_vision": pred_loss_vision,
            "skipped": False,
            "full_count": full_count,
            "view_codes": view_codes,
        }

    def _batch_inputs(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        required = {
            "collab": "collab_embedding",
            "text": "text_embedding",
            "vision": "vision_embedding",
        }
        inputs: dict[str, Tensor] = {}
        for modality, key in required.items():
            value = batch.get(key)
            if not isinstance(value, Tensor):
                raise KeyError(f"MP-SID batch must contain tensor {key!r}")
            value = value.to(device)
            if value.dim() != 2 or value.shape[1] != self.raw_dims[modality]:
                raise ValueError(
                    f"{key} shape should be (B, {self.raw_dims[modality]}), got {tuple(value.shape)}"
                )
            inputs[modality] = value

        mask = batch.get("modality_mask")
        if not isinstance(mask, Tensor):
            raise KeyError("MP-SID batch must contain tensor 'modality_mask'")
        mask = mask.to(device=device, dtype=torch.bool)
        batch_size = inputs["collab"].shape[0]
        if tuple(mask.shape) != (batch_size, len(self.modalities)):
            raise ValueError(f"modality_mask shape should be ({batch_size}, 3), got {tuple(mask.shape)}")
        if not mask[:, 0].all().item():
            raise ValueError("MP-SID expects collab modality to be observed for every valid item")
        inputs["modality_mask"] = mask
        return inputs

    def _prepare_raw_blocks(self, inputs: Mapping[str, Tensor], modality_mask: Tensor) -> dict[str, Tensor]:
        raw_blocks: dict[str, Tensor] = {}
        for index, modality in enumerate(self.modalities):
            raw_blocks[modality] = self._apply_modality_mask(inputs[modality], modality_mask[:, index])
        return raw_blocks

    def _encode_latent(self, raw_blocks: Mapping[str, Tensor], modality_mask: Tensor) -> dict[str, Tensor]:
        return {
            modality: self._apply_modality_mask(
                F.normalize(self.raw_encoders[modality](raw_blocks[modality]), p=2, dim=-1),
                modality_mask[:, index],
            )
            for index, modality in enumerate(self.modalities)
        }

    def _quantization_mask(self, modality_mask: Tensor) -> Tensor:
        # Keep the observation mask for encoding and raw supervision. Only the
        # quantizer treats zero-filled missing blocks as ordinary dimensions.
        return modality_mask if self.use_bpd else torch.ones_like(modality_mask, dtype=torch.bool)

    def _quantize(
        self,
        latent: Tensor,
        modality_mask: Tensor,
        compute_q_loss: bool = True,
        soft_indices: bool = False,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        residual = latent
        quantization_mask = self._quantization_mask(modality_mask)
        total_q_loss = torch.zeros(latent.shape[0], device=latent.device, dtype=latent.dtype)
        all_ids: list[Tensor] = []
        all_embs: list[Tensor] = []
        for layer in self.quantize_layers:
            emb, ids, q_loss = layer(
                residual,
                quantization_mask,
                compute_loss=compute_q_loss,
                soft_indices=soft_indices,
                soft_indices_temperature=self.soft_indices_temperature,
            )
            residual = residual - emb.detach()
            total_q_loss = total_q_loss + q_loss
            all_ids.append(ids)
            all_embs.append(emb)
        return sum(all_embs), total_q_loss, all_ids

    def _raw_recon_loss(
        self,
        reconstructions: Mapping[str, Tensor],
        inputs: Mapping[str, Tensor],
        modality_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        losses: dict[str, Tensor] = {}
        active_losses: list[Tensor] = []
        for index, modality in enumerate(self.modalities):
            observed = modality_mask[:, index]
            if observed.any().item():
                diff = F.mse_loss(
                    reconstructions[modality][observed],
                    inputs[modality][observed],
                    reduction="mean",
                )
                # Mean MSE divides by raw dimension; use collab as the reference.
                diff = diff * (self.raw_dims[modality] / self.raw_dims["collab"])
            else:
                diff = reconstructions[modality].new_zeros(())
            losses[modality] = diff
            if observed.any().item():
                active_losses.append(diff)
        if not active_losses:
            raise ValueError("MP-SID batch has no observed modality blocks")
        return torch.stack(active_losses).mean(), losses

    def _split_blocks(self, x: Tensor) -> dict[str, Tensor]:
        chunks = torch.split(x, self.latent_block_dim, dim=-1)
        if len(chunks) != len(self.modalities):
            raise ValueError(f"expected {len(self.modalities)} latent blocks, got {len(chunks)}")
        return {modality: chunks[index] for index, modality in enumerate(self.modalities)}

    @staticmethod
    def _apply_modality_mask(block: Tensor, observed: Tensor) -> Tensor:
        return block * observed.to(device=block.device, dtype=block.dtype).unsqueeze(-1)

    @torch.no_grad()
    def kmeans_init(self, batch: Mapping[str, Tensor], logger: object | None = None) -> None:
        """Initialize residual codebooks with availability-aware masked K-means."""
        if self._kmeans_initted:
            return
        self._kmeans_initted = True

        was_training = self.training
        self.eval()
        try:
            kmeans_n_init = max(1, int(getattr(self.config.model, "kmeans_n_init", 10)))
            kmeans_init_max_iter = max(1, int(getattr(self.config.model, "kmeans_init_max_iter", 50)))
            kmeans_center_init = str(getattr(self.config.model, "kmeans_center_init", "global_mean")).lower()
            train_config = getattr(self.config, "train", None)
            random_state = int(getattr(train_config, "seed", 19))
            total_start = time.perf_counter()

            inputs = self._batch_inputs(batch)
            modality_mask = inputs["modality_mask"]
            raw_blocks = self._prepare_raw_blocks(inputs, modality_mask)
            latent_blocks = self._encode_latent(raw_blocks, modality_mask)
            residual = torch.cat([latent_blocks[modality] for modality in self.modalities], dim=-1).clone()

            _log_if_available(
                logger,
                "MP-SID masked K-means initialization: "
                f"items={int(residual.shape[0])}, "
                f"latent_dim={int(residual.shape[1])}, "
                f"layers={self.n_layers}, codebook_size={self.codebook_size}, "
                f"n_init={kmeans_n_init}, max_iter={kmeans_init_max_iter}, "
                f"center_init={kmeans_center_init}, "
                f"device={residual.device}",
            )

            for layer_idx, qlayer in enumerate(self.quantize_layers):
                layer_num = layer_idx + 1
                _synchronize_if_cuda(residual)
                layer_start = time.perf_counter()
                use_sinkhorn = qlayer.assignment_mode == "sinkhorn"
                epsilon = max(qlayer.sk_epsilon, 1.0e-3)
                _log_if_available(
                    logger,
                    "MP-SID masked K-means "
                    f"layer {layer_num}/{self.n_layers} start: "
                    f"residual_shape={tuple(residual.shape)}, "
                    f"use_sinkhorn={use_sinkhorn}, "
                    f"epsilon={epsilon:.6g}, "
                    f"sk_iters={qlayer.sk_iters}",
                )
                centers, ids = _fit_masked_sinkhorn_balanced_kmeans(
                    residual,
                    self._quantization_mask(modality_mask),
                    self.codebook_size,
                    n_modalities=len(self.modalities),
                    block_dim=self.latent_block_dim,
                    n_init=kmeans_n_init,
                    max_iter=kmeans_init_max_iter,
                    random_state=random_state,
                    epsilon=epsilon,
                    sinkhorn_iterations=qlayer.sk_iters,
                    use_sinkhorn=use_sinkhorn,
                    center_init=kmeans_center_init,
                )
                centers = centers.to(dtype=qlayer.codebook.weight.dtype, device=qlayer.codebook.weight.device)
                qlayer.codebook.weight.data.copy_(centers)

                quantized = centers[ids]
                residual = residual - quantized
                _synchronize_if_cuda(residual)
                layer_elapsed = time.perf_counter() - layer_start
                unique_codes = int(torch.unique(ids).numel())
                _log_if_available(
                    logger,
                    "MP-SID masked K-means "
                    f"layer {layer_num}/{self.n_layers} complete: "
                    f"used_codes={unique_codes}/{self.codebook_size}, "
                    f"elapsed={layer_elapsed:.2f}s",
                )

            _synchronize_if_cuda(residual)
            total_elapsed = time.perf_counter() - total_start
            _log_if_available(logger, f"MP-SID masked K-means initialization complete: elapsed={total_elapsed:.2f}s")
        finally:
            self.train(was_training)

    @torch.no_grad()
    def get_semantic_ids(self, batch: Mapping[str, Tensor]) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            inputs = self._batch_inputs(batch)
            modality_mask = inputs["modality_mask"]
            raw_blocks = self._prepare_raw_blocks(inputs, modality_mask)
            latent_blocks = self._encode_latent(raw_blocks, modality_mask)
            residual = torch.cat([latent_blocks[modality] for modality in self.modalities], dim=-1)
            all_ids: list[Tensor] = []
            for layer in self.quantize_layers:
                ids = layer.assign_argmin(residual, self._quantization_mask(modality_mask))
                quantized = layer.get_codebook()[ids]
                residual = residual - quantized
                all_ids.append(ids)
            return torch.stack(all_ids, dim=-1)
        finally:
            self.train(was_training)

    def get_codebook_weights(self) -> list[Tensor]:
        return [layer.get_codebook().detach().clone() for layer in self.quantize_layers]
