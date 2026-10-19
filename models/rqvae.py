"""Residual quantization layers, MLP components, and initialization helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



@dataclass
class RQVAEOutput:
    loss: Tensor
    recon_loss: Tensor
    quantize_loss: Tensor
    x_hat: Tensor
    all_ids: list[Tensor]
    diversity_loss: Tensor | None = None
    diversity_weighted_loss: Tensor | None = None
    cf_loss: Tensor | None = None
    cf_weighted_loss: Tensor | None = None



def _build_hidden_dims(input_dim: int, output_dim: int) -> list[int]:
    """Build hidden dimensions by successive halving before the output projection."""
    dims = [input_dim]
    d = input_dim
    while d // 2 > output_dim:
        d = d // 2
        dims.append(d)
    dims.append(output_dim)
    return dims


def _activation_layer(name: str | None) -> nn.Module | None:
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU()
    if normalized in {"none", "identity"}:
        return None
    raise ValueError(f"Unknown activation: '{name}'")


def _normalize_hidden_dims(hidden_dims: object) -> list[int] | None:
    if hidden_dims is None:
        return None
    if not isinstance(hidden_dims, (list, tuple)):
        raise TypeError(f"model.hidden_dims should be a list or null, got {type(hidden_dims).__name__}")
    dims = [int(dim) for dim in hidden_dims]
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"model.hidden_dims must contain positive integers, got {hidden_dims!r}")
    return dims


def _normalize_enc_dec_type(enc_dec_type: object) -> str:
    normalized = str(enc_dec_type or "MPL").lower()
    if normalized == "mpl":
        return "MPL"
    if normalized == "linear":
        return "Linear"
    raise ValueError(f"model.enc_dec_type should be 'MPL' or 'Linear', got {enc_dec_type!r}")


def _normalize_sk_epsilons(value: object, n_layers: int) -> list[float]:
    if value is None:
        return [0.0 for _ in range(n_layers)]
    if isinstance(value, (int, float)):
        epsilons = [float(value) for _ in range(n_layers)]
    elif isinstance(value, (list, tuple)):
        epsilons = [float(epsilon) for epsilon in value]
        if len(epsilons) == 1:
            epsilons = epsilons * n_layers
    else:
        raise TypeError(f"model.sk_epsilons should be a list, number, or null, got {type(value).__name__}")
    if len(epsilons) != n_layers:
        raise ValueError(f"model.sk_epsilons should have {n_layers} values, got {len(epsilons)}")
    if any(epsilon < 0.0 for epsilon in epsilons):
        raise ValueError(f"model.sk_epsilons must be non-negative, got {value!r}")
    return epsilons


def _normalize_modalities(value: object, field_name: str) -> set[str]:
    valid_modalities = {"text", "vision", "collab", "fused"}
    if value is None:
        return {"text", "vision", "fused"}
    if isinstance(value, str):
        modalities = [value]
    elif isinstance(value, (list, tuple, set)):
        modalities = [str(modality) for modality in value]
    else:
        raise TypeError(f"{field_name} should be a list, string, or null, got {type(value).__name__}")

    normalized = {modality.lower() for modality in modalities}
    unknown = sorted(normalized - valid_modalities)
    if unknown:
        raise ValueError(f"{field_name} contains unknown modalities: {unknown}")
    return normalized


def _fit_constrained_kmeans(
    data: Tensor,
    n_clusters: int,
    *,
    n_init: int,
    max_iter: int,
    random_state: int | None,
    size_min_cap: int,
    size_max: int | None = None,
    size_max_multiplier: int = 4,
) -> tuple[Tensor, Tensor]:
    from k_means_constrained import KMeansConstrained

    n_samples = int(data.shape[0])
    n_clusters = int(n_clusters)
    if n_clusters <= 0:
        raise ValueError(f"n_clusters should be positive, got {n_clusters}")
    if n_samples < n_clusters:
        raise ValueError(f"constrained KMeans requires n_samples >= n_clusters, got {n_samples} < {n_clusters}")

    size_min = min(n_samples // (n_clusters * 2), int(size_min_cap))
    size_min = max(1, size_min)
    requested_size_max = size_min * int(size_max_multiplier) if size_max is None else int(size_max)
    size_max = max(size_min, requested_size_max, math.ceil(n_samples / n_clusters))

    km = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        max_iter=int(max_iter),
        n_init=max(1, int(n_init)),
        random_state=random_state,
        n_jobs=10,
        verbose=False,
    )
    km.fit(data.detach().float().cpu().numpy())

    centers = torch.from_numpy(km.cluster_centers_).to(device=data.device, dtype=data.dtype)
    labels = torch.as_tensor(km.labels_, dtype=torch.long, device=data.device)
    return centers, labels


def _squared_l2_distance(x: Tensor, centers: Tensor) -> Tensor:
    return (
        (x ** 2).sum(dim=-1, keepdim=True)
        + (centers ** 2).sum(dim=-1)
        - 2.0 * x @ centers.t()
    )


def _log_if_available(logger: object | None, message: str) -> None:
    if logger is not None and hasattr(logger, "info"):
        logger.info(message)


def _synchronize_if_cuda(tensor: Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


@torch.no_grad()
def _fit_sinkhorn_balanced_kmeans(
    data: Tensor,
    n_clusters: int,
    *,
    n_init: int,
    max_iter: int,
    random_state: int | None,
    epsilon: float,
    sinkhorn_iterations: int,
    tolerance: float = 1.0e-4,
) -> tuple[Tensor, Tensor]:
    """Fit an approximate balanced KMeans on the input tensor's device."""
    n_samples = int(data.shape[0])
    n_clusters = int(n_clusters)
    if n_clusters <= 0:
        raise ValueError(f"n_clusters should be positive, got {n_clusters}")
    if n_samples < n_clusters:
        raise ValueError(f"Sinkhorn-balanced KMeans requires n_samples >= n_clusters, got {n_samples} < {n_clusters}")

    work = data.detach().float()
    n_init = max(1, int(n_init))
    max_iter = max(1, int(max_iter))
    sinkhorn_iterations = max(1, int(sinkhorn_iterations))
    epsilon = max(float(epsilon), 1.0e-12)
    tolerance = max(0.0, float(tolerance))
    best_centers: Tensor | None = None
    best_labels: Tensor | None = None
    best_score = float("inf")

    for init_idx in range(n_init):
        if random_state is None:
            init_indices = torch.randperm(n_samples, device="cpu")[:n_clusters].to(device=work.device)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_state) + init_idx)
            init_indices = torch.randperm(n_samples, generator=generator, device="cpu")[:n_clusters].to(device=work.device)
        centers = work.index_select(0, init_indices).clone()

        for _ in range(max_iter):
            dist = _squared_l2_distance(work, centers)
            centered = QuantizeLayer._center_distance_for_constraint(dist)
            assignment = QuantizeLayer._sinkhorn_algorithm(centered, epsilon, sinkhorn_iterations)
            if torch.isnan(assignment).any() or torch.isinf(assignment).any():
                labels = dist.argmin(dim=-1)
                assignment = F.one_hot(labels, num_classes=n_clusters).to(device=work.device, dtype=work.dtype)

            cluster_mass = assignment.sum(dim=0).clamp_min(1.0e-12).unsqueeze(1)
            next_centers = assignment.t() @ work / cluster_mass
            if torch.isnan(next_centers).any() or torch.isinf(next_centers).any():
                break

            center_shift = (next_centers - centers).pow(2).sum().sqrt().item()
            centers = next_centers
            if center_shift <= tolerance:
                break

        final_dist = _squared_l2_distance(work, centers)
        final_centered = QuantizeLayer._center_distance_for_constraint(final_dist)
        final_assignment = QuantizeLayer._sinkhorn_algorithm(final_centered, epsilon, sinkhorn_iterations)
        if torch.isnan(final_assignment).any() or torch.isinf(final_assignment).any():
            final_labels = final_dist.argmin(dim=-1)
            final_score = final_dist.gather(1, final_labels.unsqueeze(1)).sum().item()
        else:
            final_labels = final_assignment.argmax(dim=-1).long()
            final_score = (final_assignment * final_dist).sum().item()

        if final_score < best_score:
            best_score = final_score
            best_centers = centers.clone()
            best_labels = final_labels.clone()

    if best_centers is None or best_labels is None:
        raise RuntimeError("Sinkhorn-balanced KMeans failed to produce centers")
    return best_centers.to(dtype=data.dtype), best_labels


def _encoder_dims(
    input_dim: int,
    embed_dim: int,
    hidden_dims: list[int] | None,
    enc_dec_type: str = "MPL",
) -> list[int]:
    if enc_dec_type == "Linear":
        return [input_dim, embed_dim]
    if hidden_dims is None:
        return _build_hidden_dims(input_dim, embed_dim)
    return [input_dim] + hidden_dims + [embed_dim]


def _decoder_dims(
    embed_dim: int,
    output_dim: int,
    hidden_dims: list[int] | None,
    enc_dec_type: str = "MPL",
) -> list[int]:
    if enc_dec_type == "Linear":
        return [embed_dim, output_dim]
    if hidden_dims is None:
        dims = _build_hidden_dims(output_dim, embed_dim)
        dims.reverse()
        return dims
    return [embed_dim] + list(reversed(hidden_dims)) + [output_dim]


class RQMLP(nn.Sequential):
    """LETTER-style tokenizer MLP."""

    def __init__(
        self,
        dims: list[int],
        dropout_prob: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "relu",
    ):
        if len(dims) < 2:
            raise ValueError(f"MLP dims should contain at least input/output dims, got {dims}")
        layers = self._build_letter(dims, dropout_prob, use_batch_norm, activation)
        super().__init__(*layers)
        self.apply(self._init_letter_weights)

    @staticmethod
    def _build_letter(
        dims: list[int],
        dropout_prob: float,
        use_batch_norm: bool,
        activation: str,
    ) -> list[nn.Module]:
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Dropout(p=float(dropout_prob)))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(num_features=dims[i + 1]))
            if i < len(dims) - 2:
                activation_func = _activation_layer(activation)
                if activation_func is not None:
                    layers.append(activation_func)
        return layers

    @staticmethod
    def _init_letter_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


# ==================== Encoder ====================
class Encoder(nn.Module):
    """MLP encoder from input_dim to embed_dim."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_prob: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "relu",
        enc_dec_type: str = "MPL",
    ):
        super().__init__()
        enc_dec_type = _normalize_enc_dec_type(enc_dec_type)
        dims = _encoder_dims(input_dim, embed_dim, hidden_dims, enc_dec_type)
        self.net = RQMLP(dims, dropout_prob, use_batch_norm, activation)

    def forward(self, x: Tensor) -> Tensor:
        """ x: (B, input_dim) → (B, embed_dim) """
        return self.net(x)


# ==================== Decoder ====================
class Decoder(nn.Module):
    """MLP decoder with an L2-normalized output."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_prob: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "relu",
        enc_dec_type: str = "MPL",
    ):
        super().__init__()
        enc_dec_type = _normalize_enc_dec_type(enc_dec_type)
        dims = _decoder_dims(embed_dim, output_dim, hidden_dims, enc_dec_type)
        self.net = RQMLP(dims, dropout_prob, use_batch_norm, activation)

    def forward(self, z_q: Tensor) -> Tensor:
        """Decode quantized embeddings to L2-normalized output vectors."""
        out = self.net(z_q)
        return F.normalize(out, dim=-1)


# ==================== QuantizeLayer ====================
class QuantizeLayer(nn.Module):
    """Vector quantizer with a straight-through gradient estimator."""

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        commitment_weight: float = 0.25,
        assignment_mode: str = "argmin",
        sk_epsilon: float = 0.0,
        sk_iters: int = 50,
        use_sinkhorn_train: bool = True,
        use_sinkhorn_eval: bool = False,
        diversity_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.commitment_weight = commitment_weight
        self.assignment_mode = str(assignment_mode).lower()
        if self.assignment_mode not in {"argmin", "sinkhorn"}:
            raise ValueError(f"model.assignment_mode should be 'argmin' or 'sinkhorn', got {assignment_mode!r}")
        self.sk_epsilon = float(sk_epsilon)
        if self.sk_epsilon < 0.0:
            raise ValueError(f"model.sk_epsilons values must be non-negative, got {self.sk_epsilon}")
        self.sk_iters = max(1, int(sk_iters))
        self.use_sinkhorn_train = bool(use_sinkhorn_train)
        self.use_sinkhorn_eval = bool(use_sinkhorn_eval)
        self.diversity_loss_weight = max(0.0, float(diversity_loss_weight))
        self.register_buffer(
            "codebook_group_labels",
            torch.full((codebook_size,), -1, dtype=torch.long),
            persistent=False,
        )
        self._last_diversity_loss: Tensor | None = None


        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

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

    def _assign_codes_for_kmeans_init(self, dist: Tensor) -> Tensor:
        use_sinkhorn = self.assignment_mode == "sinkhorn" and self.sk_epsilon > 0.0
        return self._assign_codes_with_sinkhorn(dist, use_sinkhorn)

    @torch.no_grad()
    def set_codebook_group_labels(self, labels: Tensor) -> None:
        labels = labels.to(device=self.codebook.weight.device, dtype=torch.long)
        expected = (self.codebook_size,)
        if tuple(labels.shape) != expected:
            raise ValueError(f"codebook group labels should have shape {expected}, got {tuple(labels.shape)}")
        self.codebook_group_labels.copy_(labels)

    @torch.no_grad()
    def clear_codebook_group_labels(self) -> None:
        self.codebook_group_labels.fill_(-1)

    def _compute_diversity_loss(self, quantized: Tensor, ids: Tensor, cb: Tensor) -> Tensor:
        if not self.training or self.diversity_loss_weight <= 0.0:
            return quantized.new_zeros(())
        group_labels = self.codebook_group_labels.to(device=ids.device)
        assigned_groups = group_labels[ids]
        valid_groups = assigned_groups >= 0
        if not valid_groups.any().item():
            return quantized.new_zeros(())

        code_ids = torch.arange(self.codebook_size, device=ids.device)
        same_group = group_labels.unsqueeze(0).eq(assigned_groups.unsqueeze(1))
        not_self = code_ids.unsqueeze(0).ne(ids.unsqueeze(1))
        positive_mask = same_group & not_self & valid_groups.unsqueeze(1)
        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any().item():
            return quantized.new_zeros(())

        positive_weights = positive_mask[valid_rows].float()
        positive_ids = torch.multinomial(positive_weights, num_samples=1).squeeze(1)
        logits = quantized[valid_rows] @ cb.t()
        logits = logits.clone()
        self_ids = ids[valid_rows]
        row_ids = torch.arange(logits.shape[0], device=logits.device)
        logits[row_ids, self_ids] = -1.0e12
        return F.cross_entropy(logits, positive_ids)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize input vectors and return embeddings, code indices, and per-sample losses."""

        cb = self.get_codebook()  # (CS, D)


        dist = (
            (x ** 2).sum(dim=-1, keepdim=True)            # (B, 1)
            + (cb ** 2).sum(dim=-1, keepdim=False)         # (CS,)
            - 2.0 * x @ cb.t()                            # (B, CS)
        )


        ids = self._assign_codes(dist)      # (B,)
        hard_emb = cb[ids]                  # (B, D)

        if not self.training:
            emb = hard_emb
        else:
            emb = x + (hard_emb - x).detach()


        q_loss = self._compute_loss(x, hard_emb)  # (B,)
        self._last_diversity_loss = self._compute_diversity_loss(hard_emb, ids, cb)

        return emb, ids, q_loss

    def _compute_loss(self, x: Tensor, emb: Tensor) -> Tensor:
        """Compute per-sample codebook and commitment losses with stop-gradient targets."""
        commit_loss = F.mse_loss(x, emb.detach(), reduction="none").sum(dim=-1)  # (B,)
        codebook_loss = F.mse_loss(x.detach(), emb, reduction="none").sum(dim=-1)  # (B,)
        return codebook_loss + self.commitment_weight * commit_loss



class RqVae(nn.Module):
    """Residual quantization autoencoder for item embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        modality = config.data.modality


        input_dim_map = {
            "text": config.model.text_input_dim,
            "vision": config.model.visual_input_dim,
            "collab": config.model.collab_input_dim,
            "fused": config.model.fused_input_dim,
        }
        if modality not in input_dim_map:
            raise ValueError(f"Unknown modality: '{modality}'")
        self.input_dim = input_dim_map[modality]
        self.embed_dim = config.model.embed_dim
        self.codebook_size = config.model.codebook_size
        self.n_layers = config.model.n_layers
        self.modality = modality
        self.quant_loss_weight = float(getattr(config.model, "quant_loss_weight", 1.0))
        configured_diversity_loss_weight = max(
            0.0,
            float(getattr(config.model, "diversity_loss_weight", 0.0)),
        )
        self.diversity_loss_weight = configured_diversity_loss_weight
        self.cf_loss_weight = max(0.0, float(getattr(config.model, "cf_loss_weight", 0.0)))
        self.cf_loss_source = str(getattr(config.model, "cf_loss_source", "collab_emb"))
        self.cf_loss_modalities = _normalize_modalities(
            getattr(config.model, "cf_loss_modalities", None),
            "model.cf_loss_modalities",
        )
        self.cf_loss_temperature = max(1.0e-6, float(getattr(config.model, "cf_loss_temperature", 1.0)))
        self.cf_loss_enabled = self.cf_loss_weight > 0.0 and self.modality in self.cf_loss_modalities
        self.cf_target_dim = int(getattr(config.model, "collab_input_dim", self.embed_dim))
        self.codebook_group_clusters = max(
            1,
            int(getattr(
                config.model,
                "codebook_group_clusters",
                getattr(config.model, "diversity_num_groups", 10),
            )),
        )
        hidden_dims = _normalize_hidden_dims(getattr(config.model, "hidden_dims", None))
        self.enc_dec_type = _normalize_enc_dec_type(getattr(config.model, "enc_dec_type", "MPL"))
        self.dropout_prob = float(getattr(config.model, "dropout_prob", 0.0))
        self.use_batch_norm = bool(getattr(config.model, "use_batch_norm", False))
        self.activation = str(getattr(config.model, "activation", "relu"))
        assignment_mode = str(getattr(config.model, "assignment_mode", "argmin")).lower()
        sk_epsilons = _normalize_sk_epsilons(getattr(config.model, "sk_epsilons", None), self.n_layers)
        sk_iters = int(getattr(config.model, "sk_iters", 50))
        use_sinkhorn_train = bool(getattr(config.model, "use_sinkhorn_train", True))
        use_sinkhorn_eval = bool(getattr(config.model, "use_sinkhorn_eval", False))

        # ---- Encoder / Decoder ----
        self.encoder = Encoder(
            self.input_dim, self.embed_dim,
            hidden_dims=hidden_dims,
            dropout_prob=self.dropout_prob,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
            enc_dec_type=self.enc_dec_type,
        )
        self.decoder = Decoder(
            self.embed_dim, self.input_dim,
            hidden_dims=hidden_dims,
            dropout_prob=self.dropout_prob,
            use_batch_norm=self.use_batch_norm,
            activation=self.activation,
            enc_dec_type=self.enc_dec_type,
        )


        if self.cf_loss_enabled and self.cf_target_dim != self.embed_dim:
            self.cf_target_projector: nn.Module | None = nn.Linear(self.cf_target_dim, self.embed_dim, bias=False)
        else:
            self.cf_target_projector = None

        # ---- QuantizeLayers ----
        commitment_weight = config.model.commitment_weight
        self.quantize_layers = nn.ModuleList([
            QuantizeLayer(
                self.codebook_size, self.embed_dim,
                commitment_weight,
                assignment_mode=assignment_mode,
                sk_epsilon=sk_epsilons[layer_idx],
                sk_iters=sk_iters,
                use_sinkhorn_train=use_sinkhorn_train,
                use_sinkhorn_eval=use_sinkhorn_eval,
                diversity_loss_weight=self.diversity_loss_weight,
            )
            for layer_idx in range(self.n_layers)
        ])


        self._kmeans_initted = False

    def _quantize(self, z: Tensor) -> tuple[Tensor, Tensor, list[Tensor], Tensor]:
        """Apply residual vector quantization."""
        residual = z
        total_q_loss = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        total_diversity_loss = z.new_zeros(())
        all_ids: list[Tensor] = []
        all_embs: list[Tensor] = []

        for layer in self.quantize_layers:
            emb, ids, q_loss = layer(residual)
            residual = residual - emb.detach()
            total_q_loss = total_q_loss + q_loss
            if layer._last_diversity_loss is not None:
                total_diversity_loss = total_diversity_loss + layer._last_diversity_loss
            all_ids.append(ids)
            all_embs.append(emb)

        z_q = sum(all_embs)
        return z_q, total_q_loss, all_ids, total_diversity_loss

    def _compute_cf_loss(self, z_q: Tensor, cf_embedding: Tensor | None) -> Tensor:
        if not self.cf_loss_enabled or cf_embedding is None:
            return z_q.new_zeros(())
        if cf_embedding.dim() != 2:
            raise ValueError(f"cf_embedding should be a 2D Tensor, got shape {tuple(cf_embedding.shape)}")
        if cf_embedding.shape[0] != z_q.shape[0]:
            raise ValueError(
                f"cf_embedding batch size should match z_q: {cf_embedding.shape[0]} vs {z_q.shape[0]}"
            )
        if cf_embedding.shape[1] != self.cf_target_dim:
            raise ValueError(
                f"cf_embedding dim should match model.collab_input_dim={self.cf_target_dim}, "
                f"got {cf_embedding.shape[1]}"
            )

        target = cf_embedding.to(device=z_q.device, dtype=z_q.dtype)
        if self.cf_target_projector is not None:
            target = self.cf_target_projector(target)

        quantized_rep = F.normalize(z_q, dim=-1)
        target_rep = F.normalize(target, dim=-1)
        similarities = quantized_rep @ target_rep.t()
        similarities = similarities / self.cf_loss_temperature
        labels = torch.arange(z_q.shape[0], dtype=torch.long, device=z_q.device)
        return F.cross_entropy(similarities, labels)

    def forward(
        self,
        x: Tensor,
        cf_embedding: Tensor | None = None,
    ) -> RQVAEOutput:
        """Encode and quantize normalized input embeddings; return reconstruction and quantization losses with semantic codes."""

        # ---- Encoder ----
        z = self.encoder(x)  # (B, embed_dim)


        z_q, total_q_loss, all_ids, total_diversity_loss = self._quantize(z)

        # ---- Decoder ----
        x_hat = self.decoder(z_q)  # (B, input_dim)


        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        quantize_loss = total_q_loss.mean() / max(self.n_layers * self.embed_dim, 1)
        diversity_loss = total_diversity_loss / max(self.n_layers, 1)
        diversity_weighted_loss = self.diversity_loss_weight * diversity_loss
        cf_loss = self._compute_cf_loss(z_q, cf_embedding)
        cf_weighted_loss = self.cf_loss_weight * cf_loss
        loss = recon_loss + self.quant_loss_weight * (quantize_loss + diversity_weighted_loss) + cf_weighted_loss

        return RQVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            quantize_loss=quantize_loss,
            x_hat=x_hat,
            all_ids=all_ids,
            diversity_loss=diversity_loss,
            diversity_weighted_loss=diversity_weighted_loss,
            cf_loss=cf_loss,
            cf_weighted_loss=cf_weighted_loss,
        )


    @torch.no_grad()
    def kmeans_init(self, all_embeddings: Tensor, logger: object | None = None) -> None:
        """Initialize codebooks layer by layer from encoded embeddings and successive residuals."""

        if self._kmeans_initted:
            return
        self._kmeans_initted = True

        was_training = self.training
        self.eval()
        try:
            kmeans_n_init = max(1, int(getattr(self.config.model, "kmeans_n_init", 10)))
            kmeans_init_max_iter = max(1, int(getattr(self.config.model, "kmeans_init_max_iter", 50)))
            random_state = int(getattr(self.config.train, "seed", 19))
            total_start = time.perf_counter()
            _log_if_available(
                logger,
                "K-means initialization: "
                f"items={int(all_embeddings.shape[0])}, "
                f"input_dim={int(all_embeddings.shape[1])}, "
                f"layers={self.n_layers}, codebook_size={self.codebook_size}, "
                f"n_init={kmeans_n_init}, max_iter={kmeans_init_max_iter}, "
                f"device={all_embeddings.device}",
            )


            z_all = self.encoder(all_embeddings)  # (N, embed_dim)
            residual = z_all.clone()

            for layer_idx, qlayer in enumerate(self.quantize_layers):
                layer_num = layer_idx + 1
                _synchronize_if_cuda(residual)
                layer_start = time.perf_counter()
                _log_if_available(
                    logger,
                    "K-means initialization "
                    f"layer {layer_num}/{self.n_layers} start: "
                    f"residual_shape={tuple(residual.shape)}, "
                    f"epsilon={max(qlayer.sk_epsilon, 1.0e-3):.6g}, "
                    f"sk_iters={qlayer.sk_iters}",
                )
                centers, ids = _fit_sinkhorn_balanced_kmeans(
                    residual,
                    self.codebook_size,
                    n_init=kmeans_n_init,
                    max_iter=kmeans_init_max_iter,
                    random_state=random_state,
                    epsilon=max(qlayer.sk_epsilon, 1.0e-3),
                    sinkhorn_iterations=qlayer.sk_iters,
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
                    "K-means initialization "
                    f"layer {layer_num}/{self.n_layers} complete: "
                    f"used_codes={unique_codes}/{self.codebook_size}, "
                    f"elapsed={layer_elapsed:.2f}s",
                )
            _synchronize_if_cuda(residual)
            total_elapsed = time.perf_counter() - total_start
            _log_if_available(logger, f"K-means initialization complete: elapsed={total_elapsed:.2f}s")
        finally:
            self.train(was_training)

    @torch.no_grad()
    def refresh_codebook_groups(self, random_state: int | None = None) -> list[int]:
        """Cluster each layer codebook to provide positive groups for diversity loss."""
        if self.diversity_loss_weight <= 0.0:
            for qlayer in self.quantize_layers:
                qlayer.clear_codebook_group_labels()
            return [0 for _ in self.quantize_layers]

        n_clusters = min(self.codebook_group_clusters, self.codebook_size)
        if n_clusters < 2:
            for qlayer in self.quantize_layers:
                qlayer.clear_codebook_group_labels()
            return [0 for _ in self.quantize_layers]

        kmeans_n_init = max(1, int(getattr(self.config.model, "kmeans_n_init", 10)))
        if random_state is None:
            random_state = int(getattr(self.config.train, "seed", 19))

        group_counts: list[int] = []
        for qlayer in self.quantize_layers:
            _, labels = _fit_constrained_kmeans(
                qlayer.get_codebook().detach().float(),
                n_clusters,
                n_init=kmeans_n_init,
                max_iter=100,
                random_state=random_state,
                size_min_cap=10,
                size_max=n_clusters * 6,
            )
            qlayer.set_codebook_group_labels(labels)
            group_counts.append(int(torch.unique(labels).numel()))
        return group_counts


    @torch.no_grad()
    def get_semantic_ids(self, x: Tensor) -> Tensor:
        """Return zero-based semantic code indices with shape [B, n_layers]."""

        z = self.encoder(x)
        residual = z
        all_ids = []

        for qlayer in self.quantize_layers:
            cb = qlayer.get_codebook()
            dist = (
                (residual ** 2).sum(dim=-1, keepdim=True)
                + (cb ** 2).sum(dim=-1)
                - 2.0 * residual @ cb.t()
            )
            ids = dist.argmin(dim=-1)  # (B,)
            quantized = cb[ids]
            residual = residual - quantized
            all_ids.append(ids)

        return torch.stack(all_ids, dim=-1)  # (B, n_layers)

    def get_codebook_weights(self) -> list[Tensor]:
        """Return the list of codebook weight tensors for Stage 2."""

        return [qlayer.get_codebook().detach().clone() for qlayer in self.quantize_layers]
