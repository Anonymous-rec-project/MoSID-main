"""RQ-VAE training, evaluation, early stopping, and checkpoint management."""
from __future__ import annotations

import math
import os
import shutil
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

from utils.io import save_checkpoint
from utils.logger import Logger, format_horizontal_metrics_table
from utils.data import embedding_filename
from configs.config import DotDict


class RQVAETrainer:
    """Manage RQ-VAE optimization, full-catalog evaluation, early stopping, and checkpoints."""

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        full_dataset: Dataset,
        config: DotDict,
        logger: Logger,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.full_dataset = full_dataset
        self.config = config
        self.logger = logger
        self.device = device


        self.max_epochs = config.train.max_epochs
        self.eval_every = config.train.eval_every
        self.patience = config.train.patience
        self.es_warmup = max(0, int(getattr(config.train, "es_warmup", 10)))
        self.es_metric = str(getattr(config.train, "es_metric", "collision")).lower()
        allowed_es_metrics = {"collision", "prefix_semantic"}
        if self.es_metric not in allowed_es_metrics:
            raise ValueError(
                f"train.es_metric should be one of {sorted(allowed_es_metrics)!r}, got {self.es_metric!r}"
            )
        if self.es_metric == "prefix_semantic" and str(getattr(config.data, "modality", "")) != "unified":
            raise ValueError("train.es_metric='prefix_semantic' is supported only for unified Stage 1")

        self.semantic_collision_rate_max = float(
            getattr(config.train, "semantic_collision_rate_max", 0.01)
        )
        self.semantic_max_leaf = int(getattr(config.train, "semantic_max_leaf", 3))
        raw_prefix_weights = getattr(config.train, "prefix_semantic_weights", [2.0, 1.0])
        if not isinstance(raw_prefix_weights, (list, tuple)) or not raw_prefix_weights:
            raise ValueError("train.prefix_semantic_weights must be a non-empty list")
        self.prefix_semantic_weights = tuple(float(weight) for weight in raw_prefix_weights)
        self.prefix_semantic_tolerance = float(
            getattr(config.train, "prefix_semantic_tolerance", 1.0e-4)
        )
        if self.semantic_collision_rate_max < 0.0 or self.semantic_collision_rate_max > 1.0:
            raise ValueError("train.semantic_collision_rate_max must be in [0, 1]")
        if self.semantic_max_leaf <= 0:
            raise ValueError("train.semantic_max_leaf must be positive")
        if any(weight <= 0.0 for weight in self.prefix_semantic_weights):
            raise ValueError("train.prefix_semantic_weights values must be positive")
        if len(self.prefix_semantic_weights) >= int(config.model.n_layers):
            raise ValueError(
                "train.prefix_semantic_weights must define proper SID prefixes only; "
                f"got {len(self.prefix_semantic_weights)} weights for {int(config.model.n_layers)} layers"
            )
        if self.prefix_semantic_tolerance < 0.0:
            raise ValueError("train.prefix_semantic_tolerance must be non-negative")

        self.diversity_loss_weight = max(
            0.0,
            float(getattr(model, "diversity_loss_weight", 0.0)),
        )
        self.codebook_group_refresh_every = max(
            0,
            int(getattr(config.model, "codebook_group_refresh_every", 1)),
        )
        self.cf_loss_enabled = bool(getattr(model, "cf_loss_enabled", False))
        self.cf_loss_weight = max(0.0, float(getattr(config.model, "cf_loss_weight", 0.0)))
        self.cf_embedding = self._load_cf_embedding() if self.cf_loss_enabled else None


        self.use_tb = config.train.tensorboard


        self.run_dir = str(logger.log_dir)
        self.best_ckpt_path = os.path.join(self.run_dir, "best.pt")
        self.best_collision_ckpt_path = os.path.join(self.run_dir, "best_collision.pt")
        self.best_semantic_ckpt_path = os.path.join(self.run_dir, "best_semantic.pt")
        self.fallback_ckpt_path = os.path.join(self.run_dir, "fallback.pt")

    def _load_cf_embedding(self) -> torch.Tensor:
        source = str(getattr(self.model, "cf_loss_source", "collab_emb"))
        if source != "collab_emb":
            raise ValueError(f"model.cf_loss_source currently only supports 'collab_emb', got {source!r}")

        path = os.path.join(self.config.data._data_dir, embedding_filename("collab"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"CF loss requires collab embedding file: {path}")

        cf_embedding = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(cf_embedding, torch.Tensor):
            raise TypeError(f"CF embedding file should contain a Tensor, got {type(cf_embedding).__name__}")
        if cf_embedding.dim() != 2:
            raise ValueError(f"CF embedding should be 2D, got shape {tuple(cf_embedding.shape)}")
        if cf_embedding.shape[0] <= 1:
            raise ValueError("CF embedding should include padding row 0 plus at least one item row")

        target_dim = int(getattr(self.model, "cf_target_dim", cf_embedding.shape[1]))
        if cf_embedding.shape[1] != target_dim:
            raise ValueError(
                f"CF embedding dim should match model.collab_input_dim={target_dim}, "
                f"got {cf_embedding.shape[1]}"
            )

        self.logger.info(
            "Enabled CF loss: "
            f"source={source}, weight={self.cf_loss_weight:.6f}, "
            f"target_shape={tuple(cf_embedding.shape)}"
        )
        return cf_embedding.detach().float().cpu()

    def _cf_targets_for_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if not self.cf_loss_enabled:
            return None
        if self.cf_embedding is None:
            raise RuntimeError("CF loss is enabled but cf_embedding is not loaded")
        if "item_id" not in batch:
            raise KeyError("CF loss requires batch['item_id'] to align rows in collab_emb.pt")

        item_ids = torch.as_tensor(batch["item_id"], dtype=torch.long).view(-1)
        if item_ids.numel() == 0:
            return self.cf_embedding.new_empty((0, self.cf_embedding.shape[1])).to(self.device)
        if (item_ids <= 0).any().item():
            bad_ids = item_ids[item_ids <= 0][:5].tolist()
            raise ValueError(f"CF loss should not receive padding/non-positive item_id values: {bad_ids}")
        max_item_id = int(item_ids.max().item())
        if max_item_id >= self.cf_embedding.shape[0]:
            raise IndexError(
                f"batch item_id {max_item_id} is out of range for collab_emb.pt "
                f"with {self.cf_embedding.shape[0]} rows"
            )

        targets = self.cf_embedding.index_select(0, item_ids.cpu())
        return targets.to(self.device, non_blocking=True)

    def train(self) -> None:
        """Run the training loop."""
        if hasattr(self.model, "cross_modal_pred_weight"):
            self.logger.info(
                "Cross-modal prediction (CMP): train/cmp is before weighting; "
                "train/cmp_w = train/cmp * model.cross_modal_pred_weight is the contribution to total loss. "
                "train/pred_t and train/pred_v are unweighted target-modality diagnostics. "
                "Epoch losses are batch means (skipped batches contribute zero); "
                "full_n counts complete-modality samples and skip_n counts skipped batches. "
                "TensorBoard: cross_modal_pred_loss/pred_loss_text/pred_loss_vision are unweighted; "
                "cross_modal_pred_weighted_loss is weighted."
            )
        best_selection_score = float("inf")
        best_checkpoint_recon_loss = float("inf")
        best_checkpoint_selection_score = float("inf")
        patience_counter = 0
        has_best = False
        best_eval_metrics: dict[str, float] | None = None
        best_collision_eval_metrics: dict[str, float] | None = None
        semantic_selection = self.es_metric == "prefix_semantic"

        for epoch in range(self.max_epochs):
            if hasattr(self.model, "set_epoch"):
                self.model.set_epoch(epoch)
            if self._should_refresh_codebook_groups(epoch):
                self._refresh_codebook_groups(epoch)


            train_metrics = self._train_one_epoch(epoch)


            if epoch % self.eval_every != 0:
                self._log_epoch_summary(epoch, train_metrics)

            if self.use_tb:
                self.logger.log_scalar("train/recon_loss", train_metrics["recon_loss"], epoch)
                self.logger.log_scalar("train/quantize_loss", train_metrics["quantize_loss"], epoch)
                self.logger.log_scalar("train/quant_weighted_loss", train_metrics["quant_weighted_loss"], epoch)
                self.logger.log_scalar("train/total_loss", train_metrics["total_loss"], epoch)
                if self.diversity_loss_weight > 0.0:
                    self.logger.log_scalar("train/diversity_loss", train_metrics.get("diversity_loss", 0.0), epoch)
                    self.logger.log_scalar(
                        "train/diversity_weighted_loss",
                        train_metrics.get("diversity_weighted_loss", 0.0),
                        epoch,
                    )
                if self.cf_loss_enabled:
                    self.logger.log_scalar("train/cf_loss", train_metrics.get("cf_loss", 0.0), epoch)
                    self.logger.log_scalar(
                        "train/cf_weighted_loss",
                        train_metrics.get("cf_weighted_loss", 0.0),
                        epoch,
                    )
                if "cross_modal_pred_loss" in train_metrics:
                    self.logger.log_scalar("train/cross_modal_pred_loss", train_metrics.get("cross_modal_pred_loss", 0.0), epoch)
                    self.logger.log_scalar(
                        "train/cross_modal_pred_weighted_loss",
                        train_metrics.get("cross_modal_pred_weighted_loss", 0.0),
                        epoch,
                    )
                    self.logger.log_scalar("train/pred_loss_text", train_metrics.get("pred_loss_text", 0.0), epoch)
                    self.logger.log_scalar("train/pred_loss_vision", train_metrics.get("pred_loss_vision", 0.0), epoch)

            if epoch % self.eval_every == 0:
                eval_metrics = self.evaluate()
                selection_metrics = self._checkpoint_selection_metrics(eval_metrics)
                eval_metrics.update(selection_metrics)
                selection_score = selection_metrics["selection_score"]
                self._log_epoch_summary(epoch, train_metrics, eval_metrics)

                # TensorBoard retains the full diagnostics hidden from the compact console summary.

                if self.use_tb:
                    for k, v in eval_metrics.items():
                        if math.isfinite(float(v)):
                            self.logger.log_scalar(f"eval/{k}", v, epoch)

                checkpoint_allowed = epoch >= self.es_warmup
                if (
                    semantic_selection
                    and checkpoint_allowed
                    and self._is_better_collision_checkpoint_from_metrics(
                        eval_metrics,
                        best_collision_eval_metrics,
                    )
                ):
                    best_collision_eval_metrics = dict(eval_metrics)
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        eval_metrics, self.best_collision_ckpt_path,
                    )
                    self.logger.info(
                        "  [best-collision] saved best_collision.pt "
                        f"(collision={float(eval_metrics['collision_rate']):.6f}, "
                        f"collision_n={int(eval_metrics['collision_count'])}, "
                        f"max_leaf={int(eval_metrics['exact_max_items_per_sid'])})"
                    )

                checkpoint_eligible = checkpoint_allowed and self._checkpoint_is_eligible(eval_metrics)
                if checkpoint_eligible and self._is_better_checkpoint(eval_metrics, best_eval_metrics):
                    best_selection_score = selection_score
                    best_checkpoint_recon_loss = eval_metrics["recon_loss"]
                    best_checkpoint_selection_score = best_selection_score
                    best_eval_metrics = dict(eval_metrics)
                    primary_path = self.best_semantic_ckpt_path if semantic_selection else self.best_ckpt_path
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        eval_metrics, primary_path,
                    )
                    if semantic_selection:
                        # Keep best.pt as the backward-compatible alias used by export scripts.
                        shutil.copyfile(primary_path, self.best_ckpt_path)
                    has_best = True
                    if hasattr(self.logger, "training_started_at"):
                        self.logger.info(
                            f"best_checkpoint_seconds={time.perf_counter() - self.logger.training_started_at:.6f} "
                            f"epoch={epoch + 1}"
                        )
                    patience_counter = 0
                    checkpoint_label = "best-semantic" if semantic_selection else "best"
                    self.logger.info(
                        f"  [{checkpoint_label}] saved checkpoint "
                        f"(collision={float(eval_metrics['collision_rate']):.6f}, "
                        f"collision_n={int(eval_metrics['collision_count'])}, "
                        f"max_leaf={int(eval_metrics['exact_max_items_per_sid'])}, "
                        f"quant={float(eval_metrics['quantize_loss']):.6f}, "
                        f"recon={best_checkpoint_recon_loss:.6f}, "
                        f"score={best_checkpoint_selection_score:.6f})"
                    )
                else:
                    # Do not spend patience before a semantic run first reaches
                    # its hard collision constraints.
                    if checkpoint_allowed and (has_best or not semantic_selection):
                        patience_counter += 1

                should_stop = checkpoint_allowed and has_best and patience_counter >= self.patience
                if should_stop:
                    self.logger.info(f"Early stopping: patience {patience_counter} >= {self.patience}")
                    break


        if not has_best:
            self.logger.warning(
                "No evaluated best checkpoint is available; "
                "saving the final epoch as fallback.pt"
            )
            save_checkpoint(
                self.model, self.optimizer, epoch,
                eval_metrics if "eval_metrics" in dir() else {},
                self.fallback_ckpt_path,
            )


        if has_best:
            self.logger.info(
                "Best checkpoint: "
                f"collision_rate={float(best_eval_metrics['collision_rate']):.6f}, "
                f"collision_count={int(best_eval_metrics['collision_count'])}, "
                f"max_leaf={int(best_eval_metrics['exact_max_items_per_sid'])}, "
                f"recon_loss={best_checkpoint_recon_loss:.6f}, "
                f"selection_score={best_checkpoint_selection_score:.6f}"
            )
        else:
            self.logger.info("Best checkpoint: no evaluated checkpoint is available")
        if semantic_selection and best_collision_eval_metrics is not None:
            self.logger.info(
                "Best collision checkpoint: "
                f"collision_rate={float(best_collision_eval_metrics['collision_rate']):.6f}, "
                f"collision_count={int(best_collision_eval_metrics['collision_count'])}, "
                f"max_leaf={int(best_collision_eval_metrics['exact_max_items_per_sid'])}"
            )
        self._save_config_snapshot()

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float] | None = None,
    ) -> None:
        """Emit separate aligned tables for Stage 1 train and eval metrics."""
        stage_name = "MP-SID" if str(getattr(self.config.data, "modality", "")) == "unified" else "RQ-VAE"
        train_pairs = [
            ("train/loss", f"{train_metrics['total_loss']:.6f}"),
            ("train/recon", f"{train_metrics['recon_loss']:.6f}"),
            ("train/quant", f"{train_metrics['quantize_loss']:.6f}"),
            ("train/quant_w", f"{train_metrics['quant_weighted_loss']:.6f}"),
        ]
        if "cross_modal_pred_loss" in train_metrics:
            train_pairs.extend([
                ("train/cmp", f"{train_metrics['cross_modal_pred_loss']:.6f}"),
                ("train/cmp_w", f"{train_metrics['cross_modal_pred_weighted_loss']:.6f}"),
                ("train/pred_t", f"{train_metrics.get('pred_loss_text', 0.0):.6f}"),
                ("train/pred_v", f"{train_metrics.get('pred_loss_vision', 0.0):.6f}"),
                ("train/full_n", str(int(train_metrics.get("cross_modal_pred_full_count", 0)))),
                ("train/skip_n", str(int(train_metrics.get("cross_modal_pred_skipped_batches", 0)))),
            ])
        if self.diversity_loss_weight > 0.0:
            train_pairs.append(("train/div", f"{train_metrics.get('diversity_loss', 0.0):.6f}"))
        if self.cf_loss_enabled:
            train_pairs.append(("train/cf", f"{train_metrics.get('cf_loss', 0.0):.6f}"))

        eval_pairs: list[tuple[str, str]] = []
        if eval_metrics is not None:
            eval_pairs.extend([
                ("eval/score", f"{eval_metrics['selection_score']:.6f}"),
                ("eval/collision", f"{eval_metrics['collision_rate']:.4f}"),
                ("eval/unique", str(int(eval_metrics["unique_sid_count"]))),
                ("eval/max_leaf", str(int(eval_metrics["exact_max_items_per_sid"]))),
                ("eval/recon", f"{eval_metrics['recon_loss']:.6f}"),
                ("eval/usage", f"{eval_metrics['codebook_usage']:.4f}"),
                ("eval/entropy", f"{eval_metrics['entropy_ratio']:.4f}"),
            ])
            if "semantic_constraints_met" in eval_metrics:
                eval_pairs.append(("eval/sem_ok", str(int(eval_metrics["semantic_constraints_met"]))))
            for depth in range(1, len(self.prefix_semantic_weights) + 1):
                key = f"prefix{depth}_semantic_score"
                if key in eval_metrics:
                    eval_pairs.append((f"sem/p{depth}", f"{eval_metrics[key]:.6f}"))
            for k in range(int(self.config.model.n_layers)):
                eval_pairs.extend([
                    (f"layer{k}/usage", f"{eval_metrics[f'layer{k}_usage']:.3f}"),
                    (f"layer{k}/entropy", f"{eval_metrics[f'layer{k}_entropy_ratio']:.3f}"),
                    (f"layer{k}/top1", f"{eval_metrics[f'layer{k}_top1_ratio']:.3f}"),
                    (f"layer{k}/top5", f"{eval_metrics[f'layer{k}_top5_ratio']:.3f}"),
                ])
            bucket_labels = {
                "full": "full",
                "collab_text": "ct",
                "collab_vision": "cv",
                "collab_only": "c",
            }
            for bucket, label in bucket_labels.items():
                item_key = f"bucket_{bucket}_items"
                if item_key not in eval_metrics:
                    continue
                item_count = int(eval_metrics[item_key])
                eval_pairs.append((f"bucket/{label}_n", str(item_count)))
                if item_count > 0:
                    eval_pairs.extend([
                        (f"bucket/{label}_col", f"{eval_metrics[f'bucket_{bucket}_collision_rate']:.3f}"),
                        (f"bucket/{label}_uniq", str(int(eval_metrics[f"bucket_{bucket}_unique_sid_count"]))),
                        (f"bucket/{label}_max", str(int(eval_metrics[f"bucket_{bucket}_max_leaf"]))),
                    ])

        epoch_label = f"Epoch {epoch + 1:03d}/{self.max_epochs:03d}"
        tables = [
            format_horizontal_metrics_table(
                f"{stage_name} Train {epoch_label}", train_pairs, pairs_per_row=4,
            ),
        ]
        if eval_metrics is not None:
            tables.append(
                format_horizontal_metrics_table(
                    f"{stage_name} Eval {epoch_label}", eval_pairs, pairs_per_row=4,
                )
            )
        self.logger.info("\n" + "\n\n".join(tables))

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch and return average losses."""
        self.model.train()
        accum = defaultdict(float)
        n_batches = 0

        for batch in self.dataloader:
            emb = batch["embedding"].to(self.device)  # (B, D)
            cf_embedding = self._cf_targets_for_batch(batch)
            output = self.model(emb, cf_embedding=cf_embedding)


            self.optimizer.zero_grad()
            output.loss.backward()
            self.optimizer.step()

            accum["recon_loss"] += output.recon_loss.item()
            accum["quantize_loss"] += output.quantize_loss.item()
            accum["quant_weighted_loss"] += self.model.quant_loss_weight * output.quantize_loss.item()
            diversity_loss = output.diversity_loss
            diversity_weighted_loss = output.diversity_weighted_loss
            accum["diversity_loss"] += 0.0 if diversity_loss is None else diversity_loss.item()
            accum["diversity_weighted_loss"] += (
                0.0 if diversity_weighted_loss is None else diversity_weighted_loss.item()
            )
            cf_loss = output.cf_loss
            cf_weighted_loss = output.cf_weighted_loss
            accum["cf_loss"] += 0.0 if cf_loss is None else cf_loss.item()
            accum["cf_weighted_loss"] += 0.0 if cf_weighted_loss is None else cf_weighted_loss.item()
            accum["total_loss"] += output.loss.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    def _should_refresh_codebook_groups(self, epoch: int) -> bool:
        return (
            self.diversity_loss_weight > 0.0
            and self.codebook_group_refresh_every > 0
            and epoch % self.codebook_group_refresh_every == 0
        )

    def _refresh_codebook_groups(self, epoch: int) -> list[int]:
        random_state = int(getattr(self.config.train, "seed", 19)) + int(epoch)
        return self.model.refresh_codebook_groups(random_state=random_state)

    def _checkpoint_selection_metrics(self, eval_metrics: dict[str, float]) -> dict[str, float]:
        return {
            "selection_score": float(eval_metrics["collision_rate"]),
        }

    def _checkpoint_is_eligible(self, eval_metrics: dict[str, float]) -> bool:
        return True

    def _is_better_checkpoint(
        self,
        eval_metrics: dict[str, float],
        best_eval_metrics: dict[str, float] | None,
    ) -> bool:
        if best_eval_metrics is None:
            return True
        return self._is_better_collision_checkpoint_from_metrics(eval_metrics, best_eval_metrics)

    @classmethod
    def _is_better_collision_checkpoint_from_metrics(
        cls,
        eval_metrics: dict[str, float],
        best_eval_metrics: dict[str, float] | None,
    ) -> bool:
        if best_eval_metrics is None:
            return True
        return cls._is_better_collision_checkpoint(
            eval_metrics,
            float(best_eval_metrics["collision_rate"]),
            float(best_eval_metrics["collision_count"]),
            float(best_eval_metrics["exact_max_items_per_sid"]),
            float(best_eval_metrics["recon_loss"]),
        )

    @staticmethod
    def _is_better_collision_checkpoint(
        eval_metrics: dict[str, float],
        best_collision_rate: float,
        best_collision_count: float,
        best_collision_max_leaf: float,
        best_collision_recon_loss: float,
    ) -> bool:
        rate = float(eval_metrics["collision_rate"])
        count = float(eval_metrics["collision_count"])
        max_leaf = float(eval_metrics["exact_max_items_per_sid"])
        recon = float(eval_metrics["recon_loss"])
        eps = 1.0e-12
        if rate < best_collision_rate - eps:
            return True
        if abs(rate - best_collision_rate) > eps:
            return False
        if count < best_collision_count:
            return True
        if count > best_collision_count:
            return False
        if max_leaf < best_collision_max_leaf:
            return True
        if max_leaf > best_collision_max_leaf:
            return False
        return recon < best_collision_recon_loss - 1.0e-6

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate all valid items and report reconstruction, codebook usage, entropy, and collision statistics."""
        self.model.eval()


        all_emb = torch.stack(
            [self.full_dataset[i]["embedding"] for i in range(len(self.full_dataset))]
        ).to(self.device)  # (N_valid, D)


        output = self.model(all_emb)
        recon_loss = output.recon_loss.item()
        quantize_loss = output.quantize_loss.item()
        all_ids_stacked = torch.stack(output.all_ids, dim=-1)  # (N_valid, n_layers)


        n_layers = all_ids_stacked.shape[1]
        total_items = all_ids_stacked.shape[0]
        codebook_size = self.config.model.codebook_size


        usage_per_layer = []
        entropy_per_layer = []
        top1_ratio_per_layer = []
        top5_ratio_per_layer = []

        for k in range(n_layers):
            ids_k = all_ids_stacked[:, k]
            unique_ids, counts = torch.unique(ids_k, return_counts=True)
            usage_per_layer.append(len(unique_ids) / codebook_size)

            probs = counts.float() / counts.sum()
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            entropy_per_layer.append(entropy / math.log(codebook_size))

            sorted_counts = torch.sort(counts.float(), descending=True).values
            top1_ratio_per_layer.append((sorted_counts[0] / total_items).item())
            top5_ratio_per_layer.append((sorted_counts[:5].sum() / total_items).item())

        codebook_usage = sum(usage_per_layer) / n_layers
        codebook_entropy_ratio = sum(entropy_per_layer) / n_layers

        # ---- collision_rate: 1 - unique_sids / total_items ----

        sid_counts: dict[tuple, int] = {}
        for i in range(total_items):
            sid = tuple(all_ids_stacked[i].tolist())
            sid_counts[sid] = sid_counts.get(sid, 0) + 1
        unique_sids = len(sid_counts)
        collision_count = total_items - unique_sids
        collision_rate = 1.0 - unique_sids / total_items
        exact_avg_items_per_sid = total_items / max(unique_sids, 1)
        exact_max_items_per_sid = max(sid_counts.values()) if sid_counts else 0


        counts = torch.tensor(list(sid_counts.values()), dtype=torch.float)
        probs = counts / counts.sum()
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        max_entropy = math.log(total_items)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0

        self.model.train()

        metrics = {
            "total_items": total_items,
            "recon_loss": recon_loss,
            "quantize_loss": quantize_loss,
            "codebook_usage": codebook_usage,
            "codebook_entropy_ratio": codebook_entropy_ratio,
            "collision_rate": collision_rate,
            "collision_count": collision_count,
            "unique_sid_count": unique_sids,
            "entropy_ratio": entropy_ratio,
            "exact_avg_items_per_sid": exact_avg_items_per_sid,
            "exact_max_items_per_sid": exact_max_items_per_sid,
        }

        for k in range(n_layers):
            metrics[f"layer{k}_usage"] = usage_per_layer[k]
            metrics[f"layer{k}_entropy_ratio"] = entropy_per_layer[k]
            metrics[f"layer{k}_top1_ratio"] = top1_ratio_per_layer[k]
            metrics[f"layer{k}_top5_ratio"] = top5_ratio_per_layer[k]

        return metrics

    def _save_config_snapshot(self) -> None:
        """Save the resolved configuration as YAML."""
        import yaml
        config_path = os.path.join(self.run_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True)


class MPSIDTrainer(RQVAETrainer):
    """MP-SID trainer that reuses RQVAETrainer checkpointing and logging policy."""

    def _meets_semantic_constraints(self, eval_metrics: dict[str, float]) -> bool:
        return (
            float(eval_metrics["collision_rate"]) <= self.semantic_collision_rate_max + 1.0e-12
            and int(eval_metrics["exact_max_items_per_sid"]) <= self.semantic_max_leaf
        )

    def _checkpoint_selection_metrics(self, eval_metrics: dict[str, float]) -> dict[str, float]:
        if self.es_metric != "prefix_semantic":
            return super()._checkpoint_selection_metrics(eval_metrics)
        return {
            "selection_score": float(eval_metrics.get("prefix_semantic_score", float("-inf"))),
        }

    def _checkpoint_is_eligible(self, eval_metrics: dict[str, float]) -> bool:
        if self.es_metric != "prefix_semantic":
            return True
        score = float(eval_metrics.get("prefix_semantic_score", float("-inf")))
        return self._meets_semantic_constraints(eval_metrics) and math.isfinite(score)

    def _is_better_checkpoint(
        self,
        eval_metrics: dict[str, float],
        best_eval_metrics: dict[str, float] | None,
    ) -> bool:
        if self.es_metric != "prefix_semantic":
            return super()._is_better_checkpoint(eval_metrics, best_eval_metrics)
        if best_eval_metrics is None:
            return True

        score = float(eval_metrics["prefix_semantic_score"])
        best_score = float(best_eval_metrics["prefix_semantic_score"])
        tolerance = self.prefix_semantic_tolerance
        if score > best_score + tolerance:
            return True
        if score < best_score - tolerance:
            return False

        quantize = float(eval_metrics["quantize_loss"])
        best_quantize = float(best_eval_metrics["quantize_loss"])
        eps = 1.0e-12
        if quantize < best_quantize - eps:
            return True
        if quantize > best_quantize + eps:
            return False
        return float(eval_metrics["recon_loss"]) < float(best_eval_metrics["recon_loss"]) - eps

    @staticmethod
    def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            out[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        return out

    def _iter_full_dataset_batches(self):
        batch_size = int(self.config.train.batch_size)
        iter_batches = getattr(self.full_dataset, "iter_batches", None)
        if callable(iter_batches):
            yield from iter_batches(batch_size)
            return
        yield from DataLoader(
            self.full_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(epoch)
        accum = defaultdict(float)
        n_batches = 0

        for batch in self.dataloader:
            batch_on_device = self._batch_to_device(batch, self.device)
            output = self.model(batch_on_device)

            self.optimizer.zero_grad()
            output.loss.backward()
            self.optimizer.step()

            self._accumulate_mpsid_output(accum, output)
            n_batches += 1

        averaged = {
            key: value / max(n_batches, 1)
            for key, value in accum.items()
        }
        averaged["cross_modal_pred_full_count"] = accum.get("cross_modal_pred_full_count", 0.0)
        averaged["cross_modal_pred_skipped_batches"] = accum.get("cross_modal_pred_skipped_batches", 0.0)
        return averaged

    def _accumulate_mpsid_output(
        self,
        accum: defaultdict[str, float],
        output,
        total_loss: torch.Tensor | None = None,
    ) -> None:
        accum["recon_loss"] += output.recon_loss.item()
        accum["quantize_loss"] += output.quantize_loss.item()
        accum["quant_weighted_loss"] += self.model.quant_loss_weight * output.quantize_loss.item()
        pred_loss = output.cross_modal_pred_loss
        pred_weighted_loss = output.cross_modal_pred_weighted_loss
        pred_loss_text = output.pred_loss_text
        pred_loss_vision = output.pred_loss_vision
        accum["cross_modal_pred_loss"] += 0.0 if pred_loss is None else pred_loss.item()
        accum["cross_modal_pred_weighted_loss"] += 0.0 if pred_weighted_loss is None else pred_weighted_loss.item()
        accum["pred_loss_text"] += 0.0 if pred_loss_text is None else pred_loss_text.item()
        accum["pred_loss_vision"] += 0.0 if pred_loss_vision is None else pred_loss_vision.item()
        accum["cross_modal_pred_full_count"] += float(output.cross_modal_pred_full_count)
        accum["cross_modal_pred_skipped_batches"] += 1.0 if output.cross_modal_pred_skipped else 0.0
        accum["total_loss"] += (output.loss if total_loss is None else total_loss).item()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        accum = defaultdict(float)
        n_batches = 0
        all_ids: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_item_ids: list[torch.Tensor] = []

        try:
            for batch in self._iter_full_dataset_batches():
                batch_on_device = self._batch_to_device(batch, self.device)
                output = self.model(batch_on_device)
                self._accumulate_mpsid_output(accum, output)
                all_ids.append(output.semantic_codes.detach().cpu())
                all_masks.append(batch["modality_mask"].detach().cpu().bool())
                all_item_ids.append(batch["item_id"].detach().cpu().long())
                n_batches += 1

            all_ids_stacked = torch.cat(all_ids, dim=0)
            all_masks_stacked = torch.cat(all_masks, dim=0)
            all_item_ids_stacked = torch.cat(all_item_ids, dim=0)
            metrics = self._semantic_id_metrics(all_ids_stacked)
            metrics["recon_loss"] = accum["recon_loss"] / max(n_batches, 1)
            metrics["quantize_loss"] = accum["quantize_loss"] / max(n_batches, 1)
            metrics["cross_modal_pred_loss"] = accum["cross_modal_pred_loss"] / max(n_batches, 1)
            metrics["pred_loss_text"] = accum["pred_loss_text"] / max(n_batches, 1)
            metrics["pred_loss_vision"] = accum["pred_loss_vision"] / max(n_batches, 1)
            metrics.update(self._availability_bucket_metrics(all_ids_stacked, all_masks_stacked))
            constraints_met = self._meets_semantic_constraints(metrics)
            metrics["semantic_constraints_met"] = float(constraints_met)
            if self.es_metric == "prefix_semantic" and constraints_met:
                metrics.update(
                    self._prefix_semantic_metrics(
                        all_ids_stacked,
                        all_masks_stacked,
                        all_item_ids_stacked,
                    )
                )
            return metrics
        finally:
            self.model.train(was_training)

    def _semantic_id_metrics(self, all_ids_stacked: torch.Tensor) -> dict[str, float]:
        n_layers = all_ids_stacked.shape[1]
        total_items = all_ids_stacked.shape[0]
        codebook_size = self.config.model.codebook_size

        usage_per_layer = []
        entropy_per_layer = []
        top1_ratio_per_layer = []
        top5_ratio_per_layer = []
        for k in range(n_layers):
            ids_k = all_ids_stacked[:, k]
            unique_ids, counts = torch.unique(ids_k, return_counts=True)
            usage_per_layer.append(len(unique_ids) / codebook_size)
            probs = counts.float() / counts.sum()
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            entropy_per_layer.append(entropy / math.log(codebook_size))
            sorted_counts = torch.sort(counts.float(), descending=True).values
            top1_ratio_per_layer.append((sorted_counts[0] / total_items).item())
            top5_ratio_per_layer.append((sorted_counts[:5].sum() / total_items).item())

        sid_stats = self._collision_stats(all_ids_stacked)
        counts = sid_stats["sid_counts"].to(dtype=torch.float)
        probs = counts / counts.sum()
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        max_entropy = math.log(total_items)
        metrics = {
            "total_items": total_items,
            "codebook_usage": sum(usage_per_layer) / n_layers,
            "codebook_entropy_ratio": sum(entropy_per_layer) / n_layers,
            "collision_rate": sid_stats["collision_rate"],
            "collision_count": sid_stats["collision_count"],
            "unique_sid_count": sid_stats["unique_sid_count"],
            "entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0.0,
            "exact_avg_items_per_sid": sid_stats["avg_items_per_sid"],
            "exact_max_items_per_sid": sid_stats["max_leaf"],
        }
        for k in range(n_layers):
            metrics[f"layer{k}_usage"] = usage_per_layer[k]
            metrics[f"layer{k}_entropy_ratio"] = entropy_per_layer[k]
            metrics[f"layer{k}_top1_ratio"] = top1_ratio_per_layer[k]
            metrics[f"layer{k}_top5_ratio"] = top5_ratio_per_layer[k]
        return metrics

    def _prefix_semantic_metrics(
        self,
        ids: torch.Tensor,
        masks: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, float]:
        """Compute exact catalog-level prefix cohesion without pairwise matrices.

        Prefix groups are averaged equally, matching the existing SID similarity
        analysis. For a random group sampled without replacement, the expected
        mean pairwise cosine is the catalog-wide mean over distinct pairs,
        independent of group size. This analytic expectation is therefore the
        deterministic equivalent of infinitely many size-matched baselines.
        """
        expected_shape = (ids.shape[0], 3)
        if tuple(masks.shape) != expected_shape:
            raise ValueError(f"modality masks should have shape {expected_shape}, got {tuple(masks.shape)}")
        if tuple(item_ids.shape) != (ids.shape[0],):
            raise ValueError(f"item_ids should have shape ({ids.shape[0]},), got {tuple(item_ids.shape)}")

        modality_sources = (
            ("collab", 0, "collab_embeddings"),
            ("text", 1, "text_embeddings"),
            ("vision", 2, "vision_embeddings"),
        )
        prefix_groups: list[tuple[torch.Tensor, int]] = []
        base = int(self.config.model.codebook_size)
        keys = ids[:, 0].long().clone()
        for depth in range(1, len(self.prefix_semantic_weights) + 1):
            if depth > 1:
                keys.mul_(base).add_(ids[:, depth - 1].long())
            _unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            prefix_groups.append((inverse, int(_unique_keys.numel())))

        metrics: dict[str, float] = {}
        prefix_scores: list[float] = []
        catalog_dataset = getattr(self.full_dataset, "_delegate", None)
        if catalog_dataset is None:
            catalog_dataset = self.full_dataset
        for depth, (group_ids, n_groups) in enumerate(prefix_groups, start=1):
            deltas: list[float] = []
            for modality, mask_index, embedding_attr in modality_sources:
                catalog_embeddings = getattr(catalog_dataset, embedding_attr, None)
                if not isinstance(catalog_embeddings, torch.Tensor):
                    raise TypeError(f"unified Stage 1 dataset is missing tensor {embedding_attr!r}")
                observed = masks[:, mask_index]
                observed_item_ids = item_ids[observed]
                if observed_item_ids.numel() == 0:
                    continue
                embeddings = catalog_embeddings.index_select(0, observed_item_ids)
                stats = self._prefix_cosine_statistics(
                    embeddings,
                    group_ids[observed],
                    n_groups,
                )
                if stats is None:
                    continue
                within, baseline, delta, pair_count = stats
                metrics[f"prefix{depth}_{modality}_cosine"] = within
                metrics[f"prefix{depth}_{modality}_baseline"] = baseline
                metrics[f"prefix{depth}_{modality}_delta"] = delta
                metrics[f"prefix{depth}_{modality}_pairs"] = float(pair_count)
                deltas.append(delta)

            if not deltas:
                metrics["prefix_semantic_score"] = float("-inf")
                return metrics
            prefix_score = sum(deltas) / len(deltas)
            metrics[f"prefix{depth}_semantic_score"] = prefix_score
            prefix_scores.append(prefix_score)

        weight_sum = sum(self.prefix_semantic_weights)
        metrics["prefix_semantic_score"] = sum(
            weight * score
            for weight, score in zip(self.prefix_semantic_weights, prefix_scores)
        ) / weight_sum
        return metrics

    @staticmethod
    def _prefix_cosine_statistics(
        embeddings: torch.Tensor,
        group_ids: torch.Tensor,
        n_groups: int,
    ) -> tuple[float, float, float, int] | None:
        if embeddings.dim() != 2 or tuple(group_ids.shape) != (embeddings.shape[0],):
            raise ValueError("embeddings/group_ids shapes are inconsistent")
        if embeddings.shape[0] < 2:
            return None

        work = embeddings.detach().float().cpu()
        groups = group_ids.detach().long().cpu()
        norms = work.norm(dim=-1)
        valid = torch.isfinite(work).all(dim=-1) & torch.isfinite(norms) & (norms > 0.0)
        if int(valid.sum().item()) < 2:
            return None
        work = work[valid] / norms[valid].unsqueeze(-1)
        groups = groups[valid]

        counts = torch.bincount(groups, minlength=int(n_groups))
        ordered_pair_counts = counts * (counts - 1)
        active_groups = ordered_pair_counts > 0
        pair_count = int(ordered_pair_counts[active_groups].sum().item())
        if pair_count == 0:
            return None

        group_sums = work.new_zeros((int(n_groups), work.shape[1]))
        group_sums.index_add_(0, groups, work)
        row_norm_sq = work.square().sum(dim=-1)
        group_norm_sq = work.new_zeros(int(n_groups))
        group_norm_sq.index_add_(0, groups, row_norm_sq)
        within_numerator_by_group = (
            group_sums[active_groups].square().sum(dim=-1)
            - group_norm_sq[active_groups]
        )
        within_by_group = within_numerator_by_group / ordered_pair_counts[active_groups]
        within = float(within_by_group.mean().item())

        total_items = int(work.shape[0])
        baseline_numerator = work.sum(dim=0).square().sum() - row_norm_sq.sum()
        baseline = float((baseline_numerator / (total_items * (total_items - 1))).item())
        return within, baseline, within - baseline, pair_count

    @staticmethod
    def _collision_stats(ids: torch.Tensor) -> dict[str, object]:
        if ids.dim() != 2:
            raise ValueError(f"semantic ids must have shape [N, K], got {tuple(ids.shape)}")
        if ids.shape[1] <= 0:
            raise ValueError("semantic ids must contain at least one layer")
        if ids.is_floating_point() or ids.is_complex():
            raise TypeError(f"semantic ids must use an integer dtype, got {ids.dtype}")

        total_items = int(ids.shape[0])
        if total_items == 0:
            ordered_counts = torch.empty(0, dtype=torch.long, device=ids.device)
        else:
            ids_long = ids.long()
            min_code = int(ids_long.min().item())
            if min_code < 0:
                raise ValueError(f"semantic ids must be non-negative, got minimum {min_code}")
            base = int(ids_long.max().item()) + 1
            if base ** int(ids_long.shape[1]) > torch.iinfo(torch.int64).max + 1:
                raise OverflowError(
                    f"semantic SID mixed-radix key overflows int64: base={base}, layers={ids_long.shape[1]}"
                )

            keys = ids_long[:, 0].clone()
            for layer_index in range(1, ids_long.shape[1]):
                keys.mul_(base).add_(ids_long[:, layer_index])
            _unique_keys, inverse, counts = torch.unique(
                keys,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )

            # Python dicts preserve first-occurrence order. Keep that count order so
            # the downstream float32 entropy reduction remains bitwise unchanged.
            first_positions = torch.full(
                (counts.numel(),),
                total_items,
                dtype=torch.long,
                device=ids.device,
            )
            first_positions.scatter_reduce_(
                0,
                inverse,
                torch.arange(total_items, dtype=torch.long, device=ids.device),
                reduce="amin",
                include_self=True,
            )
            ordered_counts = counts.index_select(0, torch.argsort(first_positions))

        unique_sids = int(ordered_counts.numel())
        return {
            "sid_counts": ordered_counts,
            "collision_count": total_items - unique_sids,
            "collision_rate": 1.0 - unique_sids / max(total_items, 1),
            "unique_sid_count": unique_sids,
            "avg_items_per_sid": total_items / max(unique_sids, 1),
            "max_leaf": int(ordered_counts.max().item()) if unique_sids else 0,
        }

    def _availability_bucket_metrics(self, ids: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
        buckets = {
            "full": torch.tensor([True, True, True]),
            "collab_text": torch.tensor([True, True, False]),
            "collab_vision": torch.tensor([True, False, True]),
            "collab_only": torch.tensor([True, False, False]),
        }
        metrics: dict[str, float] = {}
        for name, pattern in buckets.items():
            rows = (masks == pattern).all(dim=-1)
            bucket_ids = ids[rows]
            if bucket_ids.numel() == 0:
                metrics[f"bucket_{name}_items"] = 0.0
                metrics[f"bucket_{name}_collision_rate"] = 0.0
                metrics[f"bucket_{name}_collision_count"] = 0.0
                metrics[f"bucket_{name}_unique_sid_count"] = 0.0
                metrics[f"bucket_{name}_max_leaf"] = 0.0
                continue
            stats = self._collision_stats(bucket_ids)
            metrics[f"bucket_{name}_items"] = float(bucket_ids.shape[0])
            metrics[f"bucket_{name}_collision_rate"] = float(stats["collision_rate"])
            metrics[f"bucket_{name}_collision_count"] = float(stats["collision_count"])
            metrics[f"bucket_{name}_unique_sid_count"] = float(stats["unique_sid_count"])
            metrics[f"bucket_{name}_max_leaf"] = float(stats["max_leaf"])
        return metrics
