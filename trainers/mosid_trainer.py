"""Trainer for the MoSID Stage 2 encoder-decoder backbone."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from configs.config import DotDict
from data.prefix_hard_negatives import (
    PrefixHardNegativeSampler,
    merge_shared_raw_candidates,
    random_replacement_candidates,
)
from data.raw_embedding_catalog import RawEmbeddingCatalog
from evaluate.beam_eval import (
    MOSID_CHECKPOINT_SCHEMA_VERSION,
    RRFSettings,
    evaluate_beam,
    format_beam_metrics_summary,
    get_test_beam_size,
    get_valid_beam_size,
    get_rrf_settings,
    get_max_batches,
    get_top_k,
    rrf_settings_metrics,
    resolve_beam_metric_score,
    serialize_rrf_settings,
)
from losses.gen_loss import GenLossOutput, weighted_autoregressive_ce_loss
from losses.memory_orth import memory_orth_loss
from models.raw_compatibility import RawCompatibilityOutput, unique_target_candidates
from utils.data import (
    MPSID_INPUT_MODALITIES,
    codebook_weights_filename,
    mosid_sid_artifact_channel,
    sid_codes_filename,
)
from utils.io import load_checkpoint, save_checkpoint, save_json
from utils.logger import Logger, format_horizontal_metrics_table, progress_iter


@dataclass
class MoSIDLossOutput:
    loss: Tensor
    rg_loss: Tensor
    ce_loss: Tensor
    raw_loss: Tensor
    memory_orth_loss: Tensor
    raw_weight: float
    memory_orth_weight: float
    token_acc: float
    valid_tokens: int
    diagnostics: dict[str, Tensor | float] = field(default_factory=dict)

    @property
    def total_loss(self) -> Tensor:
        return self.loss

    @property
    def gen_loss(self) -> Tensor:
        return self.rg_loss


class MoSIDTrainer:
    """Train the MoSID Stage 2 backbone with generation and optional raw loss."""

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        config: DotDict,
        logger: Logger,
        device: torch.device,
        raw_catalog: RawEmbeddingCatalog | None = None,
        prefix_negative_sampler: PrefixHardNegativeSampler | None = None,
        beam_valid_loader: DataLoader | None = None,
        beam_trie: Any | None = None,
        selection_metric: str | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.beam_valid_loader = beam_valid_loader
        self.beam_trie = beam_trie
        self.config = config
        self.logger = logger
        self.device = device
        self.raw_catalog = raw_catalog
        self.prefix_negative_sampler = prefix_negative_sampler
        self.optimized_parameters = self._collect_optimizer_parameters()

        self.max_epochs = int(config.train.max_epochs)
        self.eval_every = int(config.train.eval_every)
        self.patience = int(getattr(config.train, "patience", 0))
        self.early_stop_min_delta = float(getattr(config.train, "early_stop_min_delta", 0.0))
        self.grad_clip = float(getattr(config.train, "grad_clip", 0.0))
        self.grad_accum_steps = int(getattr(config.train, "grad_accum_steps", 1))
        if self.grad_accum_steps <= 0:
            raise ValueError(f"train.grad_accum_steps must be positive, got {self.grad_accum_steps}")
        self.active_channels = list(getattr(self.config.model, "active_channels", ["collab"]))
        self.micro_batch_size = int(getattr(self.train_loader, "batch_size", 0) or getattr(config.train, "batch_size", 1))
        self.effective_batch_size = self.micro_batch_size * self.grad_accum_steps
        self.use_tb = bool(getattr(config.train, "tensorboard", False))
        self.is_single_channel = len(self.active_channels) == 1
        self.is_collab_only = self.active_channels == ["collab"]
        self.rrf_enabled = not self.is_single_channel and "rrf" in config
        raw_branch_weights = getattr(config.loss, "branch_weights", {"collab": 1.0})
        if hasattr(raw_branch_weights, "to_dict"):
            raw_branch_weights = raw_branch_weights.to_dict()
        self.branch_weights = {key: float(value) for key, value in raw_branch_weights.items()}
        self.raw_aux_weight = float(getattr(config.loss, "raw_aux_weight", 0.0))
        self.raw_temperature = float(getattr(config.loss, "raw_temperature", 0.07))
        self.raw_training_enabled = self.raw_aux_weight > 0.0
        self.raw_random_negatives = bool(getattr(config.loss, "raw_random_negatives", False))
        self.memory_orth_weight = float(config.loss.memory_orth_weight)
        self.memory_orth_direct = bool(getattr(config.loss, "memory_orth_direct", False))
        self.memory_orth_center = bool(getattr(config.loss, "memory_orth_center", True))
        self.memory_orth_uniform_weights = bool(getattr(config.loss, "memory_orth_uniform_weights", False))
        self.memory_orth_enabled = self.memory_orth_weight > 0.0
        if self.raw_training_enabled:
            if self.raw_catalog is None:
                raise ValueError("raw auxiliary training requires a RawEmbeddingCatalog")
            if self.prefix_negative_sampler is None:
                raise ValueError("raw auxiliary training requires a PrefixHardNegativeSampler")
            self._raw_scorer()
        self.run_dir = str(logger.log_dir)
        self.best_ckpt_path = os.path.join(self.run_dir, "best.pt")
        self.last_ckpt_path = os.path.join(self.run_dir, "last.pt")
        if self.beam_valid_loader is None or self.beam_trie is None:
            raise ValueError("MoSID training requires beam_valid_loader and beam_trie for validation")
        default_metric = getattr(config.train, "early_stop_metric", None)
        if default_metric is None and self.rrf_enabled:
            default_metric = getattr(config.rrf, "selection_metric", None)
        self.selection_metric = str(selection_metric or default_metric or "beam_ndcg@10")
        self.rrf_tune_every = self._resolve_rrf_tune_every()
        self.current_rrf_settings: RRFSettings | None = (
            get_rrf_settings(config)
            if self.rrf_enabled
            else None
        )
        self.current_rrf_selection: dict[str, Any] = {
            "source": "config_default",
            "selection_metric": self.selection_metric,
            "validation_max_batches": get_max_batches(self.config),
            "validation_complete": False,
            "search_exhaustive": False,
            "grid_size": 1,
        }
        self.baseline_name = (
            "B2-stage2-joint-raw-compatibility"
            if self.raw_training_enabled
            else (
                f"B0-{self.active_channels[0]}-generation"
                if self.is_single_channel
                else "B1-multi-channel-generation"
            )
        )

    def train(self) -> dict[str, float]:
        if self.eval_every <= 0:
            raise ValueError(f"train.eval_every must be positive, got {self.eval_every}")

        self.logger.info(
            f"MoSID training, "
            f"best checkpoint is selected by validation {self.selection_metric}."
        )
        self.logger.info(
            "gradient accumulation: "
            f"micro_batch_size={self.micro_batch_size}, "
            f"grad_accum_steps={self.grad_accum_steps}, "
            f"effective_batch_size={self.effective_batch_size}"
        )
        self.logger.info(
            (
                "RRF validation: evaluate every eval round; "
                f"tune on epochs divisible by {self.rrf_tune_every}; "
                f"sample at most {self._rrf_sample_count()} parameter combinations"
            )
            if self.rrf_enabled
            else "single-channel validation: direct Trie beam ranking; RRF disabled"
        )
        self.logger.info(self._format_branch_weight_summary())
        self.logger.info(
            "raw auxiliary training: "
            f"enabled={self.raw_training_enabled}, weight={self.raw_aux_weight:.6g}, "
            f"temperature={self.raw_temperature:.6g}"
        )
        best_score = float("-inf")
        no_improve = 0
        has_best = False
        last_metrics: dict[str, float] = {}

        for epoch in range(self.max_epochs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            start = time.time()
            train_metrics = self._train_one_epoch(epoch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.time() - start
            train_metrics["epoch_seconds"] = float(elapsed)
            train_metrics["samples_per_second"] = (
                float(train_metrics.get("samples", 0.0)) / max(elapsed, 1.0e-12)
            )
            train_metrics["peak_memory_mb"] = (
                float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0)
                if self.device.type == "cuda"
                else 0.0
            )
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.logger.info(self._format_train_summary(epoch, train_metrics, lr, elapsed))
            last_metrics = self._checkpoint_metrics(epoch, train_metrics)
            should_eval = (epoch + 1) % self.eval_every == 0
            if not should_eval:
                self._save_checkpoint(epoch, last_metrics, self.last_ckpt_path)

            if should_eval:
                eval_round = int((epoch + 1) // self.eval_every)
                tune_rrf = self._should_tune_rrf(epoch + 1)
                beam_metrics = self.evaluate_beam(split="valid", tune_rrf=tune_rrf)
                self.logger.info(
                    "validation timing: "
                    f"beam_size={beam_metrics.get('eval_beam_size', 0):.0f}, "
                    f"candidates={beam_metrics.get('eval_candidate_seconds', 0):.2f}s, "
                    f"cache={beam_metrics.get('eval_cache_seconds', 0):.2f}s, "
                    f"ranking={beam_metrics.get('eval_ranking_seconds', 0):.2f}s, "
                    f"diagnostics={beam_metrics.get('eval_diagnostics_seconds', 0):.2f}s, "
                    f"total={beam_metrics.get('eval_total_seconds', 0):.2f}s"
                )
                selection_score = resolve_beam_metric_score(
                    beam_metrics,
                    self.selection_metric,
                    label="checkpoint selection metric",
                )
                self.logger.info(self._format_beam_summary(epoch, beam_metrics, selection_score))
                checkpoint_metrics = self._checkpoint_metrics(
                    epoch,
                    train_metrics,
                    beam_metrics,
                    selection_score=selection_score,
                    rrf_eval_round=eval_round if self.rrf_enabled else None,
                    rrf_tuned_this_eval=tune_rrf if self.rrf_enabled else None,
                )
                improved = selection_score > best_score + self.early_stop_min_delta
                current_text = f"{self.selection_metric}={selection_score:.6f}"
                best_text = f"{best_score:.6f}" if has_best else "none"

                last_metrics = checkpoint_metrics
                self._save_checkpoint(epoch, checkpoint_metrics, self.last_ckpt_path)
                if improved:
                    best_score = selection_score
                    no_improve = 0
                    self._save_checkpoint(epoch, checkpoint_metrics, self.best_ckpt_path)
                    if hasattr(self.logger, "training_started_at"):
                        self.logger.info(
                            f"best_checkpoint_seconds={time.perf_counter() - self.logger.training_started_at:.6f} "
                            f"epoch={epoch + 1}"
                        )
                    has_best = True
                    self.logger.info(f"  saved best MoSID checkpoint: {current_text}")
                else:
                    no_improve += 1
                    self.logger.info(
                        f"  early-stop counter: {no_improve}/{self.patience} "
                        f"({current_text}, best={best_text})"
                    )
                if self.use_tb:
                    self._log_tensorboard(beam_metrics, epoch, prefix="valid/")
                if self.patience > 0 and no_improve >= self.patience:
                    self.logger.info(f"early stopping triggered: no improvement for {no_improve} eval rounds")
                    break

            if self.use_tb:
                self._log_tensorboard(train_metrics, epoch, prefix="train/")

        if not has_best:
            self._save_checkpoint(epoch, last_metrics, self.best_ckpt_path)
            if hasattr(self.logger, "training_started_at"):
                self.logger.info(
                    f"best_checkpoint_seconds={time.perf_counter() - self.logger.training_started_at:.6f} "
                    f"epoch={epoch + 1}"
                )
            has_best = True
            self.logger.warning("No best checkpoint was selected; saved the final epoch as best.pt")

        meta = load_checkpoint(self.best_ckpt_path, self.model, optimizer=None, device=self.device)
        self.logger.info(f"checkpoint reload check passed: epoch={meta['epoch'] + 1}")
        self._save_config_snapshot()
        self._save_artifact_snapshot(meta.get("metrics", {}))
        return dict(meta.get("metrics", {}))

    def _save_checkpoint(self, epoch: int, metrics: dict[str, Any], path: str) -> None:
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            metrics,
            path,
            extra_state=self._checkpoint_extra_state(),
        )

    def _checkpoint_extra_state(self) -> dict[str, Any]:
        extra_state: dict[str, Any] = {
            "mosid_checkpoint_schema_version": MOSID_CHECKPOINT_SCHEMA_VERSION,
            "raw_random_negatives": self.raw_random_negatives,
            "raw_use_history_reliability": bool(getattr(self.config.loss, "raw_use_history_reliability", True)),
            "memory_orth": {
                "weight": self.memory_orth_weight,
                "direct": self.memory_orth_direct,
                "center": self.memory_orth_center,
                "uniform_weights": self.memory_orth_uniform_weights,
            },
        }
        if self.rrf_enabled:
            if self.current_rrf_settings is None:
                raise RuntimeError("multi-branch checkpoint requires a selected RRF setting")
            extra_state["selected_rrf"] = serialize_rrf_settings(self.current_rrf_settings)
            extra_state["rrf_selection"] = dict(self.current_rrf_selection)
        return extra_state

    def _collect_optimizer_parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for group in self.optimizer.param_groups:
            for parameter in group.get("params", []):
                if isinstance(parameter, Tensor):
                    params.append(parameter)
        return params

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if self.prefix_negative_sampler is not None:
            self.prefix_negative_sampler.set_epoch(epoch)
        accum = defaultdict(float)
        n_batches = 0
        progress = progress_iter(
            self.train_loader,
            total=len(self.train_loader) if hasattr(self.train_loader, "__len__") else None,
            desc=f"train {epoch + 1}/{self.max_epochs}",
            leave=False,
            enabled=True,
        )
        total_batches = len(self.train_loader) if hasattr(self.train_loader, "__len__") else None
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        for batch_idx, batch in enumerate(progress):
            batch = self._move_batch(batch)
            output = self.model(batch)
            loss_output = self._compute_loss(
                output,
                batch=batch,
                debug_context=f"train epoch={epoch + 1} batch={batch_idx + 1}",
            )
            accum_window = self._accum_window_size(batch_idx, total_batches)
            (loss_output.loss / float(accum_window)).backward()
            should_step = (batch_idx + 1) % self.grad_accum_steps == 0
            if total_batches is not None:
                should_step = should_step or (batch_idx + 1) == total_batches

            grad_norm = None
            if should_step:
                grad_norm = 0.0
                if self.grad_clip > 0.0:
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.optimized_parameters, self.grad_clip)
                    self._assert_finite_scalar(
                        grad_norm_tensor,
                        context=f"gradient norm before optimizer step epoch={epoch + 1} batch={batch_idx + 1}",
                    )
                    grad_norm = float(grad_norm_tensor.item())
                next_optimizer_step = optimizer_steps + 1
                self.optimizer.step()
                self._assert_finite_model_parameters(
                    f"after optimizer step epoch={epoch + 1} batch={batch_idx + 1} opt_step={next_optimizer_step}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps = next_optimizer_step

            loss_scalars = self._loss_output_scalars(loss_output)
            self._accumulate(accum, loss_output, grad_norm, loss_scalars=loss_scalars)
            accum["samples"] += float(self._batch_size(batch))
            n_batches += 1
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=f"{loss_scalars[0]:.4f}",
                    rg=f"{loss_scalars[1]:.4f}",
                    raw=f"{loss_scalars[3]:.4f}",
                    mem=f"{loss_scalars[4]:.4f}",
                    acc=f"{loss_output.token_acc:.3f}",
                    tok=loss_output.valid_tokens,
                    accum=f"{(batch_idx % self.grad_accum_steps) + 1}/{accum_window}",
                )

        if total_batches is None and n_batches > optimizer_steps * self.grad_accum_steps:
            grad_norm = 0.0
            if self.grad_clip > 0.0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.optimized_parameters, self.grad_clip)
                self._assert_finite_scalar(
                    grad_norm_tensor,
                    context=f"gradient norm before final optimizer step epoch={epoch + 1}",
                )
                grad_norm = float(grad_norm_tensor.item())
            next_optimizer_step = optimizer_steps + 1
            self.optimizer.step()
            self._assert_finite_model_parameters(
                f"after final optimizer step epoch={epoch + 1} opt_step={next_optimizer_step}"
            )
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_steps = next_optimizer_step
            accum["grad_norm"] += float(grad_norm)
            accum["grad_norm_count"] += 1.0

        metrics = self._average_metrics(accum, n_batches)
        metrics["optimizer_steps"] = float(optimizer_steps)
        metrics["grad_accum_steps"] = float(self.grad_accum_steps)
        metrics["micro_batch_size"] = float(self.micro_batch_size)
        metrics["effective_batch_size"] = float(self.effective_batch_size)
        return metrics

    @torch.no_grad()
    def evaluate_beam(self, split: str, tune_rrf: bool | None = None) -> dict[str, float]:
        if self.beam_valid_loader is None or self.beam_trie is None:
            raise ValueError("beam validation requires beam_valid_loader and beam_trie")
        should_tune = bool(tune_rrf)
        output = evaluate_beam(
            self.model,
            self.beam_valid_loader,
            self.beam_trie,
            self.config,
            self.device,
            split=split,
            rrf_settings=self.current_rrf_settings,
            tune_rrf=should_tune,
            tune_metric=getattr(getattr(self.config, "rrf", None), "selection_metric", self.selection_metric),
            tune_seed=getattr(self.config.train, "seed", None),
            rrf_exhaustive=False,
            raw_catalog=self.raw_catalog,
            beam_size=get_valid_beam_size(self.config),
        )
        output_rrf_settings = getattr(output, "rrf_settings", None)
        if output_rrf_settings is not None:
            self.current_rrf_settings = output_rrf_settings
        if bool(getattr(output, "rrf_tuned", False)):
            max_batches = get_max_batches(self.config)
            self.current_rrf_selection = {
                "source": "validation_grid",
                "selection_metric": str(
                    getattr(getattr(self.config, "rrf", None), "selection_metric", self.selection_metric)
                ),
                "validation_max_batches": int(max_batches),
                "validation_complete": int(max_batches) == 0,
                "search_exhaustive": bool(getattr(output, "rrf_search_exhaustive", False)),
                "grid_size": int(output.metrics.get("rrf_grid_size", 1.0)),
            }
        return output.metrics

    def _resolve_rrf_tune_every(self) -> int:
        if not self.rrf_enabled:
            return 0
        rrf = getattr(self.config, "rrf", None)
        legacy_value = getattr(rrf, "eval_every", 1) if rrf is not None else 1
        value = int(getattr(rrf, "tune_every", legacy_value) or 1) if rrf is not None else 1
        if value <= 0:
            raise ValueError(f"rrf.tune_every must be positive, got {value}")
        return value

    def _should_tune_rrf(self, epoch: int) -> bool:
        return self.rrf_enabled and int(epoch) % self.rrf_tune_every == 0

    def _compute_loss(
        self,
        output,
        batch: Mapping[str, Any] | None = None,
        debug_context: str = "",
    ) -> MoSIDLossOutput:
        self._validate_output_finite(output, debug_context=debug_context)
        rg_output = self._compute_generation_loss(
            output,
            logits_finite_verified=True,
        )
        raw_weight = self.raw_aux_weight
        raw_loss = rg_output.loss.detach().new_zeros(())
        diagnostics: dict[str, Tensor | float] = {
            "raw_nonfinite_count": 0.0,
        }
        if self.raw_training_enabled:
            if batch is None:
                raise ValueError("raw auxiliary training requires the input batch")
            raw_output, candidate_count, transfer_bytes, prefix_diagnostics = self._compute_raw_loss(
                output,
                batch,
            )
            self._validate_named_tensors_finite(
                {
                    "raw_loss": raw_output.loss,
                    "raw_aggregated_scores": raw_output.aggregated_scores,
                    "raw_modality_weights": raw_output.modality_weights,
                    **{
                        f"raw_scores[{branch}]": scores
                        for branch, scores in raw_output.scores.items()
                    },
                },
                debug_context=debug_context or "raw compatibility loss",
            )
            raw_loss = raw_output.loss
            batch_size = int(next(iter(output.hidden_states.values())).shape[0])
            diagnostics.update(
                self._raw_diagnostics(
                    raw_output,
                    batch_size=batch_size,
                    candidate_count=candidate_count,
                    transfer_bytes=transfer_bytes,
                    history_reliability=batch["history_reliability"],
                    prefix_diagnostics=prefix_diagnostics,
                )
            )
        memory_loss = rg_output.loss.detach().new_zeros(())
        total_loss = rg_output.loss + raw_weight * raw_loss
        if self.memory_orth_enabled:
            if batch is None or not isinstance(batch.get("history_presence"), Tensor):
                raise ValueError("MemoryOrth training requires batch history_presence")
            memory_loss = memory_orth_loss(
                output.context.token_hidden_states,
                output.context.attention_mask,
                batch["history_presence"],
                direct=self.memory_orth_direct,
                center=self.memory_orth_center,
                uniform_weights=self.memory_orth_uniform_weights,
            )
            self._validate_named_tensors_finite(
                {"memory_orth_loss": memory_loss},
                debug_context=debug_context or "MemoryOrth loss",
            )
            total_loss = total_loss + self.memory_orth_weight * memory_loss
        return MoSIDLossOutput(
            loss=total_loss,
            rg_loss=rg_output.loss,
            ce_loss=rg_output.ce_loss,
            raw_loss=raw_loss,
            memory_orth_loss=memory_loss,
            raw_weight=raw_weight,
            memory_orth_weight=self.memory_orth_weight,
            token_acc=rg_output.token_acc,
            valid_tokens=rg_output.valid_tokens,
            diagnostics=diagnostics,
        )

    def _compute_raw_loss(
        self,
        output,
        batch: Mapping[str, Any],
    ) -> tuple[RawCompatibilityOutput, int, int, dict[str, float]]:
        target_item = getattr(output, "target_item", None)
        if not isinstance(target_item, Tensor):
            raise ValueError("raw auxiliary training requires output.target_item")
        if self.raw_catalog is None:
            raise RuntimeError("raw auxiliary training catalog is unavailable")
        history_reliability = batch.get("history_reliability")
        if not isinstance(history_reliability, Mapping):
            raise ValueError("raw auxiliary training requires batch history_reliability")

        target_items_cpu = target_item.detach().cpu()
        positive_ids, target_columns = unique_target_candidates(target_items_cpu)
        if self.prefix_negative_sampler is None:
            raise RuntimeError("raw auxiliary training prefix-negative sampler is unavailable")
        sampling = self.prefix_negative_sampler.sample(target_items_cpu)
        shared_candidates = merge_shared_raw_candidates(
            positive_ids,
            sampling,
            effective_batch_size=self.effective_batch_size,
        )
        candidate_ids = shared_candidates.candidate_ids
        if self.raw_random_negatives:
            candidate_ids = random_replacement_candidates(
                positive_ids, int(candidate_ids.numel() - positive_ids.numel()),
                self.prefix_negative_sampler.num_items,
                int(self.config.train.seed), self.prefix_negative_sampler.epoch,
            )
        prefix_diagnostics: dict[str, float] = {
            "raw_positive_candidate_count": float(positive_ids.numel()),
            "raw_prefix_unique_negative_count": float(candidate_ids.numel() - positive_ids.numel()),
            "raw_prefix_deduplicated_count": float(shared_candidates.duplicates_skipped),
            "raw_prefix_cap_dropped_count": float(shared_candidates.cap_dropped),
        }
        for lcp_depth in range(len(sampling.requested_by_lcp)):
            prefix_diagnostics[f"raw_prefix_lcp_{lcp_depth}_requested"] = float(
                sampling.requested_by_lcp[lcp_depth]
            )
            prefix_diagnostics[f"raw_prefix_lcp_{lcp_depth}_available"] = float(
                sampling.available_by_lcp[lcp_depth]
            )
            prefix_diagnostics[f"raw_prefix_lcp_{lcp_depth}_selected"] = float(
                shared_candidates.selected_by_lcp[lcp_depth]
            )
        if self.raw_random_negatives:
            prefix_diagnostics = {
                key.replace("raw_prefix_", "raw_prefix_budget_"): value
                for key, value in prefix_diagnostics.items()
            }
            prefix_diagnostics["raw_random_unique_negative_count"] = float(
                candidate_ids.numel() - positive_ids.numel()
            )
        model = self.model.module if hasattr(self.model, "module") else self.model
        queries = model.raw_history_summaries(
            output.context,
            decoder_hidden_states=output.hidden_states,
        )
        device = queries[MPSID_INPUT_MODALITIES[0]].device
        candidate_raw: dict[str, Tensor] = {}
        candidate_available: dict[str, Tensor] = {}
        transfer_bytes = (
            target_columns.numel() * target_columns.element_size()
            if device.type != "cpu"
            else 0
        )
        for branch in MPSID_INPUT_MODALITIES:
            raw_cpu, available_cpu = self.raw_catalog.gather(branch, candidate_ids)
            candidate_raw[branch] = raw_cpu.to(device)
            candidate_available[branch] = available_cpu.to(device)
            if device.type != "cpu":
                transfer_bytes += raw_cpu.numel() * raw_cpu.element_size()
                transfer_bytes += available_cpu.numel() * available_cpu.element_size()

        raw_output = self._raw_scorer()(
            queries,
            candidate_raw,
            candidate_available,
            history_reliability,
            target_columns.to(device),
            temperature=self.raw_temperature,
        )
        return raw_output, int(candidate_ids.numel()), int(transfer_bytes), prefix_diagnostics

    @staticmethod
    def _raw_diagnostics(
        raw_output: RawCompatibilityOutput,
        batch_size: int,
        candidate_count: int,
        transfer_bytes: int,
        history_reliability: Mapping[str, Tensor],
        prefix_diagnostics: Mapping[str, float],
    ) -> dict[str, float]:
        packed = torch.stack(
            [raw_output.valid_candidate_mask.float().mean()]
            + [
                history_reliability[branch].float().mean()
                for branch in MPSID_INPUT_MODALITIES
            ]
            + [
                raw_output.modality_weights[index]
                for index in range(len(MPSID_INPUT_MODALITIES))
            ]
        ).detach().cpu().tolist()
        diagnostics = {
            "raw_candidate_count": float(candidate_count),
            "raw_valid_sample_rate": float(raw_output.valid_samples) / max(batch_size, 1),
            "raw_valid_candidate_rate": float(packed[0]),
            "raw_host_to_device_bytes": float(transfer_bytes),
            **{key: float(value) for key, value in prefix_diagnostics.items()},
        }
        branch_count = len(MPSID_INPUT_MODALITIES)
        for branch_index, branch in enumerate(MPSID_INPUT_MODALITIES):
            branch_diagnostics = raw_output.diagnostics[branch]
            prefix = f"raw_{branch}"
            diagnostics[f"{prefix}_reliability_mean"] = float(packed[1 + branch_index])
            diagnostics[f"{prefix}_modality_weight"] = float(
                packed[1 + branch_count + branch_index]
            )
            diagnostics[f"{prefix}_valid_sample_rate"] = (
                float(branch_diagnostics.valid_samples) / max(batch_size, 1)
            )
            diagnostics[f"{prefix}_valid_candidates"] = float(branch_diagnostics.valid_candidates)
            diagnostics[f"{prefix}_positive_margin"] = float(branch_diagnostics.positive_margin)
            diagnostics[f"{prefix}_top1"] = float(branch_diagnostics.top1)
        return diagnostics

    def _raw_scorer(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        scorer = getattr(model, "raw_compatibility_scorer", None)
        if scorer is None:
            raise ValueError("raw compatibility is enabled but the model has no scorer")
        return scorer

    def _raw_modality_weight_snapshot(self) -> dict[str, float]:
        model = self.model.module if hasattr(self.model, "module") else self.model
        scorer = getattr(model, "raw_compatibility_scorer", None)
        if scorer is None:
            return {}
        values = scorer.modality_weights().detach().cpu().tolist()
        return {
            branch: float(values[index])
            for index, branch in enumerate(MPSID_INPUT_MODALITIES)
        }

    def _compute_generation_loss(
        self,
        output,
        logits_finite_verified: bool = False,
    ) -> GenLossOutput:
        return weighted_autoregressive_ce_loss(
            output.logits,
            output.target_sid,
            output.target_available,
            branch_weights=self.branch_weights,
            ignore_index=output.loss_ignore_index,
            logits_finite_verified=logits_finite_verified,
        )

    def _validate_output_finite(self, output, debug_context: str = "") -> None:
        named_tensors = list(self._iter_output_debug_tensors(output))
        if self._all_tensors_finite(tensor for _name, tensor in named_tensors):
            return
        problems: list[str] = []
        for name, tensor in named_tensors:
            summary = self._nonfinite_tensor_summary(name, tensor)
            if summary is not None:
                problems.append(summary)
        if not problems:
            return

        location = f" ({debug_context})" if debug_context else ""
        message = [
            f"MoSID output contains NaN/Inf before generation loss{location}.",
            "First non-finite tensors:",
        ]
        message.extend(f"  - {summary}" for summary in problems[:12])
        if len(problems) > 12:
            message.append(f"  - ... {len(problems) - 12} more tensor(s)")

        parameter_problems = self._nonfinite_model_parameter_summaries(limit=5)
        if parameter_problems:
            message.append("Non-finite model parameters at failure time:")
            message.extend(f"  - {summary}" for summary in parameter_problems)
        else:
            message.append("Model parameters were finite at failure time.")
        raise ValueError("\n".join(message))

    def _validate_named_tensors_finite(self, tensors: Mapping[str, Tensor], debug_context: str = "") -> None:
        if self._all_tensors_finite(tensors.values()):
            return
        problems = [
            summary
            for name, tensor in tensors.items()
            if (summary := self._nonfinite_tensor_summary(name, tensor)) is not None
        ]
        if not problems:
            return
        location = f" ({debug_context})" if debug_context else ""
        raise ValueError(
            "MoSID tensors contain NaN/Inf"
            f"{location}: "
            + "; ".join(problems[:8])
        )

    def _assert_finite_model_parameters(self, context: str) -> None:
        problems = self._nonfinite_model_parameter_summaries(limit=12)
        if problems:
            raise FloatingPointError(
                f"MoSID model parameters contain NaN/Inf {context}: "
                + "; ".join(problems)
            )

    def _nonfinite_model_parameter_summaries(self, limit: int | None = None) -> list[str]:
        named_parameters = list(self.model.named_parameters())
        if self._all_tensors_finite(parameter for _name, parameter in named_parameters):
            return []
        problems: list[str] = []
        for name, parameter in named_parameters:
            summary = self._nonfinite_tensor_summary(f"parameter[{name}]", parameter)
            if summary is not None:
                problems.append(summary)
                if limit is not None and len(problems) >= limit:
                    return problems
        return problems

    @staticmethod
    def _all_tensors_finite(tensors: Iterable[Tensor]) -> bool:
        """Check all floating tensors with one host synchronization per device."""
        checks_by_device: dict[torch.device, list[Tensor]] = {}
        for tensor in tensors:
            if not isinstance(tensor, Tensor):
                continue
            if not (tensor.is_floating_point() or tensor.is_complex()):
                continue
            check = torch.isfinite(tensor.detach()).all()
            checks_by_device.setdefault(check.device, []).append(check)
        return all(
            bool(torch.stack(checks).all().item())
            for checks in checks_by_device.values()
        )

    @staticmethod
    def _assert_finite_scalar(value: Tensor, context: str) -> None:
        tensor = value.detach().float()
        if torch.isfinite(tensor).all().item():
            return
        raise FloatingPointError(f"MoSID detected non-finite scalar {context}: value={tensor.item()}")

    def _iter_output_debug_tensors(self, output):
        yield from self._iter_named_tensors("logits", getattr(output, "logits", None))
        yield from self._iter_named_tensors("hidden_states", getattr(output, "hidden_states", None))
        context = getattr(output, "context", None)
        if context is None:
            return
        for attr in (
            "token_hidden_states",
        ):
            yield from self._iter_named_tensors(f"context.{attr}", getattr(context, attr, None))

    def _iter_named_tensors(self, prefix: str, value):
        if isinstance(value, Tensor):
            yield prefix, value
            return
        if isinstance(value, Mapping):
            for key in sorted(value.keys(), key=str):
                yield from self._iter_named_tensors(f"{prefix}[{key}]", value[key])
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from self._iter_named_tensors(f"{prefix}[{index}]", item)

    @staticmethod
    def _nonfinite_tensor_summary(name: str, tensor: Tensor) -> str | None:
        if not isinstance(tensor, Tensor):
            return None
        if not (tensor.is_floating_point() or tensor.is_complex()):
            return None

        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        if finite_mask.all().item():
            return None

        bad_mask = ~finite_mask
        bad_count = int(bad_mask.sum().item())
        nan_count = int(torch.isnan(detached).sum().item())
        inf_count = int(torch.isinf(detached).sum().item())
        parts = [
            f"{name}: shape={tuple(detached.shape)}",
            f"dtype={detached.dtype}",
            f"device={detached.device}",
            f"bad={bad_count}",
            f"nan={nan_count}",
            f"inf={inf_count}",
        ]
        if finite_mask.any().item():
            finite_values = detached[finite_mask]
            if finite_values.is_complex():
                finite_abs = finite_values.abs()
                parts.append(f"finite_abs_min={float(finite_abs.min().item()):.6g}")
                parts.append(f"finite_abs_max={float(finite_abs.max().item()):.6g}")
            else:
                parts.append(f"finite_min={float(finite_values.min().item()):.6g}")
                parts.append(f"finite_max={float(finite_values.max().item()):.6g}")
        else:
            parts.append("no_finite_values=true")
        return ", ".join(parts)

    @staticmethod
    def _loss_output_scalars(loss_output: MoSIDLossOutput) -> tuple[float, float, float, float, float]:
        values = (
            loss_output.loss,
            loss_output.rg_loss,
            loss_output.ce_loss,
            loss_output.raw_loss,
            loss_output.memory_orth_loss,
        )
        unique_values: list[Tensor] = []
        index_by_identity: dict[int, int] = {}
        value_indices: list[int] = []
        for value in values:
            identifier = id(value)
            if identifier not in index_by_identity:
                index_by_identity[identifier] = len(unique_values)
                unique_values.append(value)
            value_indices.append(index_by_identity[identifier])

        devices = {value.device for value in unique_values}
        if len(devices) != 1:
            return tuple(float(value.item()) for value in values)
        packed = torch.stack([value.detach().reshape(()) for value in unique_values]).cpu().tolist()
        return tuple(float(packed[index]) for index in value_indices)

    def _accumulate(
        self,
        accum: dict[str, float],
        loss_output: MoSIDLossOutput,
        grad_norm: float | None,
        loss_scalars: tuple[float, float, float, float, float] | None = None,
    ) -> None:
        values = self._loss_output_scalars(loss_output) if loss_scalars is None else loss_scalars
        accum["loss"] += values[0]
        accum["rg_loss"] += values[1]
        accum["ce_loss"] += values[2]
        accum["raw_loss"] += values[3]
        accum["memory_orth_loss"] += values[4]
        accum["raw_weight"] += float(loss_output.raw_weight)
        accum["memory_orth_weight"] += float(loss_output.memory_orth_weight)
        accum["token_acc"] += float(loss_output.token_acc)
        accum["valid_tokens"] += float(loss_output.valid_tokens)
        for key, value in loss_output.diagnostics.items():
            accum[f"diag::{key}"] += float(value.item() if isinstance(value, Tensor) else value)
            accum[f"diag::{key}::count"] += 1.0
        if grad_norm is not None:
            accum["grad_norm"] += float(grad_norm)
            accum["grad_norm_count"] += 1.0

    def _average_metrics(self, accum: dict[str, float], n_batches: int) -> dict[str, float]:
        denom = max(n_batches, 1)
        grad_denom = max(accum["grad_norm_count"], 1.0)
        metrics = {
            "loss": accum["loss"] / denom,
            "total_loss": accum["loss"] / denom,
            "rg_loss": accum["rg_loss"] / denom,
            "gen_loss": accum["rg_loss"] / denom,
            "ce_loss": accum["ce_loss"] / denom,
            "raw_loss": accum["raw_loss"] / denom,
            "raw_weight": accum["raw_weight"] / denom,
            "memory_orth_loss": accum["memory_orth_loss"] / denom,
            "memory_orth_weight": accum["memory_orth_weight"] / denom,
            "token_acc": accum["token_acc"] / denom,
            "valid_tokens": accum["valid_tokens"],
            "grad_norm": accum["grad_norm"] / grad_denom,
            "n_batches": float(n_batches),
            "samples": accum["samples"],
        }
        for key, value in list(accum.items()):
            if not key.startswith("diag::") or key.endswith("::count"):
                continue
            metric_name = key.split("::", 1)[1]
            metric_denom = max(accum[f"{key}::count"], 1.0)
            metrics[metric_name] = value / metric_denom
        return metrics

    def _accum_window_size(self, batch_idx: int, total_batches: int | None) -> int:
        if total_batches is None:
            return self.grad_accum_steps
        window_start = batch_idx - (batch_idx % self.grad_accum_steps)
        return min(self.grad_accum_steps, total_batches - window_start)

    def _format_branch_weight_summary(self) -> str:
        pairs: list[tuple[str, str]] = [
            ("active_channels", ",".join(self.active_channels)),
            ("weight_source", "loss.branch_weights"),
        ]
        for channel in self.active_channels:
            weight = self.branch_weights.get(channel)
            value = "missing" if weight is None else f"{weight:.4g}"
            pairs.append((f"w_{channel}", value))
        return "\n" + format_horizontal_metrics_table(
            "MoSID Branch Loss Weights",
            pairs,
            pairs_per_row=3,
        )

    def _checkpoint_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        valid_metrics: dict[str, float] | None = None,
        selection_score: float | None = None,
        rrf_eval_round: int | None = None,
        rrf_tuned_this_eval: bool | None = None,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "baseline": self.baseline_name,
            "selection_protocol": "beam validation item ranking",
            "selection_metric": self.selection_metric,
            "active_channels": list(self.active_channels),
            "rrf_enabled": bool(self.rrf_enabled),
            "loss_branch_weights": dict(self.branch_weights),
            "epoch": int(epoch),
        }
        if self.rrf_enabled:
            metrics["rrf_tune_every"] = int(self.rrf_tune_every)
            if self.current_rrf_settings is None:
                raise RuntimeError("multi-branch checkpoint metrics require a selected RRF setting")
            metrics.update(rrf_settings_metrics(self.current_rrf_settings))
            metrics["rrf_selection_source"] = str(self.current_rrf_selection["source"])
            metrics["rrf_validation_complete"] = bool(
                self.current_rrf_selection["validation_complete"]
            )
            metrics["rrf_selection_exhaustive"] = bool(
                self.current_rrf_selection["search_exhaustive"]
            )
            if rrf_eval_round is not None:
                metrics["rrf_eval_round"] = int(rrf_eval_round)
            if rrf_tuned_this_eval is not None:
                metrics["rrf_tuned_this_eval"] = bool(rrf_tuned_this_eval)
        if selection_score is not None:
            metrics["selection_score"] = float(selection_score)
        metrics.update({f"train/{key}": value for key, value in train_metrics.items()})
        if valid_metrics is not None:
            metrics.update({f"valid/{key}": value for key, value in valid_metrics.items()})
        return metrics

    def _format_train_summary(self, epoch: int, metrics: dict[str, float], lr: float, elapsed: float) -> str:
        pairs = [
            ("total", f"{metrics['total_loss']:.6f}"),
            ("gen", f"{metrics['gen_loss']:.6f}"),
            ("raw", f"{metrics['raw_loss']:.6f}"),
            ("mem", f"{metrics['memory_orth_loss']:.6f}"),
            ("tok_acc", f"{metrics['token_acc']:.4f}"),
            ("grad", f"{metrics['grad_norm']:.4f}"),
            ("lr", f"{lr:.2e}"),
            ("time", f"{elapsed:.1f}s"),
            ("samples/s", f"{metrics.get('samples_per_second', 0.0):.1f}"),
            ("peak_mb", f"{metrics.get('peak_memory_mb', 0.0):.1f}"),
        ]
        if self.raw_training_enabled:
            pairs.extend(
                [
                    ("raw_candidates", f"{metrics.get('raw_candidate_count', 0.0):.1f}"),
                    ("raw_valid", f"{metrics.get('raw_valid_sample_rate', 0.0):.4f}"),
                    ("random_neg" if self.raw_random_negatives else "prefix_neg",
                     f"{metrics.get('raw_random_unique_negative_count' if self.raw_random_negatives else 'raw_prefix_unique_negative_count', 0.0):.1f}"),
                ]
            )
            for branch in self.active_channels:
                prefix = f"raw_{branch}"
                pairs.extend(
                    [
                        (f"{branch}_mod_w", f"{metrics.get(f'{prefix}_modality_weight', 0.0):.4f}"),
                        (f"{branch}_top1", f"{metrics.get(f'{prefix}_top1', 0.0):.4f}"),
                        (f"{branch}_margin", f"{metrics.get(f'{prefix}_positive_margin', 0.0):.4f}"),
                    ]
                )
        return "\n" + format_horizontal_metrics_table(
            f"Train Epoch {epoch + 1:03d}/{self.max_epochs:03d}",
            pairs,
            pairs_per_row=4,
        )

    def _format_beam_summary(self, epoch: int, metrics: dict[str, float], selection_score: float) -> str:
        return "\n" + format_beam_metrics_summary(
            title=f"Beam Metrics Epoch {epoch + 1:03d}",
            metrics=metrics,
            selection_metric=self.selection_metric,
            selection_score=selection_score,
            top_k=get_top_k(self.config),
            active_channels=self.active_channels,
            rrf_enabled=self.rrf_enabled,
        )

    def _log_tensorboard(self, metrics: dict[str, float], epoch: int, prefix: str) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_scalar(f"{prefix}{key}", float(value), epoch)

    def _save_config_snapshot(self) -> None:
        import yaml

        path = os.path.join(self.run_dir, "config.yaml")
        with open(path, "w", encoding="utf-8") as file:
            yaml.dump(self.config.to_dict(), file, default_flow_style=False, allow_unicode=True)

    def _save_artifact_snapshot(self, best_metrics: dict[str, Any] | None = None) -> None:
        save_json(self._build_artifact_snapshot(best_metrics), os.path.join(self.run_dir, "artifacts.json"))

    def _build_artifact_snapshot(self, best_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        channels = list(getattr(self.config.model, "active_channels", ["collab"]))
        data_dir = self._data_dir()
        artifacts = {
            "sid_codes": {
                channel: str(data_dir / sid_codes_filename(mosid_sid_artifact_channel(self.config, channel)))
                for channel in channels
            },
            "codebook_weights": {
                channel: str(
                    data_dir / codebook_weights_filename(mosid_sid_artifact_channel(self.config, channel))
                )
                for channel in channels
            },
        }
        return {
            "mosid_checkpoint_schema_version": MOSID_CHECKPOINT_SCHEMA_VERSION,
            "baseline": self.baseline_name,
            "active_channels": channels,
            "selection_metric": self.selection_metric,
            "seed": getattr(self.config.train, "seed", None),
            "data": {
                "base_path": getattr(self.config.data, "base_path", None),
                "dataset": getattr(self.config.data, "dataset", None),
                "mode": getattr(self.config.data, "mode", None),
                "data_dir": str(data_dir),
            },
            "artifacts": artifacts,
            "beam": {
                "valid_beam_size": get_valid_beam_size(self.config),
                "test_beam_size": get_test_beam_size(self.config),
                "top_k": get_top_k(self.config),
                "max_batches": get_max_batches(self.config),
            },
            "rrf": self._plain_dict(getattr(self.config, "rrf", {})) if self.rrf_enabled else None,
            "rrf_enabled": bool(self.rrf_enabled),
            "loss": {
                "branch_weights": dict(self.branch_weights),
                "raw_aux_weight": self.raw_aux_weight,
                "raw_random_negatives": self.raw_random_negatives,
                "raw_temperature": self.raw_temperature,
                "raw_use_history_reliability": bool(getattr(self.config.loss, "raw_use_history_reliability", True)),
                "memory_orth_weight": self.memory_orth_weight,
                "memory_orth_direct": self.memory_orth_direct,
                "memory_orth_center": self.memory_orth_center,
                "memory_orth_uniform_weights": self.memory_orth_uniform_weights,
                "raw_modality_weights": self._raw_modality_weight_snapshot(),
                "raw_prefix_hard_negative_counts_by_lcp": list(
                    getattr(
                        self.config.loss,
                        "raw_prefix_hard_negative_counts_by_lcp",
                        [],
                    )
                ),
            },
            "selected_rrf": self._selected_rrf(best_metrics or {}),
            "checkpoints": {
                "best": self.best_ckpt_path,
                "last": self.last_ckpt_path,
            },
        }

    def _data_dir(self) -> Path:
        raw = getattr(self.config.data, "_data_dir", None)
        if raw:
            return Path(raw)
        return Path(self.config.data.base_path) / str(self.config.data.dataset) / str(self.config.data.mode)

    def _plain_dict(self, value) -> dict[str, Any]:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        return {}

    def _rrf_sample_count(self) -> int:
        if not self.rrf_enabled:
            return 0
        rrf = getattr(self.config, "rrf", None)
        return int(getattr(rrf, "sample_count", 0) or 0) if rrf is not None else 0

    def _selected_rrf(self, metrics: dict[str, Any]) -> dict[str, float]:
        if not self.rrf_enabled:
            return {}
        keys = [
            "rrf_k",
            "rrf_tuned",
            "rrf_grid_size",
            "rrf_rank_fusion_weight",
            "rrf_raw_compat_weight",
        ]
        selected: dict[str, float] = {}
        for key in keys:
            value = metrics.get(key, metrics.get(f"valid/{key}"))
            if isinstance(value, (int, float)):
                selected[key] = float(value)
        return selected

    def _move_batch(self, value):
        if isinstance(value, Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self._move_batch(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self._move_batch(child) for child in value]
        if isinstance(value, tuple):
            return tuple(self._move_batch(child) for child in value)
        return value

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        target_item = batch.get("target_item")
        if isinstance(target_item, Tensor) and target_item.dim() >= 1:
            return int(target_item.shape[0])
        attention_mask = batch.get("attention_mask")
        if isinstance(attention_mask, Tensor) and attention_mask.dim() >= 1:
            return int(attention_mask.shape[0])
        raise ValueError("cannot infer MoSID batch size")
