"""Evaluate MoSID Stage 2 checkpoints with Trie-constrained beam search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import (
    DotDict,
    load_from_checkpoint_or_config,
    raw_compat_inference_is_enabled,
)
from data.mosid_dataset import MoSIDSequenceDataset
from data.raw_embedding_catalog import RawEmbeddingCatalog
from evaluate.beam_eval import (
    RRFSettings,
    evaluate_beam,
    format_beam_metrics_summary,
    get_merge_branches,
    get_test_beam_size,
    get_top_k,
    is_single_channel_config,
    resolve_beam_metric_score,
    rrf_settings_from_checkpoint_metadata,
    validate_mosid_checkpoint_schema,
)
from inference.trie import SIDTrie, load_sid_trie_from_artifacts
from models.mosid import MoSIDModel
from utils.data import MPSID_INPUT_MODALITIES, mosid_sid_artifact_channel
from utils.io import load_checkpoint_payload, restore_checkpoint_payload
from utils.logger import Logger
from utils.seed import set_device, set_seed


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def build_mosid_tries(config: DotDict):
    data_dir = Path(config.data._data_dir)
    tries = {}
    tries_by_artifact = {}
    active_channels = [str(channel) for channel in config.model.active_channels]
    for channel in active_channels:
        branch = str(channel)
        artifact_channel = mosid_sid_artifact_channel(config, branch)
        if artifact_channel not in tries_by_artifact:
            tries_by_artifact[artifact_channel] = load_sid_trie_from_artifacts(
                data_dir,
                artifact_channel,
                token_offset=-1,
            )
        tries[branch] = tries_by_artifact[artifact_channel]
    if len(tries) == 1:
        return next(iter(tries.values()))
    return tries


def load_mosid_checkpoint_model(
    checkpoint: str | Path,
    config: DotDict,
    device: torch.device,
) -> tuple[MoSIDModel, dict]:
    model, meta, _ = _load_mosid_checkpoint_resources(checkpoint, config, device)
    return model, meta


def _load_mosid_checkpoint_resources(
    checkpoint: str | Path,
    config: DotDict,
    device: torch.device,
    include_validation_choices: bool = True,
    require_raw_catalog: bool = False,
) -> tuple[MoSIDModel, dict, RawEmbeddingCatalog | None]:
    payload = load_checkpoint_payload(checkpoint, device=device)
    validate_mosid_checkpoint_schema(payload)
    catalog_required = require_raw_catalog or _checkpoint_uses_raw_fusion(payload)
    if include_validation_choices:
        catalog_required = catalog_required or raw_compat_inference_is_enabled(config)
    elif not _checkpoint_has_rrf_setting(payload):
        rrf = getattr(config, "rrf", None)
        catalog_required = catalog_required or (
            rrf is not None and float(getattr(rrf, "raw_compat_weight", 0.0) or 0.0) > 0.0
        )
    raw_catalog = RawEmbeddingCatalog.from_config(config) if catalog_required else None
    loss = getattr(config, "loss", None)
    raw_training_checkpoint = (
        loss is not None and float(getattr(loss, "raw_aux_weight", 0.0) or 0.0) > 0.0
    )
    model_raw_enabled = bool(
        catalog_required
        or raw_training_checkpoint
        or _checkpoint_has_raw_scorer(payload)
    )
    raw_dimensions = None
    if model_raw_enabled:
        raw_dimensions = (
            raw_catalog.dimensions
            if raw_catalog is not None
            else _raw_dimensions_from_checkpoint(payload)
        )
    model = MoSIDModel.from_config(
        config,
        raw_dimensions=raw_dimensions,
        enable_raw_compatibility=model_raw_enabled,
    ).to(device)
    meta = restore_checkpoint_payload(payload, model)
    return model, meta, raw_catalog


def _checkpoint_uses_raw_fusion(payload: dict) -> bool:
    extra_state = payload.get("extra_state", {})
    if isinstance(extra_state, dict):
        selected_rrf = extra_state.get("selected_rrf")
        if isinstance(selected_rrf, dict):
            if float(selected_rrf.get("raw_compat_weight", 0.0) or 0.0) > 0.0:
                return True
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    for key in ("rrf_raw_compat_weight", "valid/rrf_raw_compat_weight"):
        if float(metrics.get(key, 0.0) or 0.0) > 0.0:
            return True
    return False


def _checkpoint_has_rrf_setting(payload: dict) -> bool:
    extra_state = payload.get("extra_state", {})
    if isinstance(extra_state, dict) and isinstance(extra_state.get("selected_rrf"), dict):
        return True
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    return any(key in metrics for key in ("rrf_k", "valid/rrf_k"))


def _raw_dimensions_from_checkpoint(payload: dict) -> dict[str, int]:
    state = payload.get("model_state_dict", {})
    dimensions: dict[str, int] = {}
    for branch in MPSID_INPUT_MODALITIES:
        suffix = f"raw_compatibility_scorer.projectors.{branch}.weight"
        matches = [value for key, value in state.items() if str(key).removeprefix("module.") == suffix]
        if len(matches) != 1 or not isinstance(matches[0], torch.Tensor) or matches[0].dim() != 2:
            raise ValueError(
                f"raw-enabled checkpoint must contain one rank-2 projector weight for {branch!r}"
            )
        dimensions[branch] = int(matches[0].shape[1])
    return dimensions


def _checkpoint_has_raw_scorer(payload: dict) -> bool:
    state = payload.get("model_state_dict", {})
    return any(
        str(key).removeprefix("module.").startswith("raw_compatibility_scorer.")
        for key in state
    )


def evaluate_mosid_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | None = None,
    split: str = "test",
    override_args: list[str] | None = None,
) -> dict[str, float]:
    checkpoint = resolve_checkpoint_path(checkpoint_path)
    config = load_from_checkpoint_or_config(
        config_path,
        checkpoint,
        override_args,
        fallback_config_path=str(Path("configs") / "mosid.yaml"),
    )
    set_seed(config.train.seed)
    device = set_device(config.train.gpu_id)

    model, meta, raw_catalog = _load_mosid_checkpoint_resources(
        checkpoint,
        config,
        device,
        include_validation_choices=(
            split == "valid" or _valid_tuning_before_test(config, split)
        ),
    )
    trie = build_mosid_tries(config)
    rrf_settings = None if is_single_channel_config(config) else _checkpoint_rrf_settings(meta, config)
    output, _ = _evaluate_requested_split(
        model,
        trie,
        config,
        device,
        split=split,
        rrf_settings=rrf_settings,
        raw_catalog=raw_catalog,
    )
    metrics = dict(output.metrics)
    metrics["checkpoint_epoch"] = float(meta["epoch"] + 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MoSID Stage 2 with Trie-beam item ranking")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or a run directory containing best.pt")
    parser.add_argument("--config", default=None, help="Optional config path; defaults to checkpoint run config")
    parser.add_argument("--split", default="test", choices=["valid", "test"], help="Dataset split to evaluate")
    parser.add_argument("--raw-weight-sweep", nargs="+", type=float,
                        help="Evaluate each fixed raw weight with rank weight 1 and config rrf.k")
    parser.add_argument("--raw-sweep-output", type=Path)
    known, unknown = parser.parse_known_args()
    if known.raw_weight_sweep is not None and (known.split != "test" or known.raw_sweep_output is None):
        parser.error("--raw-weight-sweep requires --split test and --raw-sweep-output")

    checkpoint = resolve_checkpoint_path(known.checkpoint)
    config = load_from_checkpoint_or_config(
        known.config,
        checkpoint,
        unknown,
        fallback_config_path=str(Path("configs") / "mosid.yaml"),
    )
    set_seed(config.train.seed)
    device = set_device(config.train.gpu_id)
    output_dir = checkpoint.parent / f"{known.split}_beam_eval"
    if known.raw_weight_sweep is not None:
        output_dir = known.raw_sweep_output.parent / known.raw_sweep_output.stem
    logger = Logger(output_dir, name="mosid_eval", tensorboard=False)
    try:
        logger.info(f"checkpoint: {checkpoint}")
        logger.info(f"split: {known.split}")
        model, meta, raw_catalog = _load_mosid_checkpoint_resources(
            checkpoint,
            config,
            device,
            include_validation_choices=(
                known.split == "valid" or _valid_tuning_before_test(config, known.split)
            ),
            require_raw_catalog=known.raw_weight_sweep is not None,
        )
        trie = build_mosid_tries(config)
        if known.raw_weight_sweep is not None:
            settings = [RRFSettings(k=float(config.rrf.k), rank_fusion_weight=1.0,
                                    raw_compat_weight=weight) for weight in known.raw_weight_sweep]
            logger.info("Fixed raw weight sensitivity: one candidate/raw-score pass; no RRF selection")
            output = evaluate_beam(
                model, _build_eval_loader(config, "test"), trie, config, device,
                split="test", raw_catalog=raw_catalog, beam_size=get_test_beam_size(config),
                fixed_rrf_settings=settings,
            )
            for setting, metrics in zip(settings, output.fixed_rrf_metrics):
                _log_rrf_settings(logger, "fixed RRF", setting)
                logger.info("\n" + _format_beam_metrics("MoSID fixed-weight test Metrics", metrics, config))
            result = {"checkpoint": str(checkpoint), "stage2_seed": int(config.train.seed),
                      "beam_size": get_test_beam_size(config), "results": output.fixed_rrf_metrics}
            path = known.raw_sweep_output
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
            return
        rrf_settings = None if is_single_channel_config(config) else _checkpoint_rrf_settings(meta, config)
        if is_single_channel_config(config):
            logger.info("single-channel evaluation: direct Trie beam ranking; RRF disabled")
        elif rrf_settings is not None:
            use_valid_tuning = _valid_tuning_before_test(config, known.split)
            prefix = "loaded checkpoint RRF as tune base" if (
                known.split == "valid" or use_valid_tuning
            ) else "loaded checkpoint-selected RRF"
            _log_rrf_settings(logger, prefix, rrf_settings)
        else:
            logger.info("checkpoint RRF not found; using config rrf defaults as tune base")
        output, selection_output = _evaluate_requested_split(
            model,
            trie,
            config,
            device,
            split=known.split,
            rrf_settings=rrf_settings,
            logger=logger,
            raw_catalog=raw_catalog,
        )
        if selection_output is not None:
            selection_metrics = dict(selection_output.metrics)
            logger.info(
                "\n"
                + _format_beam_metrics(
                    "MoSID valid RRF Selection Metrics",
                    selection_metrics,
                    config,
                )
            )
        metrics = dict(output.metrics)
        logger.info(
            "\n"
            + _format_beam_metrics(
                f"MoSID {known.split} Trie-Beam Metrics",
                metrics,
                config,
            )
        )
    finally:
        logger.close()


def _evaluation_batch_size(config: DotDict) -> int:
    if "evaluation" in config and "batch_size" in config.evaluation:
        return int(config.evaluation.batch_size)
    if "eval" in config and "batch_size" in config.eval:
        return int(config.eval.batch_size)
    return int(config.train.batch_size)


def _build_eval_loader(
    config: DotDict,
    split: str,
) -> DataLoader:
    dataset = MoSIDSequenceDataset(config, split=split)
    return DataLoader(
        dataset,
        batch_size=_evaluation_batch_size(config),
        shuffle=False,
        drop_last=False,
        num_workers=int(config.train.num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate_split(
    model: MoSIDModel,
    trie: SIDTrie | dict[str, SIDTrie],
    config: DotDict,
    device: torch.device,
    split: str,
    rrf_settings: RRFSettings | None,
    tune_rrf: bool = False,
    logger: Logger | None = None,
    raw_catalog: RawEmbeddingCatalog | None = None,
):
    if tune_rrf and split != "valid":
        raise ValueError(
            "RRF tuning is validation-only; test must use validation- or checkpoint-selected settings"
        )
    loader = _build_eval_loader(config, split)
    should_tune = bool(tune_rrf) and _rrf_gs_on_valid(config)
    if should_tune and logger is not None:
        logger.info(
            f"tuning RRF on {split}: exhaustive grid search, metric={_rrf_selection_metric(config)!r}"
        )
    output = evaluate_beam(
        model,
        loader,
        trie,
        config,
        device,
        split=split,
        rrf_settings=rrf_settings,
        tune_rrf=should_tune,
        tune_metric=_rrf_selection_metric(config),
        tune_seed=getattr(config.train, "seed", None),
        rrf_exhaustive=should_tune,
        raw_catalog=raw_catalog,
        beam_size=get_test_beam_size(config),
    )
    if should_tune and logger is not None and output.rrf_settings is not None:
        _log_rrf_settings(logger, f"selected {split} RRF", output.rrf_settings)
    return output


def _evaluate_requested_split(
    model: MoSIDModel,
    trie: SIDTrie | dict[str, SIDTrie],
    config: DotDict,
    device: torch.device,
    split: str,
    rrf_settings: RRFSettings | None,
    logger: Logger | None = None,
    raw_catalog: RawEmbeddingCatalog | None = None,
):
    """Evaluate one requested split, optionally selecting test RRF settings on valid first."""
    if _valid_tuning_before_test(config, split):
        if logger is not None:
            logger.info(
                "rrf.gs_on_valid=true: selecting RRF parameters on valid before test"
            )
        validation_output = _evaluate_split(
            model,
            trie,
            config,
            device,
            split="valid",
            rrf_settings=rrf_settings,
            tune_rrf=True,
            logger=logger,
            raw_catalog=raw_catalog,
        )
        selected_settings = validation_output.rrf_settings
        if selected_settings is None:
            raise RuntimeError("validation RRF search did not return selected settings")
        if logger is not None:
            logger.info(
                "test evaluation uses valid-selected RRF; test labels are not used for tuning"
            )
        test_output = _evaluate_split(
            model,
            trie,
            config,
            device,
            split="test",
            rrf_settings=selected_settings,
            tune_rrf=False,
            logger=logger,
            raw_catalog=raw_catalog,
        )
        return test_output, validation_output

    if split == "test" and not is_single_channel_config(config) and logger is not None:
        logger.info(
            "rrf.gs_on_valid=false: test uses checkpoint-selected RRF without validation tuning"
        )
    output = _evaluate_split(
        model,
        trie,
        config,
        device,
        split=split,
        rrf_settings=rrf_settings,
        tune_rrf=split == "valid",
        logger=logger,
        raw_catalog=raw_catalog,
    )
    return output, None


def _checkpoint_rrf_settings(meta: dict, config: DotDict):
    if is_single_channel_config(config):
        return None
    return rrf_settings_from_checkpoint_metadata(meta)


def _rrf_gs_on_valid(config: DotDict) -> bool:
    if is_single_channel_config(config):
        return False
    rrf = getattr(config, "rrf", None)
    if rrf is None:
        return False
    return bool(getattr(rrf, "gs_on_valid", False))


def _valid_tuning_before_test(config: DotDict, split: str) -> bool:
    return split == "test" and _rrf_gs_on_valid(config)


def _rrf_selection_metric(config: DotDict) -> str:
    return str(getattr(getattr(config, "rrf", None), "selection_metric", "beam_ndcg@10"))


def _log_rrf_settings(logger: Logger, prefix: str, settings: RRFSettings) -> None:
    logger.info(
        f"{prefix}: k={settings.k:.3f}, rank_weight={settings.rank_fusion_weight:.3f}, "
        f"raw_weight={settings.raw_compat_weight:.3f}"
    )


def _format_beam_metrics(title: str, metrics: dict[str, float], config: DotDict) -> str:
    selection_metric = _rrf_selection_metric(config)
    return format_beam_metrics_summary(
        title=title,
        metrics=metrics,
        selection_metric=selection_metric,
        selection_score=resolve_beam_metric_score(metrics, selection_metric),
        top_k=get_top_k(config),
        active_channels=_active_branches(config),
        rrf_enabled=not is_single_channel_config(config) and "rrf" in config,
    )


def _active_branches(config: DotDict) -> list[str]:
    merge_branches = get_merge_branches(config)
    if merge_branches is not None:
        return merge_branches
    return [str(channel) for channel in getattr(config.model, "active_channels", [])]


if __name__ == "__main__":
    main()
