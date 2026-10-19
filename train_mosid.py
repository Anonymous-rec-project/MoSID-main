"""Train the MoSID Stage 2 encoder-decoder backbone."""
from __future__ import annotations

from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from configs.config import build_run_dir, parse_args_and_load, raw_compat_is_enabled
from data.mosid_dataset import MoSIDSequenceDataset
from data.prefix_hard_negatives import PrefixHardNegativeSampler, SeparatePrefixHardNegativeSampler
from data.raw_embedding_catalog import RawEmbeddingCatalog
from inference.trie import load_sid_trie_from_artifacts
from models.mosid import MoSIDModel
from trainers.mosid_trainer import MoSIDTrainer
from utils.data import mosid_sid_artifact_channel, sid_codes_filename
from utils.logger import Logger
from utils.seed import set_device, set_seed


def main() -> None:
    config = parse_args_and_load("Train the MoSID Stage 2 encoder-decoder backbone")

    suffix = "_".join(config.model.active_channels)
    run_dir = build_run_dir(config, stage="mosid", suffix=suffix)
    run_dir_file = str(getattr(config.train, "run_dir_file", "") or "")
    if run_dir_file:
        Path(run_dir_file).write_text(str(run_dir), encoding="utf-8")
    if (config.model.sid_artifact_mode == "unified" or getattr(config.train, "seeded_run", False)) and (Path(run_dir) / "best.pt").is_file():
        print(f"Reusing Stage 2 checkpoint: {Path(run_dir) / 'best.pt'}")
        return
    set_seed(config.train.seed)
    device = set_device(config.train.gpu_id)
    training_started_at = time.perf_counter()
    logger = Logger(run_dir, name="mosid_train", tensorboard=config.train.tensorboard)
    logger.training_started_at = training_started_at
    try:
        logger.info(f"Config: {config}")
        logger.info(f"Device: {device}")

        train_dataset = MoSIDSequenceDataset(config, split="train")
        valid_dataset = MoSIDSequenceDataset(config, split="valid")
        train_loader = _build_loader(config, train_dataset, shuffle=True)
        valid_loader = _build_loader(config, valid_dataset, shuffle=False, batch_size=int(config.evaluation.batch_size))
        beam_trie = build_mosid_tries(config)
        logger.info(
            "loaded SID artifacts: "
            + ", ".join(
                f"\n{channel}={path}"
                for channel, path in mosid_sid_artifact_paths(config).items()
            )
        )
        logger.info(f"Training samples: {len(train_dataset)}  Validation samples: {len(valid_dataset)}")

        raw_catalog = RawEmbeddingCatalog.from_config(config) if raw_compat_is_enabled(config) else None
        if raw_catalog is not None:
            logger.info(f"raw compatibility catalog dimensions: {raw_catalog.dimensions}")
        prefix_negative_sampler = build_prefix_negative_sampler(config, train_dataset)
        if prefix_negative_sampler is not None:
            logger.info(
                "prefix hard-negative index: "
                f"items={prefix_negative_sampler.num_items}, "
                f"sid_depth={prefix_negative_sampler.sid_depth}, "
                f"counts_by_lcp={list(prefix_negative_sampler.counts_by_lcp)}, "
                "effective_batch_negative_budget="
                f"{int(config.train.batch_size) * int(config.train.grad_accum_steps)}"
            )
        model = MoSIDModel.from_config(
            config,
            raw_dimensions=None if raw_catalog is None else raw_catalog.dimensions,
        ).to(device)
        n_params = sum(param.numel() for param in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        logger.info(f"active_channels: {list(model.active_channels)}")

        optimizer = build_optimizer(model, config)
        trainer = MoSIDTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            beam_valid_loader=valid_loader,
            beam_trie=beam_trie,
            config=config,
            logger=logger,
            device=device,
            raw_catalog=raw_catalog,
            prefix_negative_sampler=prefix_negative_sampler,
        )
        trainer.train()
        logger.info(f"Training complete; artifacts: {run_dir}")
    finally:
        logger.close()


def _build_loader(
    config,
    dataset,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size or config.train.batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=int(config.train.num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def build_optimizer(model: MoSIDModel, config) -> torch.optim.Optimizer:
    scorer = getattr(model, "raw_compatibility_scorer", None)
    modality_logits = getattr(scorer, "raw_modality_logits", None)
    if modality_logits is None:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not modality_logits
    ]
    return torch.optim.AdamW(
        [
            {"params": other_parameters},
            {"params": [modality_logits], "weight_decay": 0.0},
        ],
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )


def build_prefix_negative_sampler(
    config,
    train_dataset: MoSIDSequenceDataset,
) -> PrefixHardNegativeSampler | None:
    if float(getattr(config.loss, "raw_aux_weight", 0.0)) <= 0.0:
        return None
    if not train_dataset.use_unified_block_views:
        return SeparatePrefixHardNegativeSampler(
            sid_tables={str(channel): train_dataset.sid_tables[str(channel)]
                        for channel in config.model.active_channels},
            counts_by_lcp=config.loss.raw_prefix_hard_negative_counts_by_lcp,
            seed=int(config.train.seed),
        )
    first_channel = str(config.model.active_channels[0])
    return PrefixHardNegativeSampler(
        sid_table=train_dataset.sid_tables[first_channel],
        counts_by_lcp=config.loss.raw_prefix_hard_negative_counts_by_lcp,
        seed=int(config.train.seed),
    )


def mosid_sid_artifact_paths(config) -> dict[str, str]:
    data_dir = Path(config.data._data_dir)
    return {
        str(channel): str(data_dir / sid_codes_filename(mosid_sid_artifact_channel(config, str(channel))))
        for channel in config.model.active_channels
    }


def build_mosid_tries(config):
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


if __name__ == "__main__":
    main()
