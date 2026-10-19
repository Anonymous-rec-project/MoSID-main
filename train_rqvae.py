"""Train Stage 1 with configs/rqvae_unified.yaml."""

from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from configs.config import parse_args_and_load, build_run_dir
from data.rqvae_dataset import RQVAEDataset
from models.rqvae import RqVae
from models.unified_rqvae import MPSIDRqVae
from trainers.rqvae_trainer import MPSIDTrainer, RQVAETrainer
from utils.data import UNIFIED_SID_CHANNEL
from utils.logger import Logger
from utils.seed import set_seed, set_device


def _is_mpsid_config(config) -> bool:
    return str(getattr(config.data, "modality", "")).lower() == UNIFIED_SID_CHANNEL


def _stack_mpsid_kmeans_batch(dataset: RQVAEDataset, device: torch.device) -> dict[str, torch.Tensor]:
    batch = dataset.get_batch(slice(0, len(dataset)))
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def main():

    config = parse_args_and_load("Train RQ-VAE")




    run_dir = build_run_dir(config, stage="rqvae", suffix=config.data.modality)
    run_dir_file = str(getattr(config.train, "run_dir_file", "") or "")
    if run_dir_file:
        Path(run_dir_file).write_text(str(run_dir), encoding="utf-8")
    if (_is_mpsid_config(config) or getattr(config.train, "seeded_run", False)) and (Path(run_dir) / "best.pt").is_file():
        print(f"Reusing Stage 1 checkpoint: {Path(run_dir) / 'best.pt'}")
        return
    set_seed(config.train.seed)
    device = set_device(config.train.gpu_id)
    training_started_at = time.perf_counter()
    logger = Logger(run_dir, tensorboard=config.train.tensorboard)
    logger.training_started_at = training_started_at
    logger.info(f"Config: {config}")
    logger.info(f"Modality: {config.data.modality}  Mode: {config.data.mode}")
    logger.info(f"Device: {device}")


    dataset = RQVAEDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
    )
    logger.info(f"Valid item count: {len(dataset)}")


    is_mpsid = _is_mpsid_config(config)
    model = MPSIDRqVae(config).to(device) if is_mpsid else RqVae(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    if is_mpsid:
        logger.info(f"MP-SID raw encoders: {model.raw_encoders}")
        logger.info(f"MP-SID raw decoders: {model.raw_decoders}")
    else:
        logger.info(f"Encoder layers: {model.encoder.net}")
        logger.info(f"Decoder layers: {model.decoder.net}")


    if config.model.kmeans_init:
        logger.info("Initializing codebooks with layer-wise K-means...")
        if is_mpsid:
            model.kmeans_init(_stack_mpsid_kmeans_batch(dataset, device), logger=logger)
        else:
            all_embeddings = torch.stack(
                [dataset[i]["embedding"] for i in range(len(dataset))]
            )
            model.kmeans_init(all_embeddings.to(device), logger=logger)
        logger.info("K-means initialization complete")


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )


    trainer_cls = MPSIDTrainer if is_mpsid else RQVAETrainer
    trainer = trainer_cls(
        model, optimizer, dataloader, dataset,
        config, logger, device,
    )
    trainer.train()

    logger.info(f"Training complete; artifacts: {run_dir}")
    logger.close()


if __name__ == "__main__":
    main()
