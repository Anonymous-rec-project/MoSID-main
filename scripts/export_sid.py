"""Export one-offset SID tables and codebook weights for Stage 2."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import (
    DotDict,
    infer_checkpoint_overrides,
    load_from_checkpoint_or_config,
    resolve_config_path,
)
from data.rqvae_dataset import MPSIDRQVAEDataset, RQVAEDataset
from models.rqvae import RqVae
from models.unified_rqvae import MPSIDRqVae
from utils.data import (
    UNIFIED_SID_CHANNEL,
    build_rqvae_valid_mask,
    codebook_weights_filename,
    load_mod_presence,
    sid_codes_filename,
)
from utils.io import load_checkpoint
from utils.seed import set_seed, set_device


def _is_mpsid_config(config: DotDict) -> bool:
    return str(getattr(config.data, "modality", "")).lower() == UNIFIED_SID_CHANNEL


def _validate_checkpoint_data_identity(config: DotDict, checkpoint_path: str) -> None:
    inferred = infer_checkpoint_overrides(checkpoint_path)
    resolved = {
        "data.dataset": str(config.data.dataset),
        "data.mode": str(config.data.mode),
        "data.modality": str(getattr(config.data, "modality", "")),
    }
    for key, expected in inferred.items():
        if key not in resolved:
            continue
        if resolved[key] != str(expected):
            raise ValueError(
                f"checkpoint path identifies {key}={expected!r}, but the resolved config uses "
                f"{resolved[key]!r}; refusing to export into the wrong dataset"
            )


def _validate_checkpoint_item_count(
    checkpoint_metadata: dict,
    expected_items: int,
    checkpoint_path: str,
    config: DotDict,
) -> None:
    metrics = checkpoint_metadata.get("metrics", {})
    reported_items = metrics.get("total_items") if isinstance(metrics, dict) else None
    if reported_items is None:
        return
    reported_items = int(reported_items)
    if reported_items != expected_items:
        raise ValueError(
            f"checkpoint {checkpoint_path!r} reports total_items={reported_items}, but resolved "
            f"dataset {config.data.dataset!r} has {expected_items} valid items; refusing to overwrite "
            f"SID artifacts under {config.data._data_dir!r}"
        )



def export_sid(config: DotDict, checkpoint_path: str, skip_collision: bool) -> None:
    """Quantize all valid items and export SID codes and codebook weights."""

    if _is_mpsid_config(config):
        export_mpsid_sid(config, checkpoint_path)
        return

    set_seed(config.train.seed)
    device = set_device(config.train.gpu_id)

    modality = config.data.modality
    data_dir = config.data._data_dir
    n_layers = config.model.n_layers

    print(f"[export_sid] Modality: {modality}  Mode: {config.data.mode}")
    print(f"[export_sid] checkpoint: {checkpoint_path}")
    print(f"[export_sid] Data directory: {data_dir}")


    model = RqVae(config).to(device)
    checkpoint_metadata = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    print(f"[export_sid] Model loaded")


    dataset = RQVAEDataset(config)
    n_total = dataset.valid_mask.shape[0]
    print(f"[export_sid] Embedding rows: {n_total} (including padding index 0)")


    valid_mask = dataset.valid_mask
    n_valid = valid_mask.sum().item()
    print(f"[export_sid] Valid item count: {n_valid}")
    _validate_checkpoint_item_count(
        checkpoint_metadata,
        int(n_valid),
        checkpoint_path,
        config,
    )


    sid_codes = torch.zeros(n_total, n_layers, dtype=torch.long)
    valid_indices = dataset.valid_item_ids

    batch_size = 1024
    for start in range(0, len(valid_indices), batch_size):
        end = min(start + batch_size, len(valid_indices))
        batch_ids = valid_indices[start:end]


        batch_emb = dataset.embeddings[start:end]
        batch_emb = F.normalize(batch_emb, dim=-1)

        batch_emb = batch_emb.to(device)
        codes = model.get_semantic_ids(batch_emb)  # (B, n_layers)
        sid_codes[batch_ids] = codes.cpu()


    sid_codes[valid_mask] += 1


    codebook_weights = model.get_codebook_weights()  # list of (CS, D)


    sid_path = os.path.join(data_dir, sid_codes_filename(modality))
    cb_path = os.path.join(data_dir, codebook_weights_filename(modality))
    torch.save(sid_codes, sid_path)
    torch.save(codebook_weights, cb_path)
    print(f"[export_sid] Saved SID table: {sid_path}")
    print(f"[export_sid] Saved codebook weights: {cb_path}")


    _verify_and_report(sid_codes, codebook_weights, valid_mask, modality, config)


    if not skip_collision:
        _try_joint_collision_check(data_dir, config)


def export_mpsid_sid(config: DotDict, checkpoint_path: str) -> None:
    """Export unified semantic IDs and block-structured codebooks."""
    set_seed(config.train.seed)
    device = set_device(config.train.gpu_id)

    data_dir = config.data._data_dir
    n_layers = config.model.n_layers
    codebook_size = config.model.codebook_size
    artifact_channel = UNIFIED_SID_CHANNEL
    print(f"[export_sid] Detected MP-SID model: data.modality={config.data.modality}")
    print(f"[export_sid] artifact channel: {artifact_channel}")
    print(f"[export_sid] checkpoint: {checkpoint_path}")
    print(f"[export_sid] Data directory: {data_dir}")

    model = MPSIDRqVae(config).to(device)
    checkpoint_metadata = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    print("[export_sid] MP-SID Model loaded")

    dataset = MPSIDRQVAEDataset(config)
    n_total = int(dataset.valid_mask.shape[0])
    valid_mask = dataset.valid_mask.bool().clone()
    valid_indices = dataset.valid_item_ids.long()
    n_valid = int(valid_mask.sum().item())
    print(f"[export_sid] Embedding rows: {n_total} (including padding index 0)")
    print(f"[export_sid] unified Valid item count: {n_valid}")
    _validate_checkpoint_item_count(
        checkpoint_metadata,
        n_valid,
        checkpoint_path,
        config,
    )

    sid_codes = torch.zeros(n_total, n_layers, dtype=torch.long)
    batch_size = int(getattr(config.train, "batch_size", 1024))
    offset = 0
    with torch.inference_mode():
        for batch in dataset.iter_batches(batch_size):
            batch_len = int(batch["item_id"].shape[0])
            batch_on_device = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            codes = model.get_semantic_ids(batch_on_device)
            item_ids = batch["item_id"].long()
            expected_ids = valid_indices[offset:offset + batch_len]
            if not torch.equal(item_ids.cpu(), expected_ids.cpu()):
                raise AssertionError("MP-SID export batch item order does not match valid indices")
            sid_codes[item_ids] = codes.cpu() + 1
            offset += batch_len

    if offset != n_valid:
        raise AssertionError(f"MP-SID export processed {offset} items, expected {n_valid}")

    codebook_weights = model.get_codebook_weights()
    _verify_mpsid_unified_artifact_contract(
        sid_codes,
        codebook_weights,
        valid_mask,
        config,
        context="pre-save",
    )

    sid_path = os.path.join(data_dir, sid_codes_filename(artifact_channel))
    cb_path = os.path.join(data_dir, codebook_weights_filename(artifact_channel))
    torch.save(sid_codes, sid_path)
    torch.save(codebook_weights, cb_path)
    print(f"[export_sid] Saved unified SID table: {sid_path}")
    print(f"[export_sid] Saved unified codebook weights: {cb_path}")
    saved_sid_codes = torch.load(sid_path, map_location="cpu", weights_only=True)
    saved_codebook_weights = torch.load(cb_path, map_location="cpu", weights_only=True)
    _verify_mpsid_unified_artifact_contract(
        saved_sid_codes,
        saved_codebook_weights,
        valid_mask,
        config,
        context="post-save",
    )
    _verify_and_report(sid_codes, codebook_weights, valid_mask, artifact_channel, config)
    print("[export_sid] MP-SID unified export complete")
    with open(os.path.join(data_dir, "sid_unified_source.json"), "w", encoding="utf-8") as file:
        json.dump({"seed": int(config.train.seed), "checkpoint": os.path.basename(checkpoint_path)}, file)


def _verify_mpsid_unified_artifact_contract(
    sid_codes: torch.Tensor,
    codebook_weights: list[torch.Tensor],
    valid_mask: torch.Tensor,
    config: DotDict,
    *,
    context: str,
) -> None:
    """Check the Stage 1 unified artifact contract without changing exported values."""
    n_total = int(valid_mask.shape[0])
    n_layers = int(config.model.n_layers)
    codebook_size = int(config.model.codebook_size)
    expected_sid_shape = (n_total, n_layers)
    if tuple(sid_codes.shape) != expected_sid_shape:
        raise AssertionError(
            f"MP-SID unified SID {context} shape should be {expected_sid_shape}, got {tuple(sid_codes.shape)}"
        )
    if sid_codes.dtype != torch.long:
        raise AssertionError(f"MP-SID unified SID {context} dtype should be torch.long, got {sid_codes.dtype}")
    if sid_codes[0].sum().item() != 0:
        raise AssertionError(f"MP-SID unified SID {context} padding row should be all zero")

    mask = valid_mask.to(device=sid_codes.device, dtype=torch.bool)
    valid_codes = sid_codes[mask]
    if valid_codes.numel() == 0:
        raise AssertionError(f"MP-SID unified SID {context} has no valid item rows")
    if (valid_codes <= 0).any().item():
        raise AssertionError(f"MP-SID unified SID {context} valid rows should be strictly positive")
    if (valid_codes > codebook_size).any().item():
        raise AssertionError(f"MP-SID unified SID {context} codes exceed codebook_size={codebook_size}")

    invalid_mask = ~mask
    invalid_mask[0] = False
    if invalid_mask.any().item() and (sid_codes[invalid_mask] != 0).any().item():
        raise AssertionError(f"MP-SID unified SID {context} invalid rows should remain zero")

    if not isinstance(codebook_weights, list):
        raise AssertionError(f"MP-SID unified codebook {context} should be a list")
    if len(codebook_weights) != n_layers:
        raise AssertionError(
            f"MP-SID unified codebook {context} should have {n_layers} layers, got {len(codebook_weights)}"
        )

    expected_codebook_dim = int(config.model.latent_block_dim) * 3
    expected_weight_shape = (codebook_size, expected_codebook_dim)
    for layer_idx, weight in enumerate(codebook_weights):
        if tuple(weight.shape) != expected_weight_shape:
            raise AssertionError(
                f"MP-SID unified codebook {context} layer {layer_idx} shape should be "
                f"{expected_weight_shape}, got {tuple(weight.shape)}"
            )



def _verify_and_report(
    sid_codes: torch.Tensor,
    codebook_weights: list[torch.Tensor],
    valid_mask: torch.Tensor,
    modality: str,
    config: DotDict,
) -> None:
    """Validate exported artifacts and report their statistics."""
    n_layers = config.model.n_layers
    codebook_size = config.model.codebook_size

    print("=" * 60)
    print("[Validation] SID export results")
    print("=" * 60)


    print(f"  shape: {sid_codes.shape}  dtype: {sid_codes.dtype}")


    assert sid_codes[0].sum().item() == 0, "Padding row (index 0) must be all zero"
    print(f"  Padding row (index 0): {sid_codes[0].tolist()} ✓")


    valid_count = (sid_codes.sum(dim=-1) > 0).sum().item()
    expected_valid = valid_mask.sum().item()
    print(f"  Valid item count: {valid_count} (expected: {expected_valid})")
    assert valid_count == expected_valid, \
        f"Valid item count mismatch: {valid_count} != {expected_valid}"
    print(f"  Valid item count verified ✓")


    valid_codes = sid_codes[valid_mask]
    min_code = valid_codes.min().item()
    max_code = sid_codes.max().item()
    print(f"  Valid code range: [{min_code}, {max_code}] (expected: [1, {codebook_size}])")
    assert min_code >= 1, f"+1 Invalid offset: min={min_code} must be >= 1"
    assert max_code <= codebook_size, f"Code out of range: max={max_code} must be <= {codebook_size}"
    print(f"  +1 Offset verified ✓")


    if modality in ("text", "vision", "fused"):
        invalid_mask = ~valid_mask
        invalid_mask[0] = False
        if invalid_mask.sum().item() > 0:
            invalid_codes = sid_codes[invalid_mask]
            assert (invalid_codes == 0).all(), "Missing-item rows must be all zero"
            print(f"  Missing-item rows ({invalid_mask.sum().item()} items): all zero ✓")


    codebook_dim = (
        int(config.model.latent_block_dim) * 3
        if modality == UNIFIED_SID_CHANNEL
        else int(config.model.embed_dim)
    )
    print(f"  Number of codebook layers: {len(codebook_weights)}")
    for k, w in enumerate(codebook_weights):
        print(f"    layer {k}: shape {w.shape}")
        assert w.shape == (codebook_size, codebook_dim), \
            f"Invalid codebook weight shape at layer {k} {w.shape}"
    print(f"  Codebook weight shapes verified ✓")


    print(f"\n  Per-layer codebook utilization (valid items {valid_count} items):")
    for k in range(n_layers):
        layer_codes = valid_codes[:, k]
        n_unique = len(torch.unique(layer_codes))
        usage = n_unique / codebook_size
        print(f"    layer {k}: {n_unique}/{codebook_size} = {usage:.4f}")


    sid_set = set()
    for i in range(valid_codes.shape[0]):
        sid_set.add(tuple(valid_codes[i].tolist()))
    collision_rate = 1.0 - len(sid_set) / valid_count
    print(f"\n  Single-branch collision rate: {collision_rate:.4f} (unique: {len(sid_set)}/{valid_count})")

    print("=" * 60)



def _try_joint_collision_check(data_dir: str, config: DotDict) -> None:
    """Compute joint collisions when all three branch artifacts exist."""
    modalities = ["text", "vision", "collab"]
    paths = {
        m: os.path.join(data_dir, sid_codes_filename(m))
        for m in modalities
    }


    all_exist = all(os.path.exists(p) for p in paths.values())
    if not all_exist:
        missing = [m for m, p in paths.items() if not os.path.exists(p)]
        print(f"[Joint collision rate] Skipping: missing {missing} SID table")
        return

    print(f"[Joint collision rate] All three SID tables found; computing collisions...")
    _compute_joint_collision_rate(data_dir, config, paths)


def _compute_joint_collision_rate(
    data_dir: str, config: DotDict, paths: dict[str, str]
) -> None:
    """Compute joint and pairwise SID collision rates."""
    sid_text = torch.load(paths["text"], map_location="cpu", weights_only=True)
    sid_vision = torch.load(paths["vision"], map_location="cpu", weights_only=True)
    sid_collab = torch.load(paths["collab"], map_location="cpu", weights_only=True)

    mod_presence = load_mod_presence(data_dir, config)

    full_mask = build_rqvae_valid_mask("fused", mod_presence, sid_text.shape[0])
    n_full = full_mask.sum().item()

    if n_full == 0:
        print("[Joint collision rate] No items with all three modalities; skipping")
        return

    full_indices = full_mask.nonzero(as_tuple=True)[0]


    joint_sids: dict[tuple, list[int]] = {}
    for item_id in full_indices.tolist():
        key = (
            tuple(sid_text[item_id].tolist()),
            tuple(sid_vision[item_id].tolist()),
            tuple(sid_collab[item_id].tolist()),
        )
        joint_sids.setdefault(key, []).append(item_id)

    n_unique_joint = len(joint_sids)
    joint_cr = 1.0 - n_unique_joint / n_full

    print(f"{'=' * 60}")
    print(f"[Joint collision rate] Items with all three modalities: {n_full}")
    print(f"  Three-branch joint collision rate: {joint_cr:.4f} (unique: {n_unique_joint}/{n_full})")


    pairs = [
        ("text+vision", sid_text, sid_vision),
        ("text+collab", sid_text, sid_collab),
        ("vision+collab", sid_vision, sid_collab),
    ]
    for pair_name, sid_a, sid_b in pairs:
        pair_sids: set[tuple] = set()
        for item_id in full_indices.tolist():
            key = (
                tuple(sid_a[item_id].tolist()),
                tuple(sid_b[item_id].tolist()),
            )
            pair_sids.add(key)
        n_unique_pair = len(pair_sids)
        pair_cr = 1.0 - n_unique_pair / n_full
        print(f"  {pair_name} Collision rate: {pair_cr:.4f} (unique: {n_unique_pair}/{n_full})")

    print(f"{'=' * 60}")



def main():
    parser = argparse.ArgumentParser(description="Export SID codes and codebook weights")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML path; otherwise use the checkpoint config or configs/rqvae_unified.yaml",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="RQ-VAE best checkpoint path")
    parser.add_argument("--skip_collision_check", action="store_true",
                        help="Skip cross-branch joint collision checks")

    known, unknown = parser.parse_known_args()
    resolved_config = resolve_config_path(known.config, known.checkpoint, "configs/rqvae_unified.yaml")
    print(f"[config] source: {resolved_config}")
    config = load_from_checkpoint_or_config(
        known.config,
        known.checkpoint,
        unknown,
        fallback_config_path="configs/rqvae_unified.yaml",
    )
    _validate_checkpoint_data_identity(config, known.checkpoint)

    export_sid(config, known.checkpoint, known.skip_collision_check)


if __name__ == "__main__":
    main()
