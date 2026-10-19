"""Serialization and checkpoint utilities."""

from __future__ import annotations

import json
import hashlib
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
try:
    import yaml
except ModuleNotFoundError:
    from configs.config import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create the directory and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ==================== Pickle ====================

def load_pkl(path: str | Path) -> Any:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: str | Path) -> None:
    """Save an object as pickle."""
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ==================== JSON ====================

def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    """Save indented JSON preserving Unicode."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


# ==================== YAML ====================

def load_yaml(path: str | Path) -> dict:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==================== Checkpoint ====================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    path: str | Path,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save model and optimizer states, epoch, metrics, and optional extra state."""
    target = Path(path)
    ensure_dir(target.parent)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    try:
        torch.save(payload, target)
    except OSError as exc:
        raise OSError(exc.errno, _checkpoint_save_error_message(target), str(target)) from exc
    except RuntimeError as exc:
        if _looks_like_torch_checkpoint_write_error(exc):
            raise RuntimeError(_checkpoint_save_error_message(target)) from exc
        raise


def _looks_like_torch_checkpoint_write_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return (
        "unexpected pos" in text
        or "inline_container" in text
        or "failed writing file" in text
        or "no space left on device" in text
    )


def _checkpoint_save_error_message(path: Path) -> str:
    return (
        f"failed to save checkpoint to {path}. "
        "The filesystem likely rejected the write; if you see 'No space left on device' "
        "or PyTorch reports 'unexpected pos', free space in train.output_dir or point "
        "train.output_dir to a filesystem with enough room for best.pt and last.pt."
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Restore model and optional optimizer states and return checkpoint metadata."""
    ckpt = load_checkpoint_payload(path, device=device)
    return restore_checkpoint_payload(ckpt, model, optimizer=optimizer)


def load_checkpoint_payload(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise TypeError(f"checkpoint {path} must contain a model_state_dict mapping")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise TypeError(f"checkpoint {path} model_state_dict must be a mapping")
    return payload


def restore_checkpoint_payload(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    model.load_state_dict(_normalized_model_state(payload))

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return _checkpoint_metadata(payload)


def _normalized_model_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise TypeError("checkpoint model_state_dict must be a mapping")
    if any(str(key).startswith("module.") for key in model_state):
        return {str(key).removeprefix("module."): value for key, value in model_state.items()}
    return {str(key): value for key, value in model_state.items()}


def _checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": payload.get("epoch", 0),
        "metrics": payload.get("metrics", {}),
        "extra_state": payload.get("extra_state", {}),
    }
