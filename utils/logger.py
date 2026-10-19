"""Console, file, and optional TensorBoard logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterable


def format_horizontal_metrics_table(
    title: str,
    pairs: list[tuple[str, str]],
    pairs_per_row: int = 4,
) -> str:
    """Format metrics as a horizontal table for epoch summaries."""
    if not pairs:
        return title
    if pairs_per_row <= 0:
        raise ValueError(f"pairs_per_row must be positive, got {pairs_per_row}")

    key_width = max(len("metric"), *(len(key) for key, _value in pairs))
    value_width = max(len("value"), *(len(value) for _key, value in pairs))
    chunk_size = pairs_per_row

    header_cells = []
    for _ in range(chunk_size):
        header_cells.extend([("metric").ljust(key_width), ("value").rjust(value_width)])
    header = " | ".join(header_cells)
    width = len(header)

    lines = [
        "=" * width,
        title,
        "=" * width,
        header,
        "-" * width,
    ]

    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start:start + chunk_size]
        row_cells = []
        for key, value in chunk:
            row_cells.extend([key.ljust(key_width), value.rjust(value_width)])
        while len(chunk) < chunk_size:
            row_cells.extend([" " * key_width, " " * value_width])
            chunk.append(("", ""))
        lines.append(" | ".join(row_cells))
    lines.append("=" * width)
    return "\n".join(lines)


def progress_iter(
    iterable: Iterable,
    total: int | None = None,
    desc: str | None = None,
    enabled: bool = True,
    leave: bool = False,
    dynamic_ncols: bool = True,
):
    """Wrap an iterable with tqdm when available, otherwise return it unchanged."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=dynamic_ncols,
    )


class _SafeFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._disabled_by_write_error = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._disabled_by_write_error:
            return
        super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, OSError):
            self._disabled_by_write_error = True
            try:
                self.close()
            except Exception:
                pass
            return
        super().handleError(record)


class Logger:
    """Write training logs to console, files, and optional TensorBoard."""

    def __init__(self, log_dir: str | Path, name: str = "MoSID",
                 tensorboard: bool = False) -> None:
        """Initialize logging destinations and optional TensorBoard output."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tb = tensorboard

        # Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)


        file_handler = _SafeFileHandler(
            self.log_dir / f"{name}.log", encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)


        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
        else:
            self._writer = None


    def info(self, msg: str) -> None:
        """Log an INFO message."""
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a WARNING message."""
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an ERROR message."""
        self._logger.error(msg)

    # ---- TensorBoard ----
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write one TensorBoard scalar when enabled."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_metrics(
        self, metrics: dict[str, float], step: int, prefix: str = "",
    ) -> None:
        """Log metrics as text and optional TensorBoard scalars."""
        parts = []
        for name, value in metrics.items():
            tag = f"{prefix}{name}" if prefix else name
            if self._writer is not None:
                self._writer.add_scalar(tag, value, step)
            parts.append(f"{name}={value:.4f}")
        self.info(f"[step={step}] {', '.join(parts)}")

    def log_config(self, config_dict: dict[str, Any]) -> None:
        """Write configuration text to TensorBoard when enabled."""
        if self._writer is None:
            return
        text = "\n".join(f"**{k}**: {v}" for k, v in _flatten_dict(config_dict).items())
        self._writer.add_text("config", text)


    def close(self) -> None:
        """Close TensorBoard and file handlers."""
        if self._writer is not None:
            self._writer.close()
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested mappings into dotted keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
