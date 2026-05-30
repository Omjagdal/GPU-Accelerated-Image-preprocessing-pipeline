"""
Structured Logging Module for FewShot-Industrial-Vision.

Provides coloured console output, rotating file handlers, and a
specialised ``TrainingLogger`` for tracking episodic training metrics
with optional CSV persistence.

Usage:
    >>> from src.utils.logger import setup_logger, TrainingLogger
    >>> log = setup_logger("train", log_file="logs/train.log")
    >>> log.info("Starting experiment")
    >>> tl = TrainingLogger(csv_path="results/metrics.csv")
    >>> tl.log_episode(epoch=1, episode=5, loss=0.42, accuracy=0.87)
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ANSI colour codes for console output
# ---------------------------------------------------------------------------

class _Colours:
    """ANSI escape sequences for terminal colouring."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GREY = "\033[90m"

    # Mapping log levels → colours
    LEVEL_MAP: Dict[int, str] = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD + RED,
    }


# ---------------------------------------------------------------------------
# Custom Formatters
# ---------------------------------------------------------------------------

_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


class ColouredFormatter(logging.Formatter):
    """Log formatter that applies ANSI colours to the level name and timestamp.

    Colour output is automatically disabled when writing to non-TTY
    streams (e.g. piped output or CI environments).
    """

    _FMT = "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__(fmt=self._FMT, datefmt=_TIMESTAMP_FMT)
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colour:
            colour = _Colours.LEVEL_MAP.get(record.levelno, _Colours.RESET)
            record.levelname = f"{colour}{record.levelname}{_Colours.RESET}"
            record.asctime = (
                f"{_Colours.DIM}"
                f"{self.formatTime(record, self.datefmt)}"
                f"{_Colours.RESET}"
            )
            record.name = f"{_Colours.CYAN}{record.name}{_Colours.RESET}"
        return super().format(record)


class FileFormatter(logging.Formatter):
    """Plain-text formatter for log files (no ANSI codes)."""

    _FMT = "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=_TIMESTAMP_FMT)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure and return a named logger with console and optional file output.

    Args:
        name: Logger name (typically module or component identifier).
        log_file: Path to a rotating log file. ``None`` disables file logging.
        level: Minimum severity level to capture.
        max_bytes: Maximum size in bytes before log rotation (default 5 MB).
        backup_count: Number of rotated backup files to retain.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = setup_logger("backbone", log_file="logs/backbone.log")
        >>> logger.info("Backbone initialised")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # --- Console handler (coloured) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    use_colour = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    console_handler.setFormatter(ColouredFormatter(use_colour=use_colour))
    logger.addHandler(console_handler)

    # --- File handler (rotating, plain-text) ---
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate messages
    logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# Training Logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Structured logger for episodic few-shot training metrics.

    Records per-episode and per-epoch statistics and optionally persists
    them to a CSV file for downstream analysis.

    Attributes:
        logger: Underlying ``logging.Logger`` instance.
        csv_path: Optional CSV file for metric persistence.
        history: In-memory list of all recorded metric dictionaries.
    """

    # CSV column order
    _CSV_COLUMNS: List[str] = [
        "timestamp",
        "epoch",
        "episode",
        "loss",
        "accuracy",
        "learning_rate",
        "phase",
    ]

    def __init__(
        self,
        name: str = "training",
        log_file: Optional[str] = None,
        csv_path: Optional[str] = None,
        level: int = logging.INFO,
    ) -> None:
        """Initialise the training logger.

        Args:
            name: Logger name.
            log_file: Path for the rotating log file.
            csv_path: Path for the CSV metrics file. Created on first write.
            level: Logging severity level.
        """
        self.logger: logging.Logger = setup_logger(
            name=name, log_file=log_file, level=level
        )
        self.csv_path: Optional[Path] = Path(csv_path) if csv_path else None
        self.history: List[Dict[str, Any]] = []

        # Create CSV with header if it does not exist yet
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                self._write_csv_header()

    # ---- Public API ----

    def log_episode(
        self,
        epoch: int,
        episode: int,
        loss: float,
        accuracy: float,
        learning_rate: Optional[float] = None,
        phase: str = "train",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metrics for a single training episode.

        Args:
            epoch: Current epoch number.
            episode: Episode index within the epoch.
            loss: Episode loss value.
            accuracy: Episode accuracy (0-1).
            learning_rate: Current optimiser learning rate.
            phase: Training phase (``'train'`` or ``'val'``).
            extra: Arbitrary additional key-value pairs.
        """
        record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "episode": episode,
            "loss": round(loss, 6),
            "accuracy": round(accuracy, 4),
            "learning_rate": learning_rate,
            "phase": phase,
        }
        if extra:
            record.update(extra)

        self.history.append(record)

        msg = (
            f"[{phase.upper():>5}] "
            f"Epoch {epoch:>4d} │ "
            f"Episode {episode:>4d} │ "
            f"Loss: {loss:.4f} │ "
            f"Acc: {accuracy:.4f}"
        )
        if learning_rate is not None:
            msg += f" │ LR: {learning_rate:.2e}"
        self.logger.info(msg)

        # Append to CSV
        if self.csv_path is not None:
            self._append_csv_row(record)

    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        best_val_acc: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        """Log a summary line at the end of an epoch.

        Args:
            epoch: Epoch number.
            train_loss: Mean training loss over the epoch.
            train_acc: Mean training accuracy over the epoch.
            val_loss: Mean validation loss (if applicable).
            val_acc: Mean validation accuracy (if applicable).
            best_val_acc: Best validation accuracy seen so far.
            elapsed_seconds: Wall-clock time for the epoch in seconds.
        """
        separator = "═" * 60
        self.logger.info(separator)

        parts = [
            f"Epoch {epoch:>4d} Summary │ "
            f"Train Loss: {train_loss:.4f} │ "
            f"Train Acc: {train_acc:.4f}"
        ]
        if val_loss is not None and val_acc is not None:
            parts.append(
                f" │ Val Loss: {val_loss:.4f} │ Val Acc: {val_acc:.4f}"
            )
        if best_val_acc is not None:
            parts.append(f" │ Best Val: {best_val_acc:.4f}")
        if elapsed_seconds is not None:
            parts.append(f" │ Time: {elapsed_seconds:.1f}s")

        self.logger.info("".join(parts))
        self.logger.info(separator)

    def log_info(self, message: str) -> None:
        """Log a general informational message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def save_history(self, path: Optional[str] = None) -> Path:
        """Persist the full in-memory history to a CSV file.

        Args:
            path: Destination path. Falls back to ``self.csv_path``.

        Returns:
            Path: Resolved path to the written CSV file.

        Raises:
            ValueError: If no path is provided and ``csv_path`` was not set.
        """
        target = Path(path) if path else self.csv_path
        if target is None:
            raise ValueError(
                "No CSV path provided. Set csv_path or pass a path argument."
            )

        target.parent.mkdir(parents=True, exist_ok=True)

        # Gather all unique columns across history records
        all_columns = list(dict.fromkeys(
            col
            for record in self.history
            for col in record.keys()
        ))

        with open(target, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(self.history)

        self.logger.info(f"Saved {len(self.history)} records to {target}")
        return target.resolve()

    # ---- Private helpers ----

    def _write_csv_header(self) -> None:
        """Write the CSV header row."""
        assert self.csv_path is not None
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(self._CSV_COLUMNS)

    def _append_csv_row(self, record: Dict[str, Any]) -> None:
        """Append a single metric row to the CSV file."""
        assert self.csv_path is not None
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=self._CSV_COLUMNS,
                extrasaction="ignore",
            )
            writer.writerow(record)
