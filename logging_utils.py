"""Logging utilities for the option scanner project."""
from __future__ import annotations

from pathlib import Path
import sys

from loguru import logger


def configure_logging(log_dir: Path, log_name: str, rotation: str = "1 week") -> None:
    """Configure a rotating log file for the given log name."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"

    logger.remove()
    logger.add(
        log_path,
        rotation=rotation,
        retention="90 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    )
    logger.add(sys.stdout, level="INFO")


def get_logger(log_dir: Path, log_name: str):
    """Helper that configures logging and returns the global logger."""

    configure_logging(log_dir, log_name)
    return logger
