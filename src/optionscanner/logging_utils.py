"""Logging utilities for the option scanner project."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from loguru import logger

try:
    from logging_loki import LokiHandler
except Exception:  # pragma: no cover - optional dependency
    LokiHandler = None


def configure_logging(
    log_dir: Path,
    log_name: str,
    rotation: str = "1 week",
    run_mode: Optional[str] = None,
    log_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Configure a rotating log file for the given log name.

    Args:
        log_dir: Directory for log files
        log_name: Base name for log files
        rotation: Log rotation interval
        run_mode: Current run mode (local, schedule)
        log_config: Optional logging configuration from config.yaml
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"

    # Parse log config
    log_config = log_config or {}
    loki_config = log_config.get("loki", {})
    console_config = log_config.get("console", {})
    file_config = log_config.get("file", {})

    console_level = console_config.get("level", "INFO")
    file_level = file_config.get("level", "DEBUG")
    rotation = file_config.get("rotation", rotation)
    retention = file_config.get("retention", "90 days")

    logger.remove()

    # Configure logger with patcher to ensure 'component' field always exists in extra
    # This must use logger.configure() to affect the global logger instance
    def _ensure_component(record: Dict[str, Any]) -> None:
        if "component" not in record["extra"]:
            record["extra"]["component"] = "app"

    logger.configure(patcher=_ensure_component)

    # Console sink with structured format
    logger.add(
        sys.stdout,
        level=console_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[component]: <20} | {message}",
    )

    # File sink with detailed format
    logger.add(
        log_path,
        level=file_level,
        rotation=rotation,
        retention=retention,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[component]: <20} | {name}:{line} | {message}",
    )

    # Loki sink for Grafana Cloud
    _configure_loki_sink(
        run_mode=run_mode,
        loki_config=loki_config,
    )


def get_logger(
    log_dir: Path,
    log_name: str,
    run_mode: Optional[str] = None,
    log_config: Optional[Dict[str, Any]] = None,
):
    """Helper that configures logging and returns the global logger."""
    configure_logging(log_dir, log_name, run_mode=run_mode, log_config=log_config)
    return logger


def _configure_loki_sink(
    run_mode: Optional[str] = None,
    loki_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Optionally forward logs to Grafana Cloud Loki when env vars are provided.

    Args:
        run_mode: Current run mode (local, schedule)
        loki_config: Loki-specific configuration
    """
    loki_config = loki_config or {}

    # Check if Loki is enabled in config
    if not loki_config.get("enabled", True):
        return

    url = os.getenv("LOKI_URL")
    username = os.getenv("LOKI_USERNAME")
    password = os.getenv("LOKI_PASSWORD")
    tenant = os.getenv("LOKI_TENANT") or os.getenv("LOKI_ORG_ID") or username

    if not (url and username and password):
        return

    if LokiHandler is None:
        logger.warning("LOKI_* env vars set but python-logging-loki not installed; skipping remote logging")
        return

    endpoint = url.rstrip("/")
    if not endpoint.endswith("loki/api/v1/push"):
        endpoint = f"{endpoint}/loki/api/v1/push"

    tags = {"app": "optionscanner"}
    run_mode_label = run_mode or os.getenv("APP_RUN_MODE") or os.getenv("RUN_MODE")
    if run_mode_label:
        tags["run_mode"] = run_mode_label
    env_label = os.getenv("APP_ENV")
    if env_label:
        tags["env"] = env_label

    batch_size = int(loki_config.get("batch_size", 100))
    flush_interval = int(loki_config.get("flush_interval", 5))

    try:
        handler = LokiHandler(
            url=endpoint,
            auth=(username, password),
            version="1",
            tags=tags,
            headers={"X-Scope-OrgID": tenant} if tenant else None,
        )
    except Exception:
        # Avoid recursion by not logging through loguru here.
        return

    loki_logger = logging.getLogger("optionscanner.loki")
    loki_logger.setLevel(logging.INFO)
    loki_logger.handlers = [handler]
    loki_logger.propagate = False

    def _loki_sink(message) -> None:
        record = message.record
        # Build structured log entry for Loki
        structured_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "logger": record["name"],
            "line": record["line"],
            "component": record["extra"].get("component", "app"),
            "event_type": record["extra"].get("event_type"),
            "symbol": record["extra"].get("symbol"),
            "strategy": record["extra"].get("strategy"),
            "direction": record["extra"].get("direction"),
            "run_mode": run_mode_label,
        }
        # Filter out None values
        structured_entry = {k: v for k, v in structured_entry.items() if v is not None}

        try:
            loki_logger.log(
                record["level"].no,
                json.dumps(structured_entry),
                extra=record["extra"],
            )
        except Exception:
            pass  # swallow to avoid impacting application logging

    logger.add(_loki_sink, level="INFO", enqueue=True)
    logger.info(
        "Loki logging enabled | endpoint={endpoint} user={user}",
        endpoint=endpoint,
        user=username,
        component="logging",
        event_type="loki_configured",
    )

    # Ensure stdlib logging messages also reach Loki.
    root_logger = logging.getLogger()
    has_loki_handler = any(isinstance(h, LokiHandler) for h in root_logger.handlers)
    if not has_loki_handler:
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)


class LoggingContext:
    """Context manager for binding structured fields to log messages.

    Usage:
        with LoggingContext(symbol="AAPL", strategy="PutCreditSpread"):
            logger.info("Processing signal")
    """

    def __init__(self, **kwargs: Any):
        self.bound_logger = logger
        self.fields = kwargs

    def __enter__(self):
        self.bound_logger = logger.bind(**self.fields)
        return self.bound_logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context cleanup - loguru handles this automatically
        pass


def log_context(**kwargs: Any):
    """Decorator or context manager for adding structured context to logs.

    Usage as decorator:
        @log_context(component="ai_selection")
        def select_signals(signals):
            ...

    Usage as context manager:
        with log_context(symbol="AAPL"):
            logger.info("Processing")
    """
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            with LoggingContext(**kwargs):
                return func(*args, **func_kwargs)
        return wrapper
    return decorator


__all__ = [
    "configure_logging",
    "get_logger",
    "LoggingContext",
    "log_context",
]
