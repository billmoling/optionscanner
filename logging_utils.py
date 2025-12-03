"""Logging utilities for the option scanner project."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Optional

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
) -> None:
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
    _configure_loki_sink(run_mode=run_mode)


def get_logger(log_dir: Path, log_name: str, run_mode: Optional[str] = None):
    """Helper that configures logging and returns the global logger."""

    configure_logging(log_dir, log_name, run_mode=run_mode)
    return logger


def _configure_loki_sink(run_mode: Optional[str] = None) -> None:
    """Optionally forward logs to Grafana Cloud Loki when env vars are provided."""

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
        try:
            loki_logger.log(record["level"].no, record["message"], extra=record["extra"])
        except Exception:
            pass  # swallow to avoid impacting application logging

    logger.add(_loki_sink, level="INFO", enqueue=True)
    logger.info("Loki logging enabled | endpoint={endpoint} user={user}", endpoint=endpoint, user=username)

    # Ensure stdlib logging messages also reach Loki.
    root_logger = logging.getLogger()
    has_loki_handler = any(isinstance(h, LokiHandler) for h in root_logger.handlers)
    if not has_loki_handler:
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)
