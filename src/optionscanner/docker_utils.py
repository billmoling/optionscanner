"""Docker orchestration helpers for the option scanner."""
from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


class DockerControllerError(RuntimeError):
    """Raised when Docker orchestration fails."""


class DockerController:
    """Utility for managing docker-compose services required by the scanner."""

    def __init__(self, compose_file: Path) -> None:
        self.compose_file = compose_file

    def start_service(self, service: str) -> None:
        command = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
            "up",
            "-d",
            service,
        ]
        logger.info("Starting docker service {service} via docker compose", service=service)
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise DockerControllerError("docker command not found. Install Docker to use this mode.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else ""
            raise DockerControllerError(
                f"Failed to start docker service '{service}': {stderr.strip()}"
            ) from exc


__all__ = ["DockerController", "DockerControllerError"]
