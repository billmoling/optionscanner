"""Slack notification utilities for formatted trade signal delivery."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yaml
from loguru import logger


PostCallable = Callable[[str, Dict[str, object]], None]


@dataclass(slots=True)
class SlackSettings:
    """Slack webhook configuration."""

    webhook_url: str
    username: Optional[str] = None
    channel: Optional[str] = None
    icon_emoji: Optional[str] = None
    timeout: float = 10.0


class SlackNotifier:
    """Formats DataFrame results and delivers them via Slack webhooks."""

    def __init__(self, config: Optional[dict], post: Optional[PostCallable] = None) -> None:
        config = config or {}
        self.enabled: bool = bool(config.get("enabled", False))
        self.title: str = config.get("title", "Option Scanner Signals")
        self.max_rows: int = int(config.get("max_rows", 10))
        webhook_url = self._resolve_webhook_url(config)
        settings = SlackSettings(
            webhook_url=webhook_url,
            username=config.get("username"),
            channel=config.get("channel"),
            icon_emoji=config.get("icon_emoji"),
            timeout=float(config.get("timeout", 10.0)),
        )
        self.settings = settings
        self._post: PostCallable = post or self._post_to_slack

    def send_signals(self, df: pd.DataFrame, csv_path: Optional[Path] = None) -> None:
        """Send each signal as an individual Slack message when enabled."""
        if not self.enabled:
            logger.debug("Slack notifications are disabled; skipping send.")
            return
        if not self.settings.webhook_url:
            logger.warning("Slack webhook URL is not configured; skipping notification.")
            return
        if df.empty:
            logger.info("No signals to send to Slack.")
            return

        for _, row in df.iterrows():
            message = self._build_signal_message(row, csv_path)
            payload = self._build_payload(message)
            try:
                self._post(self.settings.webhook_url, payload)
                logger.info(
                    "Sent Slack notification for symbol={symbol} strategy={strategy}",
                    symbol=row.get("symbol"),
                    strategy=row.get("strategy"),
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to send Slack notification | symbol={symbol} error={error}",
                    symbol=row.get("symbol"),
                    error=exc,
                )

    def _build_payload(self, message: str) -> Dict[str, object]:
        payload: Dict[str, object] = {"text": message}
        if self.settings.username:
            payload["username"] = self.settings.username
        if self.settings.channel:
            payload["channel"] = self.settings.channel
        if self.settings.icon_emoji:
            payload["icon_emoji"] = self.settings.icon_emoji
        return payload

    def _build_signal_message(self, row: pd.Series, csv_path: Optional[Path]) -> str:
        lines: List[str] = [self.title]
        summary_fields = {
            "Symbol": row.get("symbol"),
            "Strategy": row.get("strategy"),
            "Action": row.get("action") or row.get("direction"),
            "Option": f"{row.get('option_type')} {row.get('strike')} exp {row.get('expiry')}".strip(),
            "Confidence": row.get("confidence"),
        }
        for label, value in summary_fields.items():
            if value not in (None, ""):
                lines.append(f"{label}: {value}")
        explanation = row.get("explanation")
        if explanation:
            lines.append("")
            lines.append("Explanation:")
            lines.append(str(explanation))
        validation = row.get("validation")
        if validation:
            lines.append("")
            lines.append("Validation:")
            lines.append(str(validation))
        if csv_path:
            lines.append("")
            lines.append(f"CSV saved to {csv_path}")
        return "\n".join(lines)

    def _post_to_slack(self, url: str, payload: Dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.settings.timeout) as response:
                status = getattr(response, "status", response.getcode())
                if status < 200 or status >= 300:
                    raise RuntimeError(f"Slack webhook responded with status {status}")
        except (HTTPError, URLError) as exc:
            raise RuntimeError("Slack webhook request failed") from exc

    def _resolve_webhook_url(self, config: Dict[str, object]) -> str:
        configured = str(config.get("webhook_url", "") or "").strip()
        if configured:
            return configured

        env_value = os.getenv("SLACK_WEBHOOK_URL", "").strip()
        if env_value:
            return env_value

        secret_value = self._load_webhook_from_secrets()
        if secret_value:
            return secret_value

        return ""

    def _load_webhook_from_secrets(self) -> str:
        candidates: Iterable[Path] = (
            Path("config/secrets.yaml"),
            Path("secrets.yaml"),
            Path(".secrets.yaml"),
        )
        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Unable to read Slack webhook secret file | path={path} reason={error}",
                    path=str(path),
                    error=exc,
                )
                continue
            webhook_url = self._extract_webhook_from_data(data)
            if webhook_url:
                return webhook_url
        return ""

    def _extract_webhook_from_data(self, data: object) -> str:
        if not isinstance(data, dict):
            return ""

        # Common layouts: top-level key or nested under "slack"/"notifications"
        direct = data.get("slack_webhook_url")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        for key in ("slack", "notifications"):
            section = data.get(key)
            if not isinstance(section, dict):
                continue
            value = section.get("webhook_url") or section.get("slack_webhook_url")
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""
