"""Slack notification utilities for formatted trade signal delivery."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
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
        settings = SlackSettings(
            webhook_url=config.get("webhook_url", ""),
            username=config.get("username"),
            channel=config.get("channel"),
            icon_emoji=config.get("icon_emoji"),
            timeout=float(config.get("timeout", 10.0)),
        )
        self.settings = settings
        self._post: PostCallable = post or self._post_to_slack

    def send_signals(self, df: pd.DataFrame, csv_path: Optional[Path] = None) -> None:
        """Send the provided signals via Slack if enabled."""
        if not self.enabled:
            logger.debug("Slack notifications are disabled; skipping send.")
            return
        if not self.settings.webhook_url:
            logger.warning("Slack webhook URL is not configured; skipping notification.")
            return
        if df.empty:
            logger.info("No signals to send to Slack.")
            return

        payload = self._build_payload(df, csv_path)
        try:
            self._post(self.settings.webhook_url, payload)
            logger.info(
                "Sent Slack notification with {count} signals to {channel}",
                count=len(df),
                channel=self.settings.channel or "configured webhook",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to send Slack notification: {error}", error=exc)

    def _build_payload(self, df: pd.DataFrame, csv_path: Optional[Path]) -> Dict[str, object]:
        text = self._build_message(df, csv_path)
        payload: Dict[str, object] = {"text": text}
        if self.settings.username:
            payload["username"] = self.settings.username
        if self.settings.channel:
            payload["channel"] = self.settings.channel
        if self.settings.icon_emoji:
            payload["icon_emoji"] = self.settings.icon_emoji
        return payload

    def _build_message(self, df: pd.DataFrame, csv_path: Optional[Path]) -> str:
        lines: List[str] = [self.title, f"Total signals: {len(df)}"]
        display_df = df.head(self.max_rows)
        summary_columns = [
            col
            for col in ["symbol", "strategy", "action", "confidence"]
            if col in display_df.columns
        ]
        if summary_columns:
            lines.append("")
            for _, row in display_df.iterrows():
                parts = [str(row.get(col, "")) for col in summary_columns]
                lines.append("• " + " - ".join(part for part in parts if part))
        if len(df) > len(display_df):
            remaining = len(df) - len(display_df)
            lines.append(f"…and {remaining} more signal{'s' if remaining != 1 else ''}.")
        if csv_path:
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
