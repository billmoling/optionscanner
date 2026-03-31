"""Slack notification utilities for formatted trade signal delivery."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yaml
from loguru import logger

from optionscanner.signal_ranking import SignalScore

if TYPE_CHECKING:
    from optionscanner.market_context import MarketContextProvider


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

    def send_ranked_signals(
        self,
        ranked_signals: List[SignalScore],
        csv_path: Optional[Path] = None,
        market_context: Optional[MarketContextProvider] = None,
    ) -> None:
        """Send top ranked signals as a formatted Slack message when enabled."""
        if not self.enabled:
            logger.debug("Slack notifications are disabled; skipping send.")
            return
        if not self.settings.webhook_url:
            logger.warning("Slack webhook URL is not configured; skipping notification.")
            return
        if not ranked_signals:
            logger.info("No ranked signals to send to Slack.")
            return

        message = self._build_ranked_message(ranked_signals, csv_path, market_context)
        payload = self._build_payload(message)
        try:
            self._post(self.settings.webhook_url, payload)
            logger.info("Sent {count} ranked signals to Slack", count=len(ranked_signals))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to send ranked signals to Slack | error={error}", error=exc)

    def send_ai_and_quant_signals(
        self,
        ai_selections: List[tuple],
        ai_reasons: Dict[str, str],
        quant_picks: List[SignalScore],
        csv_path: Optional[Path] = None,
        market_context: Optional[MarketContextProvider] = None,
    ) -> None:
        """Send AI picks (5) and quantitative picks (5) as a formatted Slack message.

        Args:
            ai_selections: List of (strategy_name, signal) tuples from AI selection
            ai_reasons: Dict mapping signal_id to AI reason
            quant_picks: List of SignalScore from quantitative ranking
            csv_path: Optional path to CSV file
            market_context: Optional market context provider
        """
        if not self.enabled:
            logger.debug("Slack notifications are disabled; skipping send.")
            return
        if not self.settings.webhook_url:
            logger.warning("Slack webhook URL is not configured; skipping notification.")
            return
        if not ai_selections and not quant_picks:
            logger.info("No signals to send to Slack.")
            return

        message = self._build_ai_and_quant_message(
            ai_selections, ai_reasons, quant_picks, csv_path, market_context
        )
        payload = self._build_payload(message)
        try:
            self._post(self.settings.webhook_url, payload)
            total = len(ai_selections) + len(quant_picks)
            logger.info("Sent {count} signals to Slack (AI={ai} + Quant={quant})",
                       count=total, ai=len(ai_selections), quant=len(quant_picks))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to send AI+quant signals to Slack | error={error}", error=exc)

    def _build_ai_and_quant_message(
        self,
        ai_selections: List[tuple],
        ai_reasons: Dict[str, str],
        quant_picks: List[SignalScore],
        csv_path: Optional[Path] = None,
        market_context: Optional[MarketContextProvider] = None,
    ) -> str:
        """Build Slack message with AI picks and quantitative picks sections."""
        from datetime import datetime, timezone

        lines: List[str] = []

        # Header with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"*{self.title}* | {timestamp}")
        lines.append("")

        # Market context header (if available)
        if market_context:
            context_result = market_context.get_context()
            if context_result:
                context_lines = self._format_market_context(context_result)
                lines.extend(context_lines)
                lines.append("")

        # AI Picks section
        if ai_selections:
            lines.append("*-- AI Top Picks --*")
            lines.append("| # | Symbol | Direction | Strategy | AI Reason |")
            lines.append("|---|----------|-----------|------------------|------------|")

            for idx, (strategy_name, signal) in enumerate(ai_selections, start=1):
                direction = signal.direction.replace("_", " ").title()
                signal_id = f"{signal.symbol}_{strategy_name}"
                ai_reason = ai_reasons.get(signal_id, "AI selected")
                # Truncate reason if too long
                if len(ai_reason) > 60:
                    ai_reason = ai_reason[:57] + "..."
                lines.append(
                    f"| {idx} | {signal.symbol} | {direction} | "
                    f"{strategy_name.replace('Strategy', '')} | {ai_reason} |"
                )
            lines.append("")

        # Quantitative Picks section
        if quant_picks:
            lines.append("*-- Quantitative Top Picks --*")
            lines.append("| # | Symbol | Direction | Strategy | Score | Reason |")
            lines.append("|---|----------|-----------|------------------|-------|--------|")

            for idx, score in enumerate(quant_picks, start=1):
                direction = score.signal.direction.replace("_", " ").title()
                reason_short = score.reason[:50] + "..." if len(score.reason) > 50 else score.reason
                lines.append(
                    f"| {idx} | {score.signal.symbol} | {direction} | "
                    f"{score.strategy_name.replace('Strategy', '')} | "
                    f"{score.composite_score:.2f} | {reason_short} |"
                )

        # Footer
        if csv_path:
            lines.append("")
            lines.append(f"Full results: `{csv_path}`")

        return "\n".join(lines)

    def _build_ranked_message(
        self,
        ranked_signals: List[SignalScore],
        csv_path: Optional[Path] = None,
        market_context: Optional[MarketContextProvider] = None,
    ) -> str:
        """Build a formatted Slack message with ranked signals table."""
        from datetime import datetime, timezone

        lines: List[str] = []

        # Header with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"*{self.title}* | {timestamp}")
        lines.append("")

        # Market context header (if available)
        if market_context:
            context_result = market_context.get_context()
            if context_result:
                context_lines = self._format_market_context(context_result)
                lines.extend(context_lines)
                lines.append("")

        # Table header
        lines.append("| # | Symbol | Direction | Strategy | Score | Reason |")
        lines.append("|---|----------|-----------|------------------|-------|--------|")

        # Ranked signals
        for idx, score in enumerate(ranked_signals, start=1):
            direction = score.signal.direction.replace("_", " ").title()
            reason_short = score.reason[:50] + "..." if len(score.reason) > 50 else score.reason
            lines.append(
                f"| {idx} | {score.signal.symbol} | {direction} | "
                f"{score.strategy_name.replace('Strategy', '')} | "
                f"{score.composite_score:.2f} | {reason_short} |"
            )

        # Footer
        if csv_path:
            lines.append("")
            lines.append(f"Full results: `{csv_path}`")

        return "\n".join(lines)

    def _format_market_context(self, context) -> List[str]:
        """Format market context result into Slack-friendly lines.

        Args:
            context: MarketContextResult from MarketContextProvider

        Returns:
            List of formatted lines for Slack message
        """
        lines: List[str] = []

        # VIX line
        if context.vix:
            vix_emoji = {
                "LOW": ":green_circle:",
                "NORMAL": ":white_circle:",
                "HIGH": ":orange_circle:",
                "EXTREME": ":red_circle:",
            }.get(context.vix.state, ":white_circle:")
            lines.append(f"VIX: {context.vix.level:.1f} {vix_emoji} ({context.vix.state})")

        # Market states
        state_emoji = {
            "bull": ":green_circle:",
            "uptrend": ":large_blue_circle:",
            "bear": ":red_circle:",
        }

        if context.spy_state:
            emoji = state_emoji.get(context.spy_state.value, "")
            lines.append(f"SPY: {context.spy_state.value.upper()} {emoji}")

        if context.qqq_state:
            emoji = state_emoji.get(context.qqq_state.value, "")
            lines.append(f"QQQ: {context.qqq_state.value.upper()} {emoji}")

        # Context score
        score = context.context_score
        if score >= 0.7:
            score_emoji = ":green_circle:"
        elif score >= 0.4:
            score_emoji = ":yellow_circle:"
        else:
            score_emoji = ":red_circle:"
        lines.append(f"Context Score: {score:.2f} {score_emoji}")

        # Earnings watch
        if context.earnings_map:
            earnings_str = ", ".join(
                f"{s}({d}d)" for s, d in sorted(context.earnings_map.items(), key=lambda x: x[1])[:5]
            )
            lines.append(f"Earnings: {earnings_str}")

        # Economic events
        if context.economic_events:
            from datetime import date
            events_str = ", ".join(
                f"{e.event_type}({(e.date - date.today()).days}d)"
                for e in context.economic_events[:3]
            )
            lines.append(f"Events: {events_str}")

        # Warnings
        if context.warnings:
            lines.append(f":warning: {context.warnings[0]}")

        return lines

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
        legs = row.get("legs")
        leg_lines = self._format_legs(legs)
        if leg_lines:
            lines.append("")
            lines.append("Legs:")
            lines.extend(leg_lines)
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
        if configured and not self._is_placeholder_webhook(configured):
            return configured

        env_value = os.getenv("SLACK_WEBHOOK_URL", "").strip()
        if env_value:
            return env_value

        secret_value = self._load_webhook_from_secrets()
        if secret_value:
            return secret_value

        return ""

    def _is_placeholder_webhook(self, url: str) -> bool:
        """Detect sample/placeholder webhook strings so we can fall back to real secrets."""
        return any(token in url for token in ("XXX", "YYY", "ZZ", "ZZZ"))

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
            candidate = direct.strip()
            if not self._is_placeholder_webhook(candidate):
                return candidate

        for key in ("slack", "notifications"):
            section = data.get(key)
            if not isinstance(section, dict):
                continue
            value = section.get("webhook_url") or section.get("slack_webhook_url")
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if not self._is_placeholder_webhook(candidate):
                    return candidate

        return ""

    def _format_legs(self, raw: object) -> List[str]:
        """Render a human-readable leg breakdown from DataFrame/JSON payloads."""
        legs: List[Dict[str, object]] = []
        if isinstance(raw, (list, tuple)):
            legs = [
                leg for leg in raw if isinstance(leg, dict)
            ]
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    legs = [leg for leg in parsed if isinstance(leg, dict)]
            except Exception:
                legs = []
        if not legs:
            return []
        lines: List[str] = []
        for leg in legs:
            action = str(leg.get("action", "")).upper()
            option_type = str(leg.get("option_type", "")).upper()
            strike = leg.get("strike")
            expiry = leg.get("expiry")
            parts = [action or "?", option_type or "?"]
            if strike not in (None, ""):
                parts.append(f"{strike}")
            if expiry not in (None, ""):
                parts.append(f"exp {expiry}")
            lines.append(" ".join(parts))
        return lines
