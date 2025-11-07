"""Reporting utilities for the portfolio manager."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger

from gemini_client import GeminiClient, GeminiClientError
from notifications import SlackNotifier

from .rules import RiskBreach


@dataclass(slots=True)
class ReporterConfig:
    """Configuration for the portfolio reporter."""

    results_dir: Path = Path("./results")
    logs_dir: Path = Path("./logs")
    slack_config: Optional[dict] = None
    enable_gemini: bool = True


class PortfolioReporter:
    """Formats summary output and sends notifications."""

    def __init__(self, config: ReporterConfig) -> None:
        self._config = config
        self._results_dir = config.results_dir
        self._logs_dir = config.logs_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._slack = SlackNotifier(config.slack_config)
        self._gemini: Optional[GeminiClient] = None
        self._enable_gemini = config.enable_gemini

    def build_summary_message(
        self,
        totals: Dict[str, float],
        concentration: pd.DataFrame,
        breaches: Iterable[RiskBreach],
        actions: Sequence[str],
    ) -> str:
        greek_line = self._format_greek_totals(totals)
        concentration_line = self._format_concentration(concentration)
        breaches_lines = self._format_breaches(breaches)
        actions_lines = [f" • {action}" for action in actions] if actions else [" • None"]
        lines = [greek_line, concentration_line]
        if breaches_lines:
            lines.append(f"Breaches: {', '.join(breaches_lines)}")
        else:
            lines.append("Breaches: None")
        lines.append("Actions:")
        lines.extend(actions_lines)
        return "\n".join(lines)

    def write_csv(self, positions: pd.DataFrame, greek_summary: pd.DataFrame) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self._results_dir / f"portfolio_summary_{timestamp}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            positions.to_csv(fh, index=False)
            fh.write("\n")
            greek_summary.to_csv(fh, index=False)
        logger.info("Wrote portfolio summary CSV to {path}", path=str(path))
        return path

    def log_details(self, message: str) -> Path:
        path = self._logs_dir / "portfolio_manager.log"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.utcnow().isoformat()}] {message}\n")
        logger.debug("Appended portfolio summary to {path}", path=str(path))
        return path

    def send_notifications(self, message: str, csv_path: Optional[Path]) -> None:
        self._send_slack(message)
        self._send_gemini(message)
        if csv_path is not None:
            logger.info("Portfolio summary CSV stored at {path}", path=str(csv_path))

    def _format_greek_totals(self, totals: Dict[str, float]) -> str:
        formatted = []
        for greek in ("delta", "gamma", "theta", "vega"):
            value = float(totals.get(greek, 0.0))
            prefix = greek[0].upper()
            formatted.append(f"{prefix}: {value:+.1f}")
        return "Portfolio " + "  ".join(formatted)

    def _format_concentration(self, concentration: pd.DataFrame) -> str:
        if concentration.empty:
            return "Concentration: None"
        top = concentration.head(3)
        parts = [
            f"{row.underlying} {row.gross_pct:.0%}" for _, row in top.iterrows()
        ]
        return "Concentration: " + ", ".join(parts)

    def _format_breaches(self, breaches: Iterable[RiskBreach]) -> List[str]:
        return [breach.format_for_report() for breach in breaches]

    def _send_slack(self, message: str) -> None:
        try:
            if not self._slack.enabled:
                logger.debug("Slack notifications disabled for portfolio manager")
                return
            if not self._slack.settings.webhook_url:
                logger.warning("Slack webhook URL missing; portfolio summary not sent")
                return
            payload = {"text": message}
            self._slack._post(self._slack.settings.webhook_url, payload)
            logger.info("Portfolio summary sent to Slack")
        except Exception as exc:  # pragma: no cover - network interaction
            logger.warning("Failed to send portfolio summary to Slack | reason={error}", error=exc)

    def _send_gemini(self, message: str) -> None:
        if not self._enable_gemini:
            return
        try:
            if self._gemini is None:
                self._gemini = GeminiClient()
            system_prompt = "You are a risk manager generating concise explanations of option portfolios."
            user_prompt = (
                "Summarise the following daily risk report in plain language and flag any urgent actions:\n"
                f"{message}"
            )
            response = self._gemini.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            logger.info("Gemini explanation: {response}", response=response)
        except GeminiClientError as exc:
            logger.warning("Gemini client unavailable: {error}", error=exc)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Gemini summary failed: {error}", error=exc)


__all__ = ["PortfolioReporter", "ReporterConfig"]
