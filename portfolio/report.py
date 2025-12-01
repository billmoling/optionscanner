"""Reporting utilities for the portfolio manager."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
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

    def write_outputs(self, positions: pd.DataFrame, greek_summary: pd.DataFrame) -> (Path, Path, str):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = self._results_dir / f"portfolio_summary_{timestamp}.csv"

        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            positions.to_csv(fh, index=False)
            fh.write("\n")
            greek_summary.to_csv(fh, index=False)
        logger.info("Wrote portfolio summary CSV to {path}", path=str(csv_path))

        return csv_path, None, timestamp

    def log_details(self, message: str) -> Path:
        path = self._logs_dir / "portfolio_manager.log"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now(timezone.utc).isoformat()}] {message}\n")
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
            payload = self._slack._build_payload(message)
            self._slack._post(self._slack.settings.webhook_url, payload)
            logger.info("Portfolio summary sent to Slack")
        except Exception as exc:  # pragma: no cover - network interaction
            logger.warning("Failed to send portfolio summary to Slack | reason={error}", error=exc)

    def _format_gemini_slack_message(self, response: str, timestamp: str) -> str:
        """Normalize Gemini output for a concise Slack post."""
        header = f"*Gemini Position Review* ({timestamp})"
        cleaned_lines: List[str] = []
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("*", "-", "•")):
                stripped = stripped.lstrip("*-•").strip()
                cleaned_lines.append(f"• {stripped}")
            else:
                cleaned_lines.append(stripped)
        body = "\n".join(cleaned_lines).strip()
        return "\n".join([header, body]) if body else header

    def _send_gemini_slack_message(self, response: str, timestamp: str) -> None:
        try:
            if not response or not response.strip():
                logger.debug("Gemini position response empty; skipping Slack send")
                return
            if not self._slack.enabled:
                logger.debug("Slack notifications disabled; Gemini position summary not sent")
                return
            if not self._slack.settings.webhook_url:
                logger.warning("Slack webhook URL missing; Gemini position summary not sent")
                return
            message = self._format_gemini_slack_message(response, timestamp)
            payload = self._slack._build_payload(message)
            self._slack._post(self._slack.settings.webhook_url, payload)
            logger.info("Gemini position summary sent to Slack")
        except Exception as exc:  # pragma: no cover - network interaction
            logger.warning("Failed to send Gemini position summary to Slack | reason={error}", error=exc)

    def _send_gemini(self, message: str) -> None:
        # Summary Gemini call disabled; keep Slack flow intact.
        logger.debug("Gemini summary skipped: disabled for portfolio summaries")
        return

    def evaluate_positions_with_gemini(self, positions: pd.DataFrame, timestamp: str) -> None:
        if positions.empty:
            logger.debug("Gemini position evaluation skipped: no positions")
            return
        logger.info(
            "Preparing Gemini position evaluation | positions={count} enable_gemini={enabled} timestamp={ts}",
            count=len(positions.index),
            enabled=self._enable_gemini,
            ts=timestamp,
        )
        system_prompt = (
            "You are an experienced option trader. The approach is conservative, prioritizing risk control; "
            "when a position is profitable we lean toward exiting if risk is elevated. Prefer quick in/out trades."
        )
        now = datetime.now(timezone.utc)
        position_lines = []
        for _, row in positions.iterrows():
            symbol = str(row.get("symbol", row.get("underlying", "")))
            right = str(row.get("right", ""))[:1].upper()
            strike = row.get("strike", "")
            qty = row.get("quantity", "")
            avg_price = float(row.get("avg_price", 0.0) or 0.0)
            mark = float(row.get("market_price", 0.0) or 0.0)
            underlying = float(row.get("underlying_price", row.get("market_price", 0.0)) or 0.0)
            expiry_raw = row.get("expiry", "")
            try:
                expiry_ts = pd.to_datetime(expiry_raw, utc=True)
                dte = max(int((expiry_ts - now).days), 0)
                expiry = expiry_ts.date().isoformat()
            except Exception:
                dte = "?"
                expiry = str(expiry_raw)
            try:
                multiplier = float(row.get("multiplier", 1.0) or 1.0)
            except Exception:
                multiplier = 1.0
            try:
                pnl = (mark - avg_price) * float(qty or 0) * multiplier
            except Exception:
                pnl = "?"
            position_lines.append(
                f"{symbol} {right}{strike} expiry {expiry} DTE {dte} qty {qty} avg {avg_price:.2f} market {mark:.2f} underlying {underlying:.2f} pnl {pnl}"
            )
        legend = (
            "Format per line: SYMBOL RIGHT STRIKE exp YYYY-MM-DD DTE <days> qty <signed contracts> "
            "avg <entry> mark <current> underlying <spot> pnl <est unrealized>"
        )
        user_prompt = "\n".join(
            [
                "Review each option position and suggest an action (roll/close/hold/adjust) with rationale <80 words."
                " Consider mark, DTE, and risk; prefer quick exits when profit is at risk.",
                legend,
                "Positions:",
                *[f"- {line}" for line in position_lines],
            ]
        )
        logger.debug(
            "Gemini position prompt assembled | positions={count} prompt_chars={prompt_len}",
            count=len(position_lines),
            prompt_len=len(user_prompt),
        )

        prompt_path = self._results_dir / f"gemini_positions_{timestamp}.prompt.txt"
        try:
            prompt_path.write_text(
                "System prompt:\n" + system_prompt + "\n\nUser prompt:\n" + user_prompt,
                encoding="utf-8",
            )
            logger.info(
                "Wrote Gemini positions prompt | path={path} positions={count}",
                path=str(prompt_path),
                count=len(position_lines),
            )
        except Exception as exc:
            logger.warning("Failed to write Gemini positions prompt | reason={error}", error=exc)

        if not self._enable_gemini:
            logger.debug("Gemini position evaluation skipped: disabled in config")
            return
        try:
            if self._gemini is None:
                logger.debug("Initialising Gemini client for portfolio evaluation")
                self._gemini = GeminiClient()
            logger.info(
                "Calling Gemini for position actions | positions={count} prompt_chars={prompt_len}",
                count=len(position_lines),
                prompt_len=len(user_prompt),
            )
            response = self._gemini.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            logger.info(
                "Gemini position analysis received | response_chars={response_len}",
                response_len=len(response or ""),
            )
            logger.debug("Gemini position response: {response}", response=response)
            result_path = self._results_dir / f"gemini_positions_{timestamp}.json"
            result_payload = {
                "as_of": timestamp,
                "prompt": user_prompt,
                "response": response,
            }
            result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            self._send_gemini_slack_message(response, timestamp)
        except GeminiClientError as exc:
            logger.warning("Gemini client unavailable for position analysis: {error}", error=exc)
        except Exception as exc:
            logger.warning("Gemini position analysis failed: {error}", error=exc)


__all__ = ["PortfolioReporter", "ReporterConfig"]
