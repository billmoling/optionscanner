"""Email notification utilities for formatted trade signal delivery."""
from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, List, Optional
import smtplib

import pandas as pd
from loguru import logger


@dataclass(slots=True)
class SMTPSettings:
    """SMTP connectivity settings."""

    host: str
    port: int
    use_ssl: bool = True
    use_tls: bool = False
    username: Optional[str] = None
    password: Optional[str] = None


class EmailNotifier:
    """Formats DataFrame results and delivers them via email."""

    def __init__(self, config: Optional[dict]) -> None:
        config = config or {}
        self.enabled: bool = bool(config.get("enabled", False))
        self.sender: Optional[str] = config.get("sender")
        recipients: Iterable[str] = config.get("recipients", []) or []
        self.recipients: List[str] = [r for r in recipients if r]
        smtp_config = config.get("smtp", {})
        self.settings = SMTPSettings(
            host=smtp_config.get("host", ""),
            port=int(smtp_config.get("port", 465)),
            use_ssl=bool(smtp_config.get("use_ssl", True)),
            use_tls=bool(smtp_config.get("use_tls", False)),
            username=smtp_config.get("username"),
            password=smtp_config.get("password"),
        )
        self.subject_template = config.get("subject", "Option Scanner Signals")
        if self.settings.use_tls:
            # TLS and SSL are mutually exclusive. If TLS is requested, disable SSL.
            self.settings.use_ssl = False

    def send_signals(self, df: pd.DataFrame, csv_path: Path) -> None:
        """Send the provided signals via email if enabled."""
        if not self.enabled:
            logger.debug("Email notifications are disabled; skipping send.")
            return
        if not self.sender or not self.recipients:
            logger.warning("Email notifications enabled but sender/recipients missing.")
            return
        if df.empty:
            logger.info("No signals to email.")
            return
        if not self.settings.host:
            logger.warning("SMTP host is not configured; skipping email send.")
            return

        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        subject = self.subject_template
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            latest = df["timestamp"].iloc[0]
            subject = f"{subject} - {latest}"
        message["Subject"] = subject

        text_summary = self._build_text_summary(df)
        html_body = self._build_html_body(df)

        message.set_content(text_summary)
        message.add_alternative(html_body, subtype="html")

        if csv_path and csv_path.exists():
            message.add_attachment(
                csv_path.read_bytes(),
                maintype="text",
                subtype="csv",
                filename=csv_path.name,
            )

        try:
            self._deliver(message)
            logger.info(
                "Sent email with {count} signals to {recipients}",
                count=len(df),
                recipients=message["To"],
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to send email notification: {error}", error=exc)

    def _deliver(self, message: EmailMessage) -> None:
        settings = self.settings
        if settings.use_ssl:
            smtp: smtplib.SMTP = smtplib.SMTP_SSL(settings.host, settings.port, timeout=10)
        else:
            smtp = smtplib.SMTP(settings.host, settings.port, timeout=10)
            if settings.use_tls:
                smtp.starttls()
        with smtp:
            if settings.username and settings.password:
                smtp.login(settings.username, settings.password)
            smtp.send_message(message)

    @staticmethod
    def _build_text_summary(df: pd.DataFrame) -> str:
        lines = ["Trade Signals Summary:"]
        summary_columns = [col for col in ["symbol", "strategy", "action", "confidence"] if col in df.columns]
        for _, row in df.iterrows():
            parts = [str(row.get(col, "")) for col in summary_columns]
            lines.append(" - ".join(filter(None, parts)))
        return "\n".join(lines)

    @staticmethod
    def _build_html_body(df: pd.DataFrame) -> str:
        html_table = df.to_html(index=False, justify="center", border=0)
        return (
            "<html><body>"
            "<p>Automated trade signals are attached below:</p>"
            f"{html_table}"
            "</body></html>"
        )
