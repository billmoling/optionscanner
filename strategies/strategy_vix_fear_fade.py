"""VIX fear fade strategy implementation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, List, Optional, Tuple

from loguru import logger

from .base import BaseOptionStrategy, TradeSignal


@dataclass
class _VixObservation:
    session_date: date
    close: float


class VixFearFadeStrategy(BaseOptionStrategy):
    """Buy Nasdaq exposure after VIX cools down from a fear spike.

    Logic:
    1) Wait for VIX to reach or exceed the fear threshold (default 35).
    2) Once it falls below the threshold and stays there for N sessions,
       emit a buy signal on the final confirmation session.
    """

    def __init__(
        self,
        vix_symbol: str = "VIX",
        target_symbol: str = "QQQ",
        vix_threshold: float = 35.0,
        confirmation_sessions: int = 5,
        enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vix_symbol = vix_symbol.upper()
        self.target_symbol = target_symbol.upper()
        self.vix_threshold = float(vix_threshold)
        self.confirmation_sessions = max(int(confirmation_sessions), 1)
        self.enabled = enabled
        self._vix_history: List[_VixObservation] = []
        self._last_spike_date: Optional[date] = None
        self._last_signal_date: Optional[date] = None

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        if not self.enabled:
            logger.debug("VixFearFadeStrategy disabled; skipping evaluation")
            return []

        vix_snapshot = self._find_snapshot(data, self.vix_symbol)
        if vix_snapshot is None:
            logger.debug("No VIX snapshot found; skipping evaluation")
            return []

        vix_close = self._resolve_underlying_price(vix_snapshot, None)
        if vix_close is None:
            logger.debug("Unable to resolve VIX close; skipping evaluation")
            return []

        session_date = self._resolve_session_date(vix_snapshot)
        self._record_vix_close(session_date, vix_close)

        if vix_close >= self.vix_threshold:
            self._last_spike_date = session_date

        if not self._last_spike_date:
            return []

        if len(self._vix_history) < self.confirmation_sessions:
            return []

        recent = self._vix_history[-self.confirmation_sessions :]
        if not all(obs.close < self.vix_threshold for obs in recent):
            return []

        first_below_date = recent[0].session_date
        if self._last_spike_date >= first_below_date:
            return []

        trigger_date = recent[-1].session_date
        if self._last_signal_date == trigger_date:
            return []

        target_snapshot = self._find_snapshot(data, self.target_symbol)
        underlying_price = self._resolve_underlying_price(target_snapshot, None) if target_snapshot else None
        strike = float(underlying_price) if underlying_price else 0.0
        rationale = (
            f"VIX cooled below {self.vix_threshold:.0f} for {self.confirmation_sessions} sessions "
            f"after a fear spike; latest VIX {vix_close:.2f}."
        )
        signal = self.emit_signal(
            TradeSignal(
                symbol=self.target_symbol,
                expiry=datetime.now(timezone.utc),
                strike=strike,
                option_type="STOCK",
                direction="LONG_UNDERLYING",
                rationale=rationale,
            )
        )
        self._last_signal_date = trigger_date
        return [signal]

    def _find_snapshot(self, data: Iterable[Any], symbol: str) -> Optional[Any]:
        for snapshot in data:
            snapshot_symbol = self._snapshot_value(snapshot, "symbol")
            if snapshot_symbol and str(snapshot_symbol).upper() == symbol:
                return snapshot
        return None

    def _resolve_session_date(self, snapshot: Any) -> date:
        timestamp = self._snapshot_value(snapshot, "timestamp")
        if timestamp is None:
            return datetime.now(timezone.utc).date()
        if isinstance(timestamp, datetime):
            return timestamp.date()
        try:
            return datetime.fromisoformat(str(timestamp)).date()
        except ValueError:
            return datetime.now(timezone.utc).date()

    def _record_vix_close(self, session_date: date, vix_close: float) -> None:
        if self._vix_history and self._vix_history[-1].session_date == session_date:
            self._vix_history[-1] = _VixObservation(session_date=session_date, close=vix_close)
            return
        self._vix_history.append(_VixObservation(session_date=session_date, close=vix_close))


__all__ = ["VixFearFadeStrategy"]
