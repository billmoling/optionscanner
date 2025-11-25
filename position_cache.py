"""Lightweight cache linking strategy signals to open positions for exit checks."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from loguru import logger

from option_data import OptionChainSnapshot
from strategies.base import TradeSignal

ExitEvaluator = Callable[["CachedPosition", OptionChainSnapshot, datetime], Optional[str]]


@dataclass(slots=True)
class CachedPosition:
    """Represents a tracked signal/position pair."""

    uid: str
    strategy: str
    symbol: str
    direction: str
    option_type: str
    strike: float
    expiry: str
    opened_at: str
    rationale: str
    status: str = "open"
    context: Dict[str, object] = field(default_factory=dict)
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def key(self) -> str:
        return f"{self.symbol.upper()}::{self.direction}::{self.option_type.upper()}::{self.strike:.4f}::{self.expiry}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "uid": self.uid,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "direction": self.direction,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry,
            "opened_at": self.opened_at,
            "rationale": self.rationale,
            "status": self.status,
            "context": self.context,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "CachedPosition":
        return cls(
            uid=str(payload.get("uid")) or str(uuid.uuid4()),
            strategy=str(payload.get("strategy", "")),
            symbol=str(payload.get("symbol", "")),
            direction=str(payload.get("direction", "")),
            option_type=str(payload.get("option_type", "")),
            strike=float(payload.get("strike", 0.0) or 0.0),
            expiry=str(payload.get("expiry", "")),
            opened_at=str(payload.get("opened_at", datetime.now(timezone.utc).isoformat())),
            rationale=str(payload.get("rationale", "")),
            status=str(payload.get("status", "open")),
            context=dict(payload.get("context", {}) or {}),
            last_seen=str(payload.get("last_seen", datetime.now(timezone.utc).isoformat())),
        )


@dataclass(slots=True)
class ExitRecommendation:
    symbol: str
    strategy: str
    direction: str
    strike: float
    expiry: str
    reason: str
    action: str


class PositionCache:
    """Persists signal metadata and re-evaluates exits during each run."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: Dict[str, CachedPosition] = {}
        self._evaluators: Dict[str, ExitEvaluator] = self._build_default_evaluators()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Unable to load position cache | path={path} reason={error}", path=str(self._path), error=exc)
            return
        for entry in payload:
            cached = CachedPosition.from_dict(entry)
            self._entries[cached.key()] = cached

    def save(self) -> None:
        data = [entry.to_dict() for entry in self._entries.values()]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------
    def record_signal(
        self,
        strategy_name: str,
        signal: TradeSignal,
        snapshot: Optional[OptionChainSnapshot] = None,
    ) -> None:
        key = self._entry_key(signal)
        now = datetime.now(timezone.utc).isoformat()
        context: Dict[str, object] = {}
        if snapshot is not None:
            context["underlying_price"] = float(snapshot.underlying_price)
            context["snapshot_ts"] = snapshot.timestamp.isoformat()
        entry = self._entries.get(key)
        if entry:
            entry.last_seen = now
            entry.context.update(context)
            logger.debug(
                "Updated cached position | symbol={symbol} direction={direction} strategy={strategy}",
                symbol=signal.symbol,
                direction=signal.direction,
                strategy=strategy_name,
            )
            return
        cached = CachedPosition(
            uid=str(uuid.uuid4()),
            strategy=strategy_name,
            symbol=signal.symbol,
            direction=signal.direction,
            option_type=signal.option_type,
            strike=float(signal.strike),
            expiry=signal.expiry.isoformat(),
            opened_at=now,
            rationale=signal.rationale,
            context=context,
        )
        self._entries[key] = cached
        logger.info(
            "Cached new position | symbol={symbol} strategy={strategy} direction={direction}",
            symbol=signal.symbol,
            strategy=strategy_name,
            direction=signal.direction,
        )

    def reconcile_with_positions(self, contract_keys: Iterable[str]) -> None:
        """Mark cache entries as closed when the position no longer exists."""

        active = {key.upper() for key in contract_keys}
        for key, entry in list(self._entries.items()):
            if entry.status != "open":
                continue
            if entry.key().upper() not in active:
                entry.status = "closed"
                entry.last_seen = datetime.now(timezone.utc).isoformat()
                logger.info(
                    "Marking cached position as closed | symbol={symbol} strategy={strategy}",
                    symbol=entry.symbol,
                    strategy=entry.strategy,
                )

    # ------------------------------------------------------------------
    def register_evaluator(self, direction: str, evaluator: ExitEvaluator) -> None:
        self._evaluators[direction.upper()] = evaluator

    def evaluate_exits(
        self,
        snapshots: Mapping[str, OptionChainSnapshot],
        *,
        now: Optional[datetime] = None,
    ) -> List[ExitRecommendation]:
        now = now or datetime.now(timezone.utc)
        recommendations: List[ExitRecommendation] = []
        for entry in self._entries.values():
            if entry.status != "open":
                continue
            snapshot = snapshots.get(entry.symbol.upper())
            if snapshot is None:
                continue
            reason = self._exit_reason(entry, snapshot, now)
            if reason is None:
                continue
            recommendations.append(
                ExitRecommendation(
                    symbol=entry.symbol,
                    strategy=entry.strategy,
                    direction=entry.direction,
                    strike=entry.strike,
                    expiry=entry.expiry,
                    reason=reason,
                    action="CONSIDER_EXIT",
                )
            )
        return recommendations

    # ------------------------------------------------------------------
    def _entry_key(self, signal: TradeSignal) -> str:
        expiry = signal.expiry if isinstance(signal.expiry, datetime) else datetime.fromisoformat(str(signal.expiry))
        expiry_iso = expiry if isinstance(expiry, str) else expiry.isoformat()
        return f"{signal.symbol.upper()}::{signal.direction}::{signal.option_type.upper()}::{float(signal.strike):.4f}::{expiry_iso}"

    def _exit_reason(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        direction = entry.direction.upper()
        evaluator = self._evaluators.get(direction)
        if evaluator is None:
            return self._fallback_exit(entry, snapshot, now)
        return evaluator(entry, snapshot, now)

    # ------------------------------------------------------------------
    @staticmethod
    def _days_to_expiry(entry: CachedPosition, now: datetime) -> int:
        try:
            expiry = datetime.fromisoformat(entry.expiry)
        except ValueError:
            return 0
        return max((expiry - now).days, 0)

    @staticmethod
    def _underlying(snapshot: OptionChainSnapshot) -> float:
        try:
            return float(snapshot.underlying_price)
        except (TypeError, ValueError):
            return 0.0

    def _build_default_evaluators(self) -> Dict[str, ExitEvaluator]:
        return {
            "BULL_PUT_CREDIT_SPREAD": self._exit_bull_put_credit,
            "BEAR_CALL_CREDIT_SPREAD": self._exit_bear_call_credit,
            "BULL_CALL_DEBIT_SPREAD": self._exit_bull_call_debit,
            "BEAR_PUT_DEBIT_SPREAD": self._exit_bear_put_debit,
            "SHORT_CALL": self._exit_short_call,
            "SHORT_PMCC_CALL": self._exit_short_call,
            "LONG_PMCC_LEAPS": self._exit_long_pmcc_leaps,
            "SHORT_CONDOR_CALL": self._exit_short_call,
            "SHORT_CONDOR_PUT": self._exit_short_put,
            "LONG_CONDOR_CALL": self._exit_long_condor_legs,
            "LONG_CONDOR_PUT": self._exit_long_condor_legs,
        }

    # Evaluator implementations -------------------------------------------------
    def _exit_bull_put_credit(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        underlying = self._underlying(snapshot)
        dte = self._days_to_expiry(entry, now)
        if underlying <= entry.strike:
            return f"Underlying {underlying:.2f} below short strike {entry.strike:.2f}"
        if dte <= 5:
            return f"DTE {dte} below management threshold"
        return None

    def _exit_bear_call_credit(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        underlying = self._underlying(snapshot)
        dte = self._days_to_expiry(entry, now)
        if underlying >= entry.strike:
            return f"Underlying {underlying:.2f} above short strike {entry.strike:.2f}"
        if dte <= 5:
            return f"DTE {dte} below management threshold"
        return None

    def _exit_bull_call_debit(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        underlying = self._underlying(snapshot)
        dte = self._days_to_expiry(entry, now)
        if underlying >= entry.strike:
            return f"Underlying {underlying:.2f} reached strike {entry.strike:.2f}"
        if dte <= 3:
            return f"DTE {dte} below management threshold"
        return None

    def _exit_bear_put_debit(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        underlying = self._underlying(snapshot)
        dte = self._days_to_expiry(entry, now)
        if underlying <= entry.strike:
            return f"Underlying {underlying:.2f} reached strike {entry.strike:.2f}"
        if dte <= 3:
            return f"DTE {dte} below management threshold"
        return None

    def _exit_short_call(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        underlying = self._underlying(snapshot)
        dte = self._days_to_expiry(entry, now)
        if underlying >= entry.strike * 1.02:
            return f"Underlying {underlying:.2f} above risk threshold near strike {entry.strike:.2f}"
        if dte <= 5:
            return f"DTE {dte} below management threshold"
        return None

    def _exit_short_put(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        underlying = self._underlying(snapshot)
        dte = self._days_to_expiry(entry, now)
        if underlying <= entry.strike * 0.98:
            return f"Underlying {underlying:.2f} below risk threshold near strike {entry.strike:.2f}"
        if dte <= 5:
            return f"DTE {dte} below management threshold"
        return None

    def _exit_long_pmcc_leaps(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        underlying = self._underlying(snapshot)
        dte = self._days_to_expiry(entry, now)
        if underlying <= entry.strike * 0.9:
            return f"Underlying {underlying:.2f} broke below collar strike {entry.strike:.2f}"
        if dte <= 60:
            return f"LEAPS nearing expiry (DTE {dte})"
        return None

    def _exit_long_condor_legs(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        dte = self._days_to_expiry(entry, now)
        if dte <= 5:
            return f"DTE {dte} below management threshold"
        return None

    def _fallback_exit(
        self,
        entry: CachedPosition,
        snapshot: OptionChainSnapshot,
        now: datetime,
    ) -> Optional[str]:
        dte = self._days_to_expiry(entry, now)
        if dte <= 3:
            return f"DTE {dte} below default management threshold"
        return None


__all__ = ["PositionCache", "ExitRecommendation"]
