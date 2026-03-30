"""Persistent signal history tracking for performance analysis."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from strategies.base import TradeSignal


@dataclass(slots=True)
class SignalOutcome:
    """Represents a tracked signal with its outcome."""

    signal_id: str
    strategy: str
    symbol: str
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    max_profit: Optional[float]
    max_loss: Optional[float]
    outcome: str  # "WIN", "LOSS", "OPEN", "BREAKEVEN"

    def is_closed(self) -> bool:
        return self.outcome in ("WIN", "LOSS", "BREAKEVEN")

    def is_win(self) -> bool:
        return self.outcome == "WIN"


@dataclass(slots=True)
class StrategyStats:
    """Aggregated statistics for a strategy."""

    strategy: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    open_trades: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class HistoryStore:
    """Persists and analyzes signal history for performance tracking.

    Signals are recorded when generated and updated when exited.
    Storage format: JSONL in data/signal_history.jsonl
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize history store.

        Args:
            data_dir: Directory for data storage (creates signal_history.jsonl)
        """
        self._storage_path = data_dir / "signal_history.jsonl"
        self._outcomes: List[SignalOutcome] = []
        self._stats_cache: Dict[str, StrategyStats] = {}
        self._cache_dirty = False

        # Ensure directory exists
        data_dir.mkdir(parents=True, exist_ok=True)
        if not self._storage_path.exists():
            self._storage_path.touch()

        self._load_history()

    def _load_history(self) -> None:
        """Load history from JSONL file."""
        if not self._storage_path.exists():
            return

        try:
            content = self._storage_path.read_text(encoding="utf-8")
            if not content.strip():
                return

            for line in content.strip().split("\n"):
                if line.strip():
                    data = json.loads(line)
                    self._outcomes.append(SignalOutcome(**data))

            logger.info(
                "Loaded signal history | count={count}",
                count=len(self._outcomes)
            )
        except Exception as exc:
            logger.warning("Failed to load signal history | path={path} reason={error}",
                          path=str(self._storage_path), error=exc)

    def _save_history(self) -> None:
        """Persist history to JSONL file."""
        try:
            lines = []
            for outcome in self._outcomes:
                lines.append(json.dumps(asdict(outcome)))

            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._cache_dirty = False

            logger.debug("Saved signal history | count={count}", count=len(self._outcomes))
        except Exception as exc:
            logger.warning("Failed to save signal history | reason={error}", error=exc)

    def record_signal(
        self,
        signal_id: str,
        signal: TradeSignal,
        entry_price: float,
        strategy: Optional[str] = None
    ) -> None:
        """Record a new signal entry.

        Args:
            signal_id: Unique identifier for this signal
            signal: TradeSignal object
            entry_price: Entry price paid/received
            strategy: Strategy name (defaults to signal.direction)
        """
        outcome = SignalOutcome(
            signal_id=signal_id,
            strategy=strategy or signal.direction,
            symbol=signal.symbol or "UNKNOWN",
            entry_date=datetime.now(timezone.utc).isoformat(),
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            pnl=None,
            max_profit=None,
            max_loss=None,
            outcome="OPEN"
        )

        self._outcomes.append(outcome)
        self._cache_dirty = True

        logger.info(
            "Recorded signal | id={id} strategy={strategy} symbol={symbol} entry={entry}",
            id=signal_id,
            strategy=outcome.strategy,
            symbol=outcome.symbol,
            entry=entry_price
        )

    def record_exit(
        self,
        signal_id: str,
        exit_price: float,
        exit_date: Optional[datetime] = None,
        max_profit: Optional[float] = None,
        max_loss: Optional[float] = None
    ) -> None:
        """Record exit for a signal.

        Args:
            signal_id: Signal identifier to update
            exit_price: Exit price
            exit_date: Exit timestamp (defaults to now)
            max_profit: Max unrealized profit seen during trade
            max_loss: Max unrealized loss seen during trade
        """
        for outcome in self._outcomes:
            if outcome.signal_id == signal_id:
                pnl = exit_price - outcome.entry_price

                if pnl > 0.05:  # >5% gain = WIN
                    outcome_str = "WIN"
                elif pnl < -0.05:  # <5% loss = LOSS
                    outcome_str = "LOSS"
                else:
                    outcome_str = "BREAKEVEN"

                outcome.exit_date = (exit_date or datetime.now(timezone.utc)).isoformat()
                outcome.exit_price = exit_price
                outcome.pnl = pnl
                outcome.max_profit = max_profit
                outcome.max_loss = max_loss
                outcome.outcome = outcome_str

                self._cache_dirty = True

                logger.info(
                    "Recorded exit | id={id} exit={exit} pnl={pnl} outcome={outcome}",
                    id=signal_id,
                    exit=exit_price,
                    pnl=pnl,
                    outcome=outcome_str
                )
                return

        logger.warning("Signal not found for exit | id={id}", id=signal_id)

    def get_strategy_stats(
        self,
        strategy: str,
        window_days: int = 90
    ) -> StrategyStats:
        """Get aggregated statistics for a strategy.

        Args:
            strategy: Strategy name
            window_days: Lookback window in days (0 = all time)

        Returns:
            StrategyStats with aggregated metrics
        """
        cache_key = f"{strategy}_{window_days}"
        if cache_key in self._stats_cache and not self._cache_dirty:
            return self._stats_cache[cache_key]

        cutoff = datetime.now(timezone.utc).timestamp() - (window_days * 86400) if window_days > 0 else 0

        strategy_outcomes = [
            o for o in self._outcomes
            if o.strategy == strategy and o.is_closed()
        ]

        if window_days > 0:
            strategy_outcomes = [
                o for o in strategy_outcomes
                if o.exit_date and datetime.fromisoformat(o.exit_date).timestamp() > cutoff
            ]

        open_outcomes = [o for o in self._outcomes if o.strategy == strategy and not o.is_closed()]

        win_count = sum(1 for o in strategy_outcomes if o.is_win())
        loss_count = len(strategy_outcomes) - win_count

        win_rate = win_count / len(strategy_outcomes) if strategy_outcomes else 0.0

        pnl_values = [o.pnl for o in strategy_outcomes if o.pnl is not None]
        avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
        total_pnl = sum(pnl_values) if pnl_values else 0.0

        stats = StrategyStats(
            strategy=strategy,
            trade_count=len(strategy_outcomes),
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            open_trades=len(open_outcomes)
        )

        self._stats_cache[cache_key] = stats
        return stats

    def get_all_outcomes(self) -> List[SignalOutcome]:
        """Return all signal outcomes."""
        return list(self._outcomes)

    def get_closed_outcomes(self) -> List[SignalOutcome]:
        """Return only closed outcomes."""
        return [o for o in self._outcomes if o.is_closed()]

    def get_open_outcomes(self) -> List[SignalOutcome]:
        """Return only open outcomes."""
        return [o for o in self._outcomes if not o.is_closed()]

    def flush(self) -> None:
        """Force save history to disk."""
        if self._cache_dirty:
            self._save_history()


__all__ = ["HistoryStore", "SignalOutcome", "StrategyStats"]
