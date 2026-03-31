"""Persistent trade history tracker for live performance scoring."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(slots=True)
class TradeResult:
    """Represents a single executed trade with its outcome."""

    strategy: str
    symbol: str
    direction: str
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED_WIN, CLOSED_LOSS, CLOSED_EVEN
    quantity: int = 1
    notes: str = ""

    def is_closed(self) -> bool:
        return self.status.startswith("CLOSED")

    def is_win(self) -> bool:
        return self.status == "CLOSED_WIN" or (self.pnl is not None and self.pnl > 0)


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


class TradeHistory:
    """Persists and analyzes trade execution history.

    Trade results are submitted by dropping JSON files into the configured directory.
    Each JSON file can contain a single trade result or an array of results.

    Example trade result JSON:
    {
        "strategy": "VerticalSpreadStrategy",
        "symbol": "NVDA",
        "direction": "BULL_CALL_DEBIT_SPREAD",
        "entry_date": "2026-03-21T10:00:00Z",
        "entry_price": 2.50,
        "exit_date": "2026-03-25T15:30:00Z",
        "exit_price": 4.80,
        "pnl": 2.30,
        "status": "CLOSED_WIN",
        "quantity": 1,
        "notes": "Reached max profit target"
    }
    """

    def __init__(self, history_path: Path) -> None:
        self._history_path = history_path
        self._input_dir = history_path.parent / "trade_results"
        self._trades: List[TradeResult] = []
        self._processed_files: set[str] = set()
        self._stats_cache: Dict[str, StrategyStats] = {}
        self._cache_dirty = False

        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._input_dir.mkdir(parents=True, exist_ok=True)

        self._load_history()
        self._scan_input_dir()

    def _load_history(self) -> None:
        """Load persistent trade history from JSON file."""
        if not self._history_path.exists():
            return
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
            self._trades = [self._trade_from_dict(item) for item in data]
            self._cache_dirty = False
        except Exception as exc:
            from loguru import logger
            logger.warning("Unable to load trade history | path={path} reason={error}",
                          path=str(self._history_path), error=exc)

    def _save_history(self) -> None:
        """Persist trade history to JSON file."""
        try:
            data = [self._trade_to_dict(trade) for trade in self._trades]
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            self._history_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            self._cache_dirty = False
        except Exception as exc:
            from loguru import logger
            logger.warning("Failed to save trade history | path={path} reason={error}",
                          path=str(self._history_path), error=exc)

    def _scan_input_dir(self) -> None:
        """Scan input directory for new trade result files."""
        if not self._input_dir.exists():
            return

        for file_path in self._input_dir.glob("*.json"):
            if file_path.name in self._processed_files:
                continue
            self._process_trade_file(file_path)
            self._processed_files.add(file_path.name)

        if self._cache_dirty:
            self._save_history()

    def _process_trade_file(self, file_path: Path) -> None:
        """Process a single trade result JSON file."""
        from loguru import logger
        try:
            content = json.loads(file_path.read_text(encoding="utf-8"))
            trades = content if isinstance(content, list) else [content]

            for trade_data in trades:
                trade = self._trade_from_dict(trade_data)
                self._trades.append(trade)
                logger.info(
                    "Loaded trade result | strategy={strategy} symbol={symbol} status={status} pnl={pnl}",
                    strategy=trade.strategy,
                    symbol=trade.symbol,
                    status=trade.status,
                    pnl=trade.pnl
                )

            self._cache_dirty = True
        except Exception as exc:
            logger.warning(
                "Failed to process trade file | path={path} reason={error}",
                path=str(file_path),
                error=exc
            )

    def append_trade(self, trade: TradeResult) -> None:
        """Add a new trade result programmatically."""
        from loguru import logger
        self._trades.append(trade)
        self._cache_dirty = True
        logger.info(
            "Added trade to history | strategy={strategy} symbol={symbol} status={status}",
            strategy=trade.strategy,
            symbol=trade.symbol,
            status=trade.status
        )

    def get_strategy_stats(self, strategy: str) -> StrategyStats:
        """Get aggregated statistics for a specific strategy."""
        if strategy in self._stats_cache and not self._cache_dirty:
            return self._stats_cache[strategy]

        strategy_trades = [t for t in self._trades if t.strategy == strategy]
        closed_trades = [t for t in strategy_trades if t.is_closed()]
        open_trades = [t for t in strategy_trades if not t.is_closed()]

        win_count = sum(1 for t in closed_trades if t.is_win())
        loss_count = len(closed_trades) - win_count

        win_rate = win_count / len(closed_trades) if closed_trades else 0.0

        pnl_values = [t.pnl for t in closed_trades if t.pnl is not None]
        avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
        total_pnl = sum(pnl_values) if pnl_values else 0.0

        stats = StrategyStats(
            strategy=strategy,
            trade_count=len(strategy_trades),
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            open_trades=len(open_trades)
        )

        self._stats_cache[strategy] = stats
        return stats

    def get_all_trades(self) -> List[TradeResult]:
        """Return all trades in history."""
        return list(self._trades)

    def get_closed_trades(self) -> List[TradeResult]:
        """Return only closed trades."""
        return [t for t in self._trades if t.is_closed()]

    def get_open_trades(self) -> List[TradeResult]:
        """Return only open trades."""
        return [t for t in self._trades if not t.is_closed()]

    def get_trades_for_strategy(self, strategy: str) -> List[TradeResult]:
        """Return all trades for a specific strategy."""
        return [t for t in self._trades if t.strategy == strategy]

    def flush(self) -> None:
        """Force save history to disk."""
        if self._cache_dirty:
            self._save_history()

    @staticmethod
    def _trade_from_dict(data: Dict[str, object]) -> TradeResult:
        return TradeResult(
            strategy=str(data.get("strategy", "")),
            symbol=str(data.get("symbol", "")),
            direction=str(data.get("direction", "")),
            entry_date=str(data.get("entry_date", datetime.now(timezone.utc).isoformat())),
            entry_price=float(data.get("entry_price", 0.0)),
            exit_date=data.get("exit_date"),
            exit_price=float(data["exit_price"]) if data.get("exit_price") else None,
            pnl=float(data["pnl"]) if data.get("pnl") else None,
            status=str(data.get("status", "OPEN")),
            quantity=int(data.get("quantity", 1)),
            notes=str(data.get("notes", ""))
        )

    @staticmethod
    def _trade_to_dict(trade: TradeResult) -> Dict[str, object]:
        return {
            "strategy": trade.strategy,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_date": trade.entry_date,
            "entry_price": trade.entry_price,
            "exit_date": trade.exit_date,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "status": trade.status,
            "quantity": trade.quantity,
            "notes": trade.notes
        }


__all__ = ["TradeHistory", "TradeResult", "StrategyStats"]
