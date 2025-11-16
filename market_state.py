"""Market state classification utilities for underlying stocks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Protocol

import pandas as pd

from technical_indicators import TechnicalIndicatorProcessor
from stock_data import StockDataFetcher


class MarketState(str, Enum):
    """Enumeration of supported market states."""

    BULL = "bull"
    BEAR = "bear"


@dataclass(frozen=True)
class MarketStateResult:
    """Container representing a classification decision."""

    symbol: str
    state: MarketState
    as_of: pd.Timestamp
    close: float


class MarketStateProvider(Protocol):
    """Protocol describing how strategies request state information."""

    def get_state(self, symbol: str) -> Optional[MarketState]:  # pragma: no cover - protocol
        ...


class MarketStateClassifier:
    """Classifies a security's regime based on moving averages."""

    def classify(self, history: pd.DataFrame, symbol: Optional[str] = None) -> Optional[MarketStateResult]:
        if history is None or history.empty:
            return None
        required_cols = {"close", "ma5", "ma10", "ma30"}
        if not required_cols.issubset(history.columns):
            return None
        filtered = history.dropna(subset=list(required_cols))
        if filtered.empty:
            return None
        latest = filtered.iloc[-1]
        close = float(latest["close"])
        ma5 = float(latest["ma5"])
        ma10 = float(latest["ma10"])
        ma30 = float(latest["ma30"])
        if close > ma5 > ma10 > ma30:
            state = MarketState.BULL
        else:
            state = MarketState.BEAR
        timestamp_value = latest.get("timestamp")
        if pd.isna(timestamp_value):
            timestamp_value = filtered.index[-1]
        timestamp = pd.to_datetime(timestamp_value)
        if getattr(timestamp, "tzinfo", None) is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        resolved_symbol = symbol or str(latest.get("symbol", ""))
        return MarketStateResult(symbol=resolved_symbol, state=state, as_of=timestamp, close=close)


class DictMarketStateProvider(MarketStateProvider):
    """Simple provider backed by an in-memory dictionary."""

    def __init__(self, mapping: Optional[Dict[str, MarketState]] = None) -> None:
        self._mapping: Dict[str, MarketState] = {k.upper(): v for k, v in (mapping or {}).items()}

    def set_state(self, symbol: str, state: MarketState) -> None:
        self._mapping[symbol.upper()] = state

    def get_state(self, symbol: str) -> Optional[MarketState]:
        return self._mapping.get(symbol.upper())


class StockMarketStateProvider(MarketStateProvider):
    """Provider that downloads data via IBKR and classifies the market state."""

    def __init__(
        self,
        fetcher: StockDataFetcher,
        indicator_processor: Optional[TechnicalIndicatorProcessor] = None,
        classifier: Optional[MarketStateClassifier] = None,
    ) -> None:
        self._fetcher = fetcher
        self._processor = indicator_processor or TechnicalIndicatorProcessor()
        self._classifier = classifier or MarketStateClassifier()
        self._cache: Dict[str, MarketStateResult] = {}

    async def refresh(self, symbols: Iterable[str], **history_kwargs: Any) -> Dict[str, Optional[MarketStateResult]]:
        """Download the latest data for ``symbols`` and update the cache."""

        results: Dict[str, Optional[MarketStateResult]] = {}
        for symbol in symbols:
            history = await self._fetcher.fetch_history(symbol, **history_kwargs)
            enriched = self._processor.process(history)
            result = self._classifier.classify(enriched, symbol=symbol)
            if result:
                self._cache[symbol.upper()] = result
            results[symbol] = result
        return results

    def get_state(self, symbol: str) -> Optional[MarketState]:
        cached = self._cache.get(symbol.upper())
        if cached is None:
            return None
        return cached.state


__all__ = [
    "MarketState",
    "MarketStateClassifier",
    "MarketStateResult",
    "MarketStateProvider",
    "DictMarketStateProvider",
    "StockMarketStateProvider",
]
