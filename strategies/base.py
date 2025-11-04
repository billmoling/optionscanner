"""Base classes and utilities for NautilusTrader option strategies."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Optional

from loguru import logger

try:
    from nautilus_trader.trading.strategy import Strategy
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "NautilusTrader must be installed to use the strategy framework."
    ) from exc


@dataclass(slots=True)
class TradeSignal:
    """Represents a trade signal emitted by a strategy."""

    symbol: str
    expiry: datetime
    strike: float
    option_type: str
    direction: str
    rationale: str


class BaseOptionStrategy(Strategy, abc.ABC):
    """Abstract base class for option strategies using NautilusTrader."""

    def __init__(self, name: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if name is not None:
            self.name = name
        elif not hasattr(self, "name") or getattr(self, "name", None) in (None, ""):
            self.name = self.__class__.__name__
        logger.debug("Initialized strategy: {name}", name=self.name)

    @abc.abstractmethod
    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """Process incoming data and return trade signals."""

    def emit_signal(self, signal: TradeSignal) -> TradeSignal:
        """Utility for logging and returning signals."""

        logger.info(
            "Signal emitted | strategy={strategy} symbol={symbol} strike={strike} expiry={expiry} type={opt_type} direction={direction} rationale={rationale}",
            strategy=self.name,
            symbol=signal.symbol,
            strike=signal.strike,
            expiry=signal.expiry.isoformat(),
            opt_type=signal.option_type,
            direction=signal.direction,
            rationale=signal.rationale,
        )
        return signal


__all__ = ["BaseOptionStrategy", "TradeSignal"]
