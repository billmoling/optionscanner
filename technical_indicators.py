"""Technical indicator processing utilities for underlying stock data."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict

import pandas as pd

IndicatorFunc = Callable[[pd.DataFrame], pd.Series]


@dataclass
class TechnicalIndicatorProcessor:
    """Applies a configurable set of technical indicators to price history.

    The processor keeps a registry of indicator callables so additional
    indicators can be registered without altering the processing pipeline.
    """

    indicators: Dict[str, IndicatorFunc] = field(default_factory=OrderedDict)

    def register(self, name: str, func: IndicatorFunc) -> None:
        """Register a new indicator calculation by name."""

        if not callable(func):  # pragma: no cover - defensive programming
            raise TypeError("Indicator function must be callable")
        self.indicators[name] = func

    def ensure_default_moving_averages(self) -> None:
        """Ensure MA5/MA10/MA30 are part of the registry."""

        for period in (5, 10, 30):
            name = f"ma{period}"
            if name not in self.indicators:
                self.register(name, self.simple_moving_average(period))

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``data`` with all registered indicators applied."""

        if data is None:
            raise ValueError("Price history is required for indicator processing")
        frame = data.copy()
        if "close" not in frame.columns:
            raise KeyError("Expected 'close' column in price history")
        if not self.indicators:
            self.ensure_default_moving_averages()
        for name, func in self.indicators.items():
            frame[name] = func(frame)
        return frame

    @staticmethod
    def simple_moving_average(period: int) -> IndicatorFunc:
        """Return a callable that computes a simple moving average."""

        if period <= 0:
            raise ValueError("MA period must be positive")

        def _sma(df: pd.DataFrame) -> pd.Series:
            return df["close"].rolling(window=period, min_periods=period).mean()

        return _sma


__all__ = ["TechnicalIndicatorProcessor", "IndicatorFunc"]
