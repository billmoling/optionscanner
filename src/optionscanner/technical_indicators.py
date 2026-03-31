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
        
        if "rsi14" not in self.indicators:
            self.register("rsi14", self.relative_strength_index(14))

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

    @staticmethod
    def relative_strength_index(period: int) -> IndicatorFunc:
        """Return a callable that computes the Relative Strength Index (RSI)."""

        if period <= 0:
            raise ValueError("RSI period must be positive")

        def _rsi(df: pd.DataFrame) -> pd.Series:
            delta = df["close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            # Use exponential moving average (Wilder's method)
            ma_up = up.ewm(com=period - 1, adjust=False, min_periods=period).mean()
            ma_down = down.ewm(com=period - 1, adjust=False, min_periods=period).mean()
            rsi = ma_up / ma_down
            rsi = 100 - (100 / (1 + rsi))
            return rsi

        return _rsi


__all__ = ["TechnicalIndicatorProcessor", "IndicatorFunc"]
