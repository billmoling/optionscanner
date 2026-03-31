"""Technical pattern recognition for entry timing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass(slots=True)
class PatternSignal:
    """Detected technical pattern."""

    pattern_type: str  # "BREAKOUT", "REVERSAL", "CONSOLIDATION"
    symbol: str
    timestamp: datetime
    confidence: float
    price_level: float


class PatternRecognizer:
    """Detects technical patterns in price data.

    Patterns:
    - BREAKOUT: Price above N-day high with volume confirmation
    - CONSOLIDATION: Low ATR relative to recent history
    - REVERSAL: RSI divergence + candlestick pattern
    """

    def detect_breakout(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20
    ) -> Optional[PatternSignal]:
        """Detect breakout above lookback high.

        Args:
            ohlcv: OHLCV DataFrame with columns: high, low, close, volume
            lookback: Number of days for high comparison

        Returns:
            PatternSignal if breakout detected, None otherwise
        """
        if len(ohlcv) < lookback + 5:
            return None

        # Calculate 20-day high
        rolling_high = ohlcv["high"].rolling(window=lookback).max()

        # Check if current price broke above high
        current_price = ohlcv["close"].iloc[-1]
        prev_high = rolling_high.iloc[-2] if len(rolling_high) > 1 else ohlcv["high"].iloc[-lookback:-1].max()

        if current_price <= prev_high:
            return None

        # Volume confirmation: current volume > 1.5x average
        avg_volume = ohlcv["volume"].iloc[-10:-1].mean()
        current_volume = ohlcv["volume"].iloc[-1]

        if current_volume < avg_volume * 1.5:
            return None

        # Calculate confidence
        breakout_pct = (current_price - prev_high) / prev_high
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        confidence = min(0.5 + breakout_pct * 10 + (volume_ratio - 1.5) * 0.2, 1.0)

        logger.debug(
            "Breakout detected | price={price} prev_high={high} breakout_pct={pct:.2%} volume_ratio={vol:.2f}",
            price=current_price,
            high=prev_high,
            pct=breakout_pct,
            vol=volume_ratio
        )

        return PatternSignal(
            pattern_type="BREAKOUT",
            symbol="",
            timestamp=datetime.now(),
            confidence=confidence,
            price_level=current_price
        )

    def detect_consolidation(
        self,
        ohlcv: pd.DataFrame,
        min_days: int = 5
    ) -> Optional[PatternSignal]:
        """Detect consolidation (low volatility) pattern.

        Args:
            ohlcv: OHLCV DataFrame
            min_days: Minimum days of consolidation

        Returns:
            PatternSignal if consolidation detected
        """
        if len(ohlcv) < min_days + 10:
            return None

        # Calculate ATR (simplified: high-low range)
        ohlcv = ohlcv.copy()
        ohlcv["range"] = ohlcv["high"] - ohlcv["low"]

        # Recent ATR vs 20-day ATR
        recent_atr = ohlcv["range"].iloc[-min_days:].mean()
        base_atr = ohlcv["range"].iloc[-30:-10].mean() if len(ohlcv) >= 30 else ohlcv["range"].iloc[:-min_days].mean()

        if base_atr <= 0:
            return None

        atr_ratio = recent_atr / base_atr

        # Consolidation: ATR < 50% of baseline
        if atr_ratio >= 0.5:
            return None

        confidence = 0.5 + (0.5 - atr_ratio)

        current_price = ohlcv["close"].iloc[-1]

        logger.debug(
            "Consolidation detected | atr_ratio={ratio:.2f} price={price}",
            ratio=atr_ratio,
            price=current_price
        )

        return PatternSignal(
            pattern_type="CONSOLIDATION",
            symbol="",
            timestamp=datetime.now(),
            confidence=min(confidence, 1.0),
            price_level=current_price
        )

    def detect_reversal(
        self,
        ohlcv: pd.DataFrame
    ) -> Optional[PatternSignal]:
        """Detect reversal pattern (RSI divergence + candlestick).

        Simplified implementation - checks for hammer/shooting star.

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            PatternSignal if reversal detected
        """
        if len(ohlcv) < 10:
            return None

        current = ohlcv.iloc[-1]
        prev = ohlcv.iloc[-2]

        # Hammer: small body, long lower shadow
        body = abs(current["close"] - current["open"])
        total_range = current["high"] - current["low"]
        lower_shadow = min(current["open"], current["close"]) - current["low"]
        upper_shadow = current["high"] - max(current["open"], current["close"])

        is_hammer = (
            total_range > 0 and
            body < total_range * 0.3 and
            lower_shadow > body * 2 and
            upper_shadow < body * 0.5
        )

        if is_hammer:
            # Bullish reversal after downtrend
            downtrend = current["close"] < prev["close"] < ohlcv.iloc[-3]["close"]
            if downtrend:
                return PatternSignal(
                    pattern_type="REVERSAL_BULLISH",
                    symbol="",
                    timestamp=datetime.now(),
                    confidence=0.6,
                    price_level=current["close"]
                )

        return None


__all__ = ["PatternRecognizer", "PatternSignal"]
