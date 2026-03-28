# tests/test_patterns.py
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from entry.patterns import PatternRecognizer, PatternSignal


class TestPatternRecognizer:
    def test_detect_breakout_above_high(self):
        recognizer = PatternRecognizer()

        # Create OHLCV data with 20-day breakout
        # First 25 days: flat at 100 (high = 101), then breakout on last day
        dates = [datetime.now(timezone.utc) - timedelta(days=29-i) for i in range(30)]
        prices = [100.0] * 25 + [100.0, 100.0, 100.0, 100.0, 115.0]
        # Volume: low for first 25 days, then spike on breakout day (last day)
        volumes = [1000] * 25 + [1000, 1000, 1000, 1000, 3000]

        df = pd.DataFrame({
            "close": prices,
            "volume": volumes,
        }, index=dates)
        # High: 101 for first 25 days, then breakout day high = 115 * 1.01 = 116.15
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
        df["open"] = df["close"]

        result = recognizer.detect_breakout(df, lookback=20)

        assert result is not None
        assert result.pattern_type == "BREAKOUT"
        assert result.confidence > 0.5

    def test_detect_consolidation_low_atr(self):
        recognizer = PatternRecognizer()

        # Create OHLCV data with low volatility (consolidation)
        # Need at least min_days (5) + 10 = 15 days of data
        # First 10 days: high volatility (range = 2.0), last 5 days: low volatility (range = 0.5)
        dates = [datetime.now(timezone.utc) - timedelta(days=14-i) for i in range(15)]
        prices = [100.0] * 10 + [100.0, 100.1, 99.9, 100.0, 100.05]
        # High volatility for first 10 days, low for last 5

        df = pd.DataFrame({"close": prices}, index=dates)
        # First 10 days: wide range (high - low = 2.0), last 5 days: narrow range (high - low = 0.5)
        df["high"] = [prices[i] + (2.0 if i < 10 else 0.5) for i in range(len(prices))]
        df["low"] = [prices[i] - (2.0 if i < 10 else 0.5) for i in range(len(prices))]
        df["open"] = df["close"]
        df["volume"] = 1000

        result = recognizer.detect_consolidation(df)

        assert result is not None
        assert result.pattern_type == "CONSOLIDATION"
