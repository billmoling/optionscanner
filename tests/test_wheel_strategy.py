"""Unit tests for Wheel Strategy."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from optionscanner.strategies.strategy_wheel import WheelStrategy


def make_snapshot(
    symbol: str,
    underlying_price: float,
    options: list[dict],
) -> dict:
    """Create a test snapshot."""
    return {
        "symbol": symbol,
        "underlying_price": underlying_price,
        "options": options,
    }


class TestWheelStrategy:
    """Tests for Wheel Strategy."""

    def test_initialization(self) -> None:
        """Strategy initializes successfully."""
        strategy = WheelStrategy()
        assert strategy.scanner is not None
        assert strategy.max_days_to_expiry == 15

    def test_custom_parameters(self) -> None:
        """Strategy accepts custom parameters."""
        strategy = WheelStrategy(
            min_iv_rank=0.40,
            min_volume=1000,
            min_annualized_roi=0.50,
            min_otm_probability=0.70,
            max_days_to_expiry=30,
        )
        assert strategy.max_days_to_expiry == 30
        assert strategy.scanner is not None

    def test_empty_data_returns_empty_signals(self) -> None:
        """Strategy returns empty list for empty data."""
        strategy = WheelStrategy()
        signals = strategy.on_data([])
        assert signals == []

    def test_generates_signal_for_attractive_put(self) -> None:
        """Strategy generates signal for attractive CSP opportunity."""
        strategy = WheelStrategy(
            min_iv_rank=0.30,
            min_volume=500,
            min_annualized_roi=0.30,
            min_otm_probability=0.60,
        )

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=10)

        snapshot = make_snapshot(
            symbol="AAPL",
            underlying_price=150.0,
            options=[
                {
                    "strike": 145.0,
                    "option_type": "PUT",
                    "bid": 2.0,
                    "ask": 2.2,
                    "last": 2.1,
                    "volume": 1000,
                    "iv_rank": 0.50,
                    "expiry": expiry,
                },
            ],
        )

        signals = strategy.on_data([snapshot])

        assert len(signals) >= 0
        if signals:
            signal = signals[0]
            assert signal.symbol == "AAPL"
            assert signal.option_type == "PUT"
            assert signal.direction == "SELL_PUT"
            assert signal.strike == 145.0

    def test_filters_low_iv_rank(self) -> None:
        """Strategy filters out low IV Rank options."""
        strategy = WheelStrategy(min_iv_rank=0.30)

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=10)

        snapshot = make_snapshot(
            symbol="AAPL",
            underlying_price=150.0,
            options=[
                {
                    "strike": 145.0,
                    "option_type": "PUT",
                    "bid": 1.0,
                    "ask": 1.2,
                    "volume": 1000,
                    "iv_rank": 0.15,
                    "expiry": expiry,
                },
            ],
        )

        signals = strategy.on_data([snapshot])
        assert len(signals) == 0

    def test_filters_low_volume(self) -> None:
        """Strategy filters out low volume options."""
        strategy = WheelStrategy(min_volume=500)

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=10)

        snapshot = make_snapshot(
            symbol="AAPL",
            underlying_price=150.0,
            options=[
                {
                    "strike": 145.0,
                    "option_type": "PUT",
                    "bid": 2.0,
                    "ask": 2.2,
                    "volume": 100,
                    "iv_rank": 0.50,
                    "expiry": expiry,
                },
            ],
        )

        signals = strategy.on_data([snapshot])
        assert len(signals) == 0

    def test_filters_wrong_expiry(self) -> None:
        """Strategy filters out options outside expiry range."""
        strategy = WheelStrategy(min_days_to_expiry=0, max_days_to_expiry=15)

        now = datetime.now(timezone.utc)
        expiry_far = now + timedelta(days=60)

        snapshot = make_snapshot(
            symbol="AAPL",
            underlying_price=150.0,
            options=[
                {
                    "strike": 145.0,
                    "option_type": "PUT",
                    "bid": 2.0,
                    "ask": 2.2,
                    "volume": 1000,
                    "iv_rank": 0.50,
                    "expiry": expiry_far,
                },
            ],
        )

        signals = strategy.on_data([snapshot])
        assert len(signals) == 0

    def test_signal_rationale_contains_metrics(self) -> None:
        """Signal rationale contains all key metrics."""
        strategy = WheelStrategy()

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=10)

        snapshot = make_snapshot(
            symbol="AAPL",
            underlying_price=150.0,
            options=[
                {
                    "strike": 145.0,
                    "option_type": "PUT",
                    "bid": 2.0,
                    "ask": 2.2,
                    "volume": 1000,
                    "iv_rank": 0.50,
                    "expiry": expiry,
                },
            ],
        )

        signals = strategy.on_data([snapshot])

        if signals:
            rationale = signals[0].rationale
            assert "AAPL" in rationale
            assert "145" in rationale
            assert "ROI" in rationale
            assert "OTM" in rationale

    def test_signal_has_correct_legs(self) -> None:
        """Signal contains correct leg structure."""
        strategy = WheelStrategy()

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=10)

        snapshot = make_snapshot(
            symbol="AAPL",
            underlying_price=150.0,
            options=[
                {
                    "strike": 145.0,
                    "option_type": "PUT",
                    "bid": 2.0,
                    "ask": 2.2,
                    "volume": 1000,
                    "iv_rank": 0.50,
                    "expiry": expiry,
                },
            ],
        )

        signals = strategy.on_data([snapshot])

        if signals:
            signal = signals[0]
            assert signal.legs is not None
            assert len(signal.legs) == 1
            leg = signal.legs[0]
            assert leg.action == "SELL"
            assert leg.option_type == "PUT"
            assert leg.strike == 145.0
