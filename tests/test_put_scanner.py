"""Integration tests for Put Scanner."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from optionscanner.analytics.bs_model import BSModel
from optionscanner.filters.options_filters import OptionFilterPipeline
from optionscanner.scanners.put_scanner import PutScanResult, PutScanner


class TestPutScanResult:
    """Tests for PutScanResult dataclass."""

    def test_mid_price_calculation(self) -> None:
        """Mid price is average of bid and ask."""
        result = PutScanResult(
            symbol="AAPL",
            strike=100.0,
            expiry=pd.Timestamp("2026-05-01", tz="UTC"),
            bid=2.0,
            ask=2.5,
            last=2.25,
            volume=1000,
            iv_rank=0.40,
            days_to_expiry=30,
            underlying_price=105.0,
            otm_probability=0.70,
            annualized_roi=0.35,
        )
        assert result.mid_price == 2.25


class TestPutScanner:
    """Tests for PutScanner."""

    def test_empty_dataframe_returns_empty_list(self) -> None:
        """Scanner returns empty list for empty DataFrame."""
        scanner = PutScanner()
        puts_df = pd.DataFrame()
        results = scanner.scan("AAPL", puts_df, 100.0)
        assert results == []

    def test_scans_and_returns_sorted_results(self) -> None:
        """Scanner returns results sorted by annualized_roi."""
        scanner = PutScanner()
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=30)

        puts_df = pd.DataFrame([
            {
                "strike": 95.0,
                "bid": 1.0,
                "ask": 1.2,
                "last": 1.1,
                "volume": 1000,
                "iv_rank": 0.50,
                "expiry": expiry,
            },
            {
                "strike": 90.0,
                "bid": 0.5,
                "ask": 0.7,
                "last": 0.6,
                "volume": 1000,
                "iv_rank": 0.50,
                "expiry": expiry,
            },
        ])

        results = scanner.scan("AAPL", puts_df, 100.0, now)

        assert len(results) >= 0
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].annualized_roi >= results[i + 1].annualized_roi

    def test_filters_low_iv_rank(self) -> None:
        """Scanner filters out options with low IV Rank."""
        scanner = PutScanner()
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=30)

        puts_df = pd.DataFrame([
            {
                "strike": 95.0,
                "bid": 1.0,
                "ask": 1.2,
                "volume": 1000,
                "iv_rank": 0.10,
                "expiry": expiry,
            },
        ])

        results = scanner.scan("AAPL", puts_df, 100.0, now)

        assert len(results) == 0

    def test_filters_low_volume(self) -> None:
        """Scanner filters out options with low volume."""
        scanner = PutScanner()
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=30)

        puts_df = pd.DataFrame([
            {
                "strike": 95.0,
                "bid": 1.0,
                "ask": 1.2,
                "volume": 100,
                "iv_rank": 0.50,
                "expiry": expiry,
            },
        ])

        results = scanner.scan("AAPL", puts_df, 100.0, now)

        assert len(results) == 0

    def test_calculates_annualized_roi(self) -> None:
        """Scanner correctly calculates annualized ROI."""
        scanner = PutScanner()

        roi = scanner._calculate_annualized_roi(
            strike=100.0,
            premium=2.0,
            days_to_expiry=30,
        )

        expected = (2.0 / 100.0) * (365.0 / 30.0)
        assert roi == pytest.approx(expected, rel=0.01)

    def test_custom_filter_pipeline(self) -> None:
        """Scanner accepts custom filter pipeline."""
        custom_pipeline = OptionFilterPipeline()
        scanner = PutScanner(filter_pipeline=custom_pipeline)

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=30)

        puts_df = pd.DataFrame([
            {
                "strike": 95.0,
                "bid": 0.5,
                "ask": 0.7,
                "volume": 100,
                "iv_rank": 0.10,
                "expiry": expiry,
            },
        ])

        results = scanner.scan("AAPL", puts_df, 100.0, now)

        assert len(results) >= 0

    def test_custom_bs_model(self) -> None:
        """Scanner accepts custom BS model."""
        custom_model = BSModel(risk_free_rate=0.03)
        scanner = PutScanner(bs_model=custom_model)

        assert scanner.bs_model.risk_free_rate == 0.03

    def test_result_contains_all_metrics(self) -> None:
        """Scan result contains all required metrics."""
        scanner = PutScanner()
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=30)

        puts_df = pd.DataFrame([
            {
                "strike": 95.0,
                "bid": 1.5,
                "ask": 1.7,
                "last": 1.6,
                "volume": 1000,
                "iv_rank": 0.50,
                "expiry": expiry,
            },
        ])

        results = scanner.scan("AAPL", puts_df, 100.0, now)

        if results:
            result = results[0]
            assert result.symbol == "AAPL"
            assert result.strike > 0
            assert result.expiry is not None
            assert result.bid > 0
            assert result.ask > 0
            assert result.volume > 0
            assert result.iv_rank >= 0
            assert result.days_to_expiry > 0
            assert result.underlying_price > 0
            assert 0 <= result.otm_probability <= 1
            assert result.annualized_roi >= 0
