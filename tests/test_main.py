from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from optionscanner.main import discover_strategies, resolve_market_data_type
from optionscanner.strategies.strategy_put_credit_spread import PutCreditSpreadStrategy


def test_discover_strategies_respects_enabled_flag() -> None:
    strategies = discover_strategies({"VerticalSpreadStrategy": {"enabled": False}})
    names = {strategy.__class__.__name__ for strategy in strategies}
    assert "VerticalSpreadStrategy" not in names


def test_discover_strategies_applies_params_override() -> None:
    overrides = {"PutCreditSpreadStrategy": {"params": {"min_credit": 1.25}}}
    strategies = discover_strategies(overrides)
    target = next(strategy for strategy in strategies if isinstance(strategy, PutCreditSpreadStrategy))
    assert target.min_credit == pytest.approx(1.25)


def test_resolve_market_data_type_live() -> None:
    """Test that LIVE is returned unchanged."""
    assert resolve_market_data_type("LIVE") == "LIVE"


def test_resolve_market_data_type_frozen() -> None:
    """Test that FROZEN is returned unchanged."""
    assert resolve_market_data_type("FROZEN") == "FROZEN"


def test_resolve_market_data_type_auto_lowercase() -> None:
    """Test that auto (lowercase) is resolved correctly."""
    test_time = datetime(2026, 4, 6, 10, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))  # Monday during hours
    with patch("optionscanner.main.MarketHoursChecker") as mock_checker_class:
        mock_checker = mock_checker_class.return_value
        mock_checker.get_market_data_type.return_value = "LIVE"
        result = resolve_market_data_type("auto")
        assert result == "LIVE"


def test_resolve_market_data_type_auto_uppercase() -> None:
    """Test that AUTO (uppercase) is resolved correctly."""
    test_time = datetime(2026, 4, 6, 15, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))  # Monday after hours
    with patch("optionscanner.main.MarketHoursChecker") as mock_checker_class:
        mock_checker = mock_checker_class.return_value
        mock_checker.get_market_data_type.return_value = "FROZEN"
        result = resolve_market_data_type("AUTO")
        assert result == "FROZEN"

