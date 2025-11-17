from __future__ import annotations

import pytest

from main import discover_strategies
from strategies.strategy_put_credit_spread import PutCreditSpreadStrategy


def test_discover_strategies_respects_enabled_flag() -> None:
    strategies = discover_strategies({"VerticalSpreadStrategy": {"enabled": False}})
    names = {strategy.__class__.__name__ for strategy in strategies}
    assert "VerticalSpreadStrategy" not in names


def test_discover_strategies_applies_params_override() -> None:
    overrides = {"PutCreditSpreadStrategy": {"params": {"min_credit": 1.25}}}
    strategies = discover_strategies(overrides)
    target = next(strategy for strategy in strategies if isinstance(strategy, PutCreditSpreadStrategy))
    assert target.min_credit == pytest.approx(1.25)
