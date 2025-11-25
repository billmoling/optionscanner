from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from option_data import OptionChainSnapshot
from position_cache import PositionCache
from strategies.base import TradeSignal


def _make_signal(symbol: str, strike: float, direction: str) -> TradeSignal:
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return TradeSignal(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        option_type="PUT",
        direction=direction,
        rationale="test signal",
    )


def test_position_cache_records_and_triggers_exit(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = PositionCache(path)
    signal = _make_signal("NVDA", 95.0, "BULL_PUT_CREDIT_SPREAD")
    snapshot = OptionChainSnapshot(
        symbol="NVDA",
        underlying_price=90.0,
        timestamp=datetime.now(timezone.utc),
        options=[],
    )

    cache.record_signal("PutCredit", signal, snapshot)
    cache.save()

    reloaded = PositionCache(path)
    recommendations = reloaded.evaluate_exits({"NVDA": snapshot}, now=datetime.now(timezone.utc))

    assert recommendations, "Expected an exit recommendation when price breaches the short strike"
    assert "short strike" in recommendations[0].reason


def test_position_cache_updates_existing_entry(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = PositionCache(path)
    signal = _make_signal("AAPL", 150.0, "BULL_PUT_CREDIT_SPREAD")
    snapshot = OptionChainSnapshot(
        symbol="AAPL",
        underlying_price=170.0,
        timestamp=datetime.now(timezone.utc),
        options=[],
    )

    cache.record_signal("PutCredit", signal, snapshot)
    cache.record_signal("PutCredit", signal, snapshot)
    cache.save()
    payload = json.loads(path.read_text())
    assert len(payload) == 1


def test_position_cache_supports_custom_evaluator(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = PositionCache(path)
    signal = TradeSignal(
        symbol="TSLA",
        expiry=datetime.now(timezone.utc) + timedelta(days=2),
        strike=250.0,
        option_type="CALL",
        direction="CUSTOM_STRATEGY",
        rationale="custom",
    )
    snapshot = OptionChainSnapshot(
        symbol="TSLA",
        underlying_price=260.0,
        timestamp=datetime.now(timezone.utc),
        options=[],
    )

    cache.register_evaluator("CUSTOM_STRATEGY", lambda entry, snap, now: "custom exit")
    cache.record_signal("Custom", signal, snapshot)

    recs = cache.evaluate_exits({"TSLA": snapshot})
    assert recs and recs[0].reason == "custom exit"


def test_position_cache_handles_short_call_exit(tmp_path) -> None:
    path = tmp_path / "cache.json"
    cache = PositionCache(path)
    signal = TradeSignal(
        symbol="MSFT",
        expiry=datetime.now(timezone.utc) + timedelta(days=10),
        strike=300.0,
        option_type="CALL",
        direction="SHORT_CALL",
        rationale="covered call",
    )
    snapshot = OptionChainSnapshot(
        symbol="MSFT",
        underlying_price=310.0,
        timestamp=datetime.now(timezone.utc),
        options=[],
    )

    cache.record_signal("CoveredCall", signal, snapshot)
    recs = cache.evaluate_exits({"MSFT": snapshot})

    assert recs, "Short call evaluator should trigger when price is above strike"
