import unittest
from datetime import datetime, timedelta, timezone
from typing import Optional

from market_state import MarketState, MarketStateProvider
from strategies.strategy_covered_call import CoveredCallStrategy
from strategies.strategy_iron_condor import IronCondorStrategy
from strategies.strategy_pmcc import PoorMansCoveredCallStrategy
from strategies.strategy_put_credit_spread import PutCreditSpreadStrategy
from strategies.strategy_vertical_spread import VerticalSpreadStrategy


def make_snapshot(underlying: float, options: list[dict]) -> dict:
    return {
        "symbol": "NVDA",
        "underlying_price": underlying,
        "options": options,
    }


class _StaticStateProvider(MarketStateProvider):
    def __init__(self, state: Optional[MarketState]) -> None:
        self._state = state

    def get_state(self, symbol: str) -> Optional[MarketState]:
        return self._state


class CoveredCallStrategyTests(unittest.TestCase):
    def test_generates_signal_for_rich_premium(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=30)
        options = [
            {
                "expiry": expiry,
                "strike": 525.0,
                "option_type": "CALL",
                "bid": 6.0,
                "ask": 6.5,
                "mark": 6.25,
            },
        ]
        snapshot = make_snapshot(underlying=500.0, options=options)
        strategy = CoveredCallStrategy(enabled=True)

        signals = strategy.on_data([snapshot])

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "NVDA")
        self.assertEqual(signals[0].option_type, "CALL")


class VerticalSpreadStrategyTests(unittest.TestCase):
    def test_builds_bull_and_bear_spreads(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=40)
        options = [
            {"expiry": expiry, "strike": 100.0, "option_type": "CALL", "mark": 5.0, "theta": -0.02, "implied_volatility": 0.4},
            {"expiry": expiry, "strike": 105.0, "option_type": "CALL", "mark": 3.0, "theta": -0.015, "implied_volatility": 0.45},
            {"expiry": expiry, "strike": 95.0, "option_type": "PUT", "mark": 4.5, "theta": -0.01, "implied_volatility": 0.5},
            {"expiry": expiry, "strike": 90.0, "option_type": "PUT", "mark": 3.5, "theta": -0.012, "implied_volatility": 0.55},
        ]
        snapshot = make_snapshot(underlying=100.0, options=options)
        strategy = VerticalSpreadStrategy(spread_width=5.0, min_days_to_expiry=10)

        signals = strategy.on_data([snapshot])

        self.assertGreaterEqual(len(signals), 2)
        self.assertTrue(any(signal.option_type == "CALL" for signal in signals))
        self.assertTrue(any(signal.option_type == "PUT" for signal in signals))

    def test_respects_bull_state(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=40)
        options = [
            {"expiry": expiry, "strike": 100.0, "option_type": "CALL", "mark": 5.0, "theta": -0.02, "implied_volatility": 0.4},
            {"expiry": expiry, "strike": 105.0, "option_type": "CALL", "mark": 3.0, "theta": -0.015, "implied_volatility": 0.45},
            {"expiry": expiry, "strike": 95.0, "option_type": "PUT", "mark": 4.5, "theta": -0.01, "implied_volatility": 0.5},
            {"expiry": expiry, "strike": 90.0, "option_type": "PUT", "mark": 3.5, "theta": -0.012, "implied_volatility": 0.55},
        ]
        snapshot = make_snapshot(underlying=100.0, options=options)
        strategy = VerticalSpreadStrategy(
            spread_width=5.0,
            min_days_to_expiry=10,
            market_state_provider=_StaticStateProvider(MarketState.BULL),
        )

        signals = strategy.on_data([snapshot])

        self.assertTrue(signals)
        self.assertTrue(all(signal.option_type == "CALL" for signal in signals))

    def test_allows_uptrend_state_for_bullish_spreads(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=30)
        options = [
            {"expiry": expiry, "strike": 100.0, "option_type": "CALL", "mark": 5.0, "theta": -0.02, "implied_volatility": 0.4},
            {"expiry": expiry, "strike": 105.0, "option_type": "CALL", "mark": 3.0, "theta": -0.015, "implied_volatility": 0.45},
            {"expiry": expiry, "strike": 95.0, "option_type": "PUT", "mark": 4.5, "theta": -0.01, "implied_volatility": 0.5},
        ]
        snapshot = make_snapshot(underlying=100.0, options=options)
        strategy = VerticalSpreadStrategy(
            spread_width=5.0,
            min_days_to_expiry=10,
            market_state_provider=_StaticStateProvider(MarketState.UPTREND),
        )

        signals = strategy.on_data([snapshot])

        self.assertTrue(signals)
        self.assertTrue(all(signal.option_type == "CALL" for signal in signals))

    def test_respects_bear_state(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=40)
        options = [
            {"expiry": expiry, "strike": 100.0, "option_type": "CALL", "mark": 5.0, "theta": -0.02, "implied_volatility": 0.4},
            {"expiry": expiry, "strike": 105.0, "option_type": "CALL", "mark": 3.0, "theta": -0.015, "implied_volatility": 0.45},
            {"expiry": expiry, "strike": 95.0, "option_type": "PUT", "mark": 4.5, "theta": -0.01, "implied_volatility": 0.5},
            {"expiry": expiry, "strike": 90.0, "option_type": "PUT", "mark": 3.5, "theta": -0.012, "implied_volatility": 0.55},
        ]
        snapshot = make_snapshot(underlying=100.0, options=options)
        strategy = VerticalSpreadStrategy(
            spread_width=5.0,
            min_days_to_expiry=10,
            market_state_provider=_StaticStateProvider(MarketState.BEAR),
        )

        signals = strategy.on_data([snapshot])

        self.assertTrue(signals)
        self.assertTrue(all(signal.option_type == "PUT" for signal in signals))


class IronCondorStrategyTests(unittest.TestCase):
    def test_identifies_credit_opportunity(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=35)
        call_strikes = [110.0, 115.0, 120.0]
        put_strikes = [80.0, 85.0, 90.0]
        options: list[dict] = []
        for strike in call_strikes:
            options.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": "CALL",
                    "bid": 1.5,
                    "delta": 0.2,
                }
            )
        for strike in put_strikes:
            options.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": "PUT",
                    "bid": 1.6,
                    "delta": -0.25,
                }
            )
        options.append(
            {
                "expiry": expiry,
                "strike": 70.0,
                "option_type": "PUT",
                "bid": 1.0,
                "delta": -0.35,
            }
        )
        options.append(
            {
                "expiry": expiry,
                "strike": 130.0,
                "option_type": "CALL",
                "bid": 0.8,
                "delta": 0.05,
            }
        )
        snapshot = make_snapshot(underlying=100.0, options=options)
        strategy = IronCondorStrategy(target_delta=0.2, premium_threshold=0.5, min_credit_pct=0.001)

        signals = strategy.on_data([snapshot])

        self.assertEqual(len(signals), 4)
        directions = {signal.direction for signal in signals}
        self.assertSetEqual(
            directions,
            {"SHORT_CONDOR_CALL", "LONG_CONDOR_CALL", "SHORT_CONDOR_PUT", "LONG_CONDOR_PUT"},
        )


class PoorMansCoveredCallStrategyTests(unittest.TestCase):
    def test_pmcc_combines_leaps_and_short_call(self) -> None:
        now = datetime.now(timezone.utc)
        leaps_expiry = now + timedelta(days=270)
        short_expiry = now + timedelta(days=35)
        options = [
            {
                "expiry": leaps_expiry,
                "strike": 440.0,
                "option_type": "CALL",
                "ask": 30.0,
                "bid": 29.5,
                "mark": 29.75,
                "delta": 0.75,
                "theta": -0.02,
            },
            {
                "expiry": short_expiry,
                "strike": 525.0,
                "option_type": "CALL",
                "bid": 5.0,
                "ask": 5.2,
                "mark": 5.1,
                "delta": 0.3,
                "theta": -0.05,
            },
        ]
        snapshot = make_snapshot(underlying=500.0, options=options)
        strategy = PoorMansCoveredCallStrategy(min_return_on_capital=0.1)

        signals = strategy.on_data([snapshot])

        self.assertEqual(len(signals), 2)
        directions = {signal.direction for signal in signals}
        self.assertSetEqual(directions, {"LONG_PMCC_LEAPS", "SHORT_PMCC_CALL"})


class PutCreditSpreadStrategyTests(unittest.TestCase):
    def test_emits_signal_in_uptrend(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=25)
        options = [
            {"expiry": expiry, "strike": 95.0, "option_type": "PUT", "bid": 2.5, "ask": 2.6, "delta": -0.25},
            {"expiry": expiry, "strike": 90.0, "option_type": "PUT", "bid": 1.2, "ask": 1.25, "delta": -0.18},
            {"expiry": expiry, "strike": 85.0, "option_type": "PUT", "bid": 0.5, "ask": 0.55, "delta": -0.1},
        ]
        snapshot = make_snapshot(underlying=100.0, options=options)
        strategy = PutCreditSpreadStrategy(
            spread_width=5.0,
            min_credit=0.5,
            market_state_provider=_StaticStateProvider(MarketState.UPTREND),
        )

        signals = strategy.on_data([snapshot])

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].option_type, "PUT")
        self.assertEqual(signals[0].direction, "BULL_PUT_CREDIT_SPREAD")

    def test_respects_market_state_filter(self) -> None:
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(days=25)
        options = [
            {"expiry": expiry, "strike": 95.0, "option_type": "PUT", "bid": 2.5, "ask": 2.6, "delta": -0.25},
            {"expiry": expiry, "strike": 90.0, "option_type": "PUT", "bid": 1.2, "ask": 1.25, "delta": -0.18},
            {"expiry": expiry, "strike": 85.0, "option_type": "PUT", "bid": 0.5, "ask": 0.55, "delta": -0.1},
        ]
        snapshot = make_snapshot(underlying=100.0, options=options)
        strategy = PutCreditSpreadStrategy(
            spread_width=5.0,
            min_credit=0.5,
            market_state_provider=_StaticStateProvider(MarketState.BEAR),
        )

        signals = strategy.on_data([snapshot])

        self.assertFalse(signals)


if __name__ == "__main__":
    unittest.main()
