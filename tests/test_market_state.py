from __future__ import annotations

import unittest

import pandas as pd

from market_state import MarketState, MarketStateClassifier
from technical_indicators import TechnicalIndicatorProcessor


class MarketStateClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = TechnicalIndicatorProcessor()
        self.classifier = MarketStateClassifier()

    def _make_history(self, closes: list[float]) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
        return pd.DataFrame({"timestamp": dates, "close": closes})

    def test_flags_bull_when_mas_stack(self) -> None:
        history = self._make_history([100 + i for i in range(60)])
        enriched = self.processor.process(history)

        result = self.classifier.classify(enriched, symbol="NVDA")

        self.assertIsNotNone(result)
        self.assertEqual(result.state, MarketState.BULL)

    def test_flags_bear_when_stack_breaks(self) -> None:
        history = self._make_history([200 - i for i in range(60)])
        enriched = self.processor.process(history)

        result = self.classifier.classify(enriched, symbol="NVDA")

        self.assertIsNotNone(result)
        self.assertEqual(result.state, MarketState.BEAR)

    def test_processor_adds_expected_columns(self) -> None:
        history = self._make_history([100 + i for i in range(35)])
        enriched = self.processor.process(history)

        self.assertIn("ma5", enriched.columns)
        self.assertIn("ma10", enriched.columns)
        self.assertIn("ma30", enriched.columns)


if __name__ == "__main__":
    unittest.main()
