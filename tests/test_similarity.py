"""Tests for SimilarityMatcher historical signal comparison."""
import pytest
from datetime import datetime, timezone
from pathlib import Path

from data.similarity import SimilarityMatcher, SignalFeatures
from data.history import HistoryStore, SignalOutcome
from strategies.base import TradeSignal


class TestSimilarityMatcher:
    @pytest.fixture
    def history_store(self, tmp_path):
        """Create a HistoryStore instance for testing."""
        return HistoryStore(tmp_path / "history.jsonl")

    def test_extract_features(self, history_store):
        """Test feature extraction from signal and context."""
        matcher = SimilarityMatcher(history_store)
        signal = TradeSignal(
            symbol="NVDA",
            expiry=datetime.now(timezone.utc),
            strike=100.0,
            option_type="CALL",
            direction="BULL_CALL_DEBIT_SPREAD",
            rationale="Test"
        )
        context = {
            "market_state": "bull",
            "vix_level": 18.0,
            "dte": 30,
            "delta": 0.35,
            "underlying_price": 105.0,
            "ma50": 100.0
        }

        features = matcher.extract_features(signal, context)

        assert features.strategy == "BULL_CALL_DEBIT_SPREAD"
        assert features.symbol == "NVDA"
        assert features.market_state == "bull"
        assert features.vix_level == 18.0
        assert features.dte == 30
        assert features.underlying_ma_position == pytest.approx(0.05, rel=0.01)

    def test_find_similar_empty_history(self, history_store):
        """Test finding similar signals with empty history."""
        matcher = SimilarityMatcher(history_store)
        features = SignalFeatures(
            strategy="BULL_CALL", symbol="NVDA", market_state="bull",
            vix_level=18.0, dte=30, delta=0.35, underlying_ma_position=0.05
        )

        similar = matcher.find_similar(features, top_k=10)
        assert len(similar) == 0

    def test_compute_historical_win_rate(self, history_store):
        """Test win rate computation from historical outcomes."""
        matcher = SimilarityMatcher(history_store)

        # Add 4 wins, 1 loss
        outcomes = []
        for i in range(4):
            outcomes.append(SignalOutcome(
                signal_id=f"win_{i}", strategy="Test", symbol="NVDA",
                entry_date="2026-01-01", exit_date="2026-01-05",
                entry_price=2.0, exit_price=4.0, pnl=2.0, outcome="WIN",
                max_profit=2.5, max_loss=0.0
            ))
        outcomes.append(SignalOutcome(
            signal_id="loss_1", strategy="Test", symbol="NVDA",
            entry_date="2026-01-01", exit_date="2026-01-05",
            entry_price=2.0, exit_price=1.0, pnl=-1.0, outcome="LOSS",
            max_profit=0.0, max_loss=-1.0
        ))

        win_rate = matcher.compute_historical_win_rate(outcomes)
        assert win_rate == pytest.approx(0.80)
