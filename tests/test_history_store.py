"""Tests for HistoryStore signal tracking."""
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone

from strategies.base import TradeSignal
from data.history import HistoryStore, SignalOutcome


class TestHistoryStore:
    """Tests for HistoryStore class."""

    def test_init_creates_directory(self, tmp_path):
        """Test that initializing HistoryStore creates the data directory."""
        store = HistoryStore(tmp_path / "history.jsonl")
        assert (tmp_path / "history.jsonl").exists()

    def test_record_signal_writes_to_file(self, tmp_path):
        """Test that record_signal writes to the JSONL file."""
        data_dir = tmp_path / "data"
        store = HistoryStore(data_dir)
        signal = TradeSignal(
            symbol="NVDA",
            expiry=datetime.now(timezone.utc),
            strike=100.0,
            option_type="CALL",
            direction="BULL_CALL_DEBIT_SPREAD",
            rationale="Test signal"
        )
        store.record_signal("test_001", signal, 2.50)
        store.flush()

        lines = (data_dir / "signal_history.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["signal_id"] == "test_001"
        assert data["entry_price"] == 2.50

    def test_record_exit_updates_record(self, tmp_path):
        """Test that record_exit updates the signal record."""
        data_dir = tmp_path / "data"
        store = HistoryStore(data_dir)
        signal = TradeSignal(
            symbol="NVDA",
            expiry=datetime.now(timezone.utc),
            strike=100.0,
            option_type="CALL",
            direction="BULL",
            rationale="Test"
        )
        store.record_signal("test_001", signal, 2.50)
        store.record_exit("test_001", 4.80, datetime.now(timezone.utc))

        outcomes = store.get_all_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].exit_price == 4.80
        assert outcomes[0].outcome == "WIN"

    def test_get_strategy_stats_empty(self, tmp_path):
        """Test get_strategy_stats with no data."""
        data_dir = tmp_path / "data"
        store = HistoryStore(data_dir)
        stats = store.get_strategy_stats("TestStrategy")
        assert stats.trade_count == 0
        assert stats.win_rate == 0.0

    def test_get_strategy_stats_with_data(self, tmp_path):
        """Test get_strategy_stats computes correct statistics."""
        data_dir = tmp_path / "data"
        store = HistoryStore(data_dir)
        # Add 3 wins, 1 loss
        for i in range(3):
            signal = TradeSignal(
                symbol="NVDA",
                expiry=datetime.now(timezone.utc),
                strike=100.0,
                option_type="CALL",
                direction="BULL",
                rationale="Test"
            )
            store.record_signal(f"win_{i}", signal, 2.0, strategy="TestStrategy")
            store.record_exit(f"win_{i}", 4.0, datetime.now(timezone.utc))

        signal = TradeSignal(
            symbol="NVDA",
            expiry=datetime.now(timezone.utc),
            strike=100.0,
            option_type="CALL",
            direction="BULL",
            rationale="Test"
        )
        store.record_signal("loss_1", signal, 2.0, strategy="TestStrategy")
        store.record_exit("loss_1", 1.0, datetime.now(timezone.utc))

        stats = store.get_strategy_stats("TestStrategy", window_days=90)
        assert stats.trade_count == 4
        assert stats.win_rate == 0.75
