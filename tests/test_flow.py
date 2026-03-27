# tests/test_flow.py
import pytest
from datetime import datetime, timezone
from data.flow import OptionsFlowFetcher, FlowAlert


class TestOptionsFlowFetcher:
    def test_compute_flow_score_high_volume(self):
        fetcher = OptionsFlowFetcher()
        alerts = [
            FlowAlert(
                symbol="NVDA", timestamp=datetime.now(timezone.utc),
                volume=10000, open_interest=2000, volume_oi_ratio=5.0,
                premium=50000.0, side="BUY", sweep_detected=False
            )
        ]

        score = fetcher.compute_flow_score("NVDA", alerts)
        assert score > 0.7  # High volume/OI ratio should score high

    def test_compute_flow_score_sweep(self):
        fetcher = OptionsFlowFetcher()
        alerts = [
            FlowAlert(
                symbol="NVDA", timestamp=datetime.now(timezone.utc),
                volume=5000, open_interest=2000, volume_oi_ratio=2.5,
                premium=25000.0, side="BUY", sweep_detected=True
            )
        ]

        score = fetcher.compute_flow_score("NVDA", alerts)
        assert score > 0.5  # Sweep detection should boost score

    def test_compute_flow_score_low_activity(self):
        fetcher = OptionsFlowFetcher()
        alerts = [
            FlowAlert(
                symbol="NVDA", timestamp=datetime.now(timezone.utc),
                volume=100, open_interest=2000, volume_oi_ratio=0.05,
                premium=500.0, side="BUY", sweep_detected=False
            )
        ]

        score = fetcher.compute_flow_score("NVDA", alerts)
        assert score < 0.3  # Low activity should score low
