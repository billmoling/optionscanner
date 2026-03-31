# tests/test_regime_detector.py
import pytest
from datetime import datetime, timezone
from optionscanner.regime.detector import RegimeDetector, RegimeType, RegimeResult


class TestRegimeDetector:
    def test_low_vol_bull(self):
        detector = RegimeDetector()
        market_data = {
            "vix_level": 12.0,
            "spy_vs_ma50": 0.05,  # 5% above MA50
            "qqq_vs_ma50": 0.03,
            "iwm_vs_ma50": 0.02,
        }

        result = detector.detect(market_data)

        assert result.regime == RegimeType.LOW_VOL_BULL
        assert result.confidence > 0.7

    def test_high_vol_bear(self):
        detector = RegimeDetector()
        market_data = {
            "vix_level": 28.0,
            "spy_vs_ma50": -0.08,  # 8% below MA50
            "qqq_vs_ma50": -0.10,
            "iwm_vs_ma50": -0.12,
        }

        result = detector.detect(market_data)

        assert result.regime == RegimeType.HIGH_VOL_BEAR
        assert result.confidence > 0.7

    def test_crush_regime(self):
        detector = RegimeDetector()
        market_data = {
            "vix_level": 40.0,
            "spy_vs_ma50": -0.15,
            "qqq_vs_ma50": -0.18,
            "iwm_vs_ma50": -0.20,
        }

        result = detector.detect(market_data)

        assert result.regime == RegimeType.CRUSH
        assert result.confidence > 0.9
