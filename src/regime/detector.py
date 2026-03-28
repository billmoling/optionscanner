"""Market regime detection based on VIX and market state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from loguru import logger


class RegimeType(str, Enum):
    """Market regime classifications."""

    LOW_VOL_BULL = "LOW_VOL_BULL"
    NORMAL_BULL = "NORMAL_BULL"
    NORMAL_BEAR = "NORMAL_BEAR"
    HIGH_VOL_BEAR = "HIGH_VOL_BEAR"
    CRUSH = "CRUSH"
    TRANSITION_UP = "TRANSITION_UP"
    TRANSITION_DOWN = "TRANSITION_DOWN"


@dataclass(slots=True)
class RegimeResult:
    """Result of regime detection."""

    regime: RegimeType
    confidence: float
    as_of: datetime
    signals: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "as_of": self.as_of.isoformat(),
            **self.signals
        }


class RegimeDetector:
    """Detects market regime based on VIX and market state.

    Regime taxonomy:
    - LOW_VOL_BULL: VIX < 15, SPY > MA50
    - NORMAL_BULL: VIX 15-20, SPY > MA50
    - NORMAL_BEAR: VIX 15-20, SPY < MA50
    - HIGH_VOL_BEAR: VIX 20-35, SPY < MA50
    - CRUSH: VIX > 35 (any market direction)
    - TRANSITION_*: Detected via trend changes
    """

    def __init__(
        self,
        vix_low: float = 15.0,
        vix_high: float = 25.0,
        vix_extreme: float = 35.0
    ) -> None:
        """Initialize detector.

        Args:
            vix_low: Threshold for low volatility
            vix_high: Threshold for high volatility
            vix_extreme: Threshold for extreme/panic
        """
        self._vix_low = vix_low
        self._vix_high = vix_high
        self._vix_extreme = vix_extreme

    def detect(self, market_data: Dict[str, Any]) -> RegimeResult:
        """Detect current market regime.

        Args:
            market_data: Dict with keys:
                - vix_level: Current VIX
                - spy_vs_ma50: SPY % above/below MA50
                - qqq_vs_ma50: QQQ % above/below MA50
                - iwm_vs_ma50: IWM % above/below MA50

        Returns:
            RegimeResult with classification
        """
        vix = market_data.get("vix_level", 20.0)
        spy_pct = market_data.get("spy_vs_ma50", 0.0)
        qqq_pct = market_data.get("qqq_vs_ma50", 0.0)
        iwm_pct = market_data.get("iwm_vs_ma50", 0.0)

        # Average market breadth
        breadth = (spy_pct + qqq_pct + iwm_pct) / 3.0

        # Determine regime
        regime: RegimeType
        confidence: float
        signals = {
            "vix_level": vix,
            "breadth": breadth,
            "spy_pct": spy_pct,
        }

        # CRUSH regime (VIX > 35 overrides everything)
        if vix > self._vix_extreme:
            regime = RegimeType.CRUSH
            confidence = min(0.9 + (vix - self._vix_extreme) / 20.0, 1.0)

        # HIGH_VOL_BEAR
        elif vix > self._vix_high and breadth < 0:
            regime = RegimeType.HIGH_VOL_BEAR
            confidence = 0.7 + min(abs(breadth) * 2, 0.3)

        # LOW_VOL_BULL
        elif vix < self._vix_low and breadth > 0:
            regime = RegimeType.LOW_VOL_BULL
            confidence = 0.7 + min((self._vix_low - vix) / 10.0, 0.3)

        # NORMAL_BULL
        elif vix < self._vix_high and breadth > 0:
            regime = RegimeType.NORMAL_BULL
            confidence = 0.6

        # NORMAL_BEAR
        elif breadth < 0:
            regime = RegimeType.NORMAL_BEAR
            confidence = 0.6

        # Default
        else:
            regime = RegimeType.NORMAL_BEAR
            confidence = 0.5

        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            as_of=datetime.now(timezone.utc),
            signals=signals
        )

        logger.info(
            "Regime detected | regime={regime} confidence={conf:.2f} vix={vix} breadth={breadth:.2%}",
            regime=regime.value,
            conf=confidence,
            vix=vix,
            breadth=breadth
        )

        return result


__all__ = ["RegimeDetector", "RegimeType", "RegimeResult"]
