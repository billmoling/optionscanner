"""Options flow detection for unusual activity monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger


@dataclass(slots=True)
class FlowAlert:
    """Represents unusual options activity alert."""

    symbol: str
    timestamp: datetime
    volume: int
    open_interest: int
    volume_oi_ratio: float
    premium: float
    side: str  # "BUY" or "SELL"
    sweep_detected: bool


class OptionsFlowFetcher:
    """Detects and scores unusual options flow activity.

    Monitors volume/OI ratio and sweep detection to identify
    potentially informed trading activity.
    """

    def __init__(
        self,
        volume_oi_threshold: float = 2.0
    ) -> None:
        """Initialize flow fetcher.

        Args:
            volume_oi_threshold: Volume/OI ratio threshold for alerts
        """
        self._threshold = volume_oi_threshold

    def compute_flow_score(
        self,
        symbol: str,
        alerts: List[FlowAlert]
    ) -> float:
        """Compute flow score (0-1) from alerts.

        Args:
            symbol: Underlying symbol
            alerts: List of flow alerts for this symbol

        Returns:
            Flow score (0-1, higher = more unusual activity)
        """
        if not alerts:
            return 0.5  # Neutral if no data

        scores: List[float] = []

        for alert in alerts:
            # Base score from volume/OI ratio
            ratio_score = min(alert.volume_oi_ratio / self._threshold, 2.0) / 2.0

            # Sweep bonus
            sweep_bonus = 0.2 if alert.sweep_detected else 0.0

            # Premium size bonus (scale by premium, cap at 0.2)
            premium_bonus = min(alert.premium / 100000.0, 0.2)

            # Combine
            alert_score = min(ratio_score + sweep_bonus + premium_bonus, 1.0)
            scores.append(alert_score)

        # Average score across all alerts
        avg_score = sum(scores) / len(scores)

        logger.debug(
            "Computed flow score | symbol={symbol} score={score:.2f} alerts={count}",
            symbol=symbol,
            score=avg_score,
            count=len(alerts)
        )

        return avg_score

    def fetch_unusual_activity(
        self,
        symbols: List[str]
    ) -> Dict[str, List[FlowAlert]]:
        """Fetch unusual activity for symbols.

        Note: This is a placeholder for future IBKR integration.
        Currently returns empty dicts - actual implementation requires
        IBKR option chain volume/OI data.

        Args:
            symbols: Symbols to check

        Returns:
            Dict mapping symbol to list of alerts
        """
        # Placeholder - returns empty for all symbols
        # Future implementation will query IBKR for:
        # - Current option chain volume
        # - Historical average volume
        # - Open interest by strike
        # - Multi-leg detection for sweeps

        logger.debug(
            "Unusual activity fetch (placeholder) | symbols={count}",
            count=len(symbols)
        )

        return {symbol: [] for symbol in symbols}


__all__ = ["OptionsFlowFetcher", "FlowAlert"]
