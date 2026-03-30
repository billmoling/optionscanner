"""Historical signal similarity matching for performance estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from data.history import HistoryStore, SignalOutcome
from strategies.base import TradeSignal


@dataclass(slots=True)
class SignalFeatures:
    """Feature vector for signal similarity comparison."""

    strategy: str
    symbol: str
    market_state: str
    vix_level: float
    dte: int
    delta: float
    underlying_ma_position: float  # % above/below MA50

    def normalize(self) -> Dict[str, float]:
        """Return normalized feature vector for distance calculation."""
        # Market state encoding
        state_map = {"bull": 1.0, "uptrend": 0.7, "bear": 0.0}

        return {
            "vix": self.vix_level / 50.0,  # Normalize to 0-1 (VIX typically 10-50)
            "dte": min(self.dte / 90.0, 1.0),  # Cap at 90 days
            "delta": abs(self.delta),  # 0-1 range
            "ma_position": max(-1.0, min(self.underlying_ma_position, 1.0)),  # -100% to +100%
            "state": state_map.get(self.market_state.lower(), 0.5),
        }


class SimilarityMatcher:
    """Finds historical signals similar to current candidate.

    Uses Euclidean distance in normalized feature space.
    Weights recent signals higher with exponential decay (half-life = 30 days).
    """

    def __init__(
        self,
        history_store: HistoryStore,
        half_life_days: int = 30
    ) -> None:
        """Initialize matcher.

        Args:
            history_store: History store to query
            half_life_days: Decay half-life for recency weighting
        """
        self._history = history_store
        self._half_life = half_life_days

    def extract_features(
        self,
        signal: TradeSignal,
        context: Dict[str, object]
    ) -> SignalFeatures:
        """Extract feature vector from signal and market context.

        Args:
            signal: Current trade signal
            context: Market context dict with vix_level, market_state, etc.

        Returns:
            SignalFeatures for similarity comparison
        """
        underlying_price = context.get("underlying_price", signal.strike)
        ma50 = context.get("ma50", underlying_price)
        ma_position = (underlying_price - ma50) / ma50 if ma50 > 0 else 0.0

        return SignalFeatures(
            strategy=signal.direction,
            symbol=(signal.symbol or "").upper(),
            market_state=str(context.get("market_state", "unknown")),
            vix_level=float(context.get("vix_level", 20.0)),
            dte=context.get("dte", 30),
            delta=context.get("delta", 0.3),
            underlying_ma_position=ma_position
        )

    def find_similar(
        self,
        features: SignalFeatures,
        top_k: int = 20
    ) -> List[SignalOutcome]:
        """Find historically similar signals.

        Args:
            features: Current signal features
            top_k: Number of similar signals to return

        Returns:
            List of similar historical outcomes, sorted by similarity
        """
        all_outcomes = self._history.get_closed_outcomes()

        if not all_outcomes:
            return []

        # Calculate distance for each historical outcome
        scored: List[tuple[float, SignalOutcome]] = []
        now = datetime.now(timezone.utc)

        for outcome in all_outcomes:
            # Skip outcomes without required features
            if not outcome.exit_date:
                continue

            # Reconstruct features from stored outcome (simplified - uses defaults)
            # In production, would store full feature vector with each outcome
            hist_features = SignalFeatures(
                strategy=outcome.strategy,
                symbol=outcome.symbol,
                market_state="unknown",
                vix_level=20.0,
                dte=30,
                delta=0.3,
                underlying_ma_position=0.0
            )

            distance = self._euclidean_distance(features, hist_features)

            # Apply recency weight
            try:
                exit_date = datetime.fromisoformat(outcome.exit_date)
                days_ago = (now - exit_date).days
                recency_weight = math.pow(0.5, days_ago / self._half_life)
            except (ValueError, TypeError):
                recency_weight = 0.5

            weighted_distance = distance / recency_weight if recency_weight > 0 else distance
            scored.append((weighted_distance, outcome))

        # Sort by weighted distance and return top_k
        scored.sort(key=lambda x: x[0])
        return [outcome for _, outcome in scored[:top_k]]

    def compute_historical_win_rate(
        self,
        outcomes: List[SignalOutcome]
    ) -> float:
        """Compute win rate from a list of outcomes.

        Args:
            outcomes: List of signal outcomes

        Returns:
            Win rate as a float between 0 and 1
        """
        if not outcomes:
            return 0.0

        closed_outcomes = [o for o in outcomes if o.is_closed()]
        if not closed_outcomes:
            return 0.0

        win_count = sum(1 for o in closed_outcomes if o.is_win())
        return win_count / len(closed_outcomes)

    def _euclidean_distance(
        self,
        features: SignalFeatures,
        other: SignalFeatures
    ) -> float:
        """Calculate Euclidean distance between two feature vectors.

        Args:
            features: First feature vector
            other: Second feature vector

        Returns:
            Euclidean distance
        """
        norm1 = features.normalize()
        norm2 = other.normalize()

        sum_squared_diff = 0.0
        for key in norm1:
            diff = norm1[key] - norm2.get(key, 0.0)
            sum_squared_diff += diff * diff

        return math.sqrt(sum_squared_diff)


__all__ = ["SimilarityMatcher", "SignalFeatures"]
