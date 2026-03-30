"""Composite signal ranking engine for selecting top trade ideas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from loguru import logger

from optionscanner.data.history import HistoryStore
from optionscanner.data.similarity import SimilarityMatcher, SignalFeatures
from optionscanner.strategies.base import TradeSignal
from optionscanner.trade_history import TradeHistory, StrategyStats

if TYPE_CHECKING:
    from optionscanner.market_context import MarketContextProvider


@dataclass(slots=True)
class SignalScore:
    """Composite scoring result for a trade signal."""

    signal: TradeSignal
    strategy_name: str
    composite_score: float
    win_rate_score: float
    rr_score: float
    perf_score: float
    similarity_score: float
    reason: str
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.signal.symbol,
            "strategy": self.strategy_name,
            "direction": self.signal.direction,
            "strike": self.signal.strike,
            "expiry": self.signal.expiry.isoformat() if hasattr(self.signal.expiry, "isoformat") else str(self.signal.expiry),
            "composite_score": round(self.composite_score, 3),
            "win_rate_score": round(self.win_rate_score, 3),
            "rr_score": round(self.rr_score, 3),
            "perf_score": round(self.perf_score, 3),
            "similarity_score": round(self.similarity_score, 3),
            "reason": self.reason,
        }


@dataclass(slots=True)
class StrategyConfig:
    """Configuration for a strategy's scoring parameters."""

    name: str
    published_win_rate: float = 0.5  # Default 50% if not specified
    target_trade_count: int = 30  # Number of trades before switching to live performance


class SignalRanker:
    """Ranks trade signals using composite scoring.

    Composite score formula:
        score = (win_rate_w * win_rate_score) + (rr_w * rr_score) + (perf_w * perf_score)
                + (similarity_w * similarity_score) - context_penalty

    Weights shift based on trade count per strategy:
        - < 30 trades: win_rate=0.5, rr=0.3, perf=0.2, similarity=0.0
        - >= 30 trades: win_rate=0.2, rr=0.3, perf=0.4, similarity=0.1

    Context penalties:
        - Pre-earnings symbols: -0.20
        - Economic blackout period: -0.15
        - High VIX (25-35): -0.15, Extreme VIX (>35): -0.30
    """

    def __init__(
        self,
        trade_history: TradeHistory,
        strategy_configs: Dict[str, StrategyConfig],
        top_k: int = 5,
        market_context: Optional[MarketContextProvider] = None,
        signal_history: Optional[HistoryStore] = None,
    ) -> None:
        self._trade_history = trade_history
        self._strategy_configs = strategy_configs or {}
        self._top_k = top_k
        self._market_context = market_context
        self._signal_history = signal_history

        # Initialize similarity matcher if signal history provided
        self._similarity_matcher: Optional[SimilarityMatcher] = None
        if signal_history is not None:
            self._similarity_matcher = SimilarityMatcher(signal_history)

        # Default weights
        self._weights_initial = {"win_rate": 0.5, "rr": 0.3, "perf": 0.2, "similarity": 0.0}
        self._weights_mature = {"win_rate": 0.2, "rr": 0.3, "perf": 0.4, "similarity": 0.1}

    def rank_signals(
        self,
        signals: List[Tuple[str, TradeSignal]],
    ) -> List[SignalScore]:
        """Rank a list of signals and return top K by composite score.

        Args:
            signals: List of (strategy_name, TradeSignal) tuples

        Returns:
            List of SignalScore objects sorted by composite_score (descending)
        """
        scored: List[SignalScore] = []

        for strategy_name, signal in signals:
            score = self._score_signal(strategy_name, signal)
            if score is not None:
                scored.append(score)

        # Sort by composite score descending
        scored.sort(key=lambda s: s.composite_score, reverse=True)

        # Return top K
        return scored[: self._top_k]

    def _score_signal(
        self,
        strategy_name: str,
        signal: TradeSignal,
    ) -> Optional[SignalScore]:
        """Compute composite score for a single signal."""
        config = self._strategy_configs.get(
            strategy_name,
            StrategyConfig(name=strategy_name),
        )

        # Get live performance stats
        stats = self._trade_history.get_strategy_stats(strategy_name)
        trade_count = stats.trade_count

        # Determine weights based on trade count
        weights = self._get_weights(trade_count, config.target_trade_count)

        # Compute component scores
        win_rate_score = self._compute_win_rate_score(stats, config.published_win_rate)
        rr_score = self._compute_rr_score(signal)
        perf_score = self._compute_perf_score(stats)
        similarity_score = self._compute_similarity_score(strategy_name, signal)

        # Compute context penalty
        context_penalty = self._compute_context_penalty(signal.symbol)

        # Composite score (with penalty)
        composite = (
            weights["win_rate"] * win_rate_score
            + weights["rr"] * rr_score
            + weights["perf"] * perf_score
            + weights["similarity"] * similarity_score
            - context_penalty
        )

        # Generate reason
        reason = self._generate_reason(
            strategy_name,
            stats,
            config.published_win_rate,
            win_rate_score,
            rr_score,
            perf_score,
            trade_count,
            context_penalty,
            similarity_score,
        )

        return SignalScore(
            signal=signal,
            strategy_name=strategy_name,
            composite_score=composite,
            win_rate_score=win_rate_score,
            rr_score=rr_score,
            perf_score=perf_score,
            similarity_score=similarity_score,
            reason=reason,
            weights=weights,
        )

    def _get_weights(self, trade_count: int, target: int) -> Dict[str, float]:
        """Get scoring weights based on trade count."""
        if trade_count >= target:
            return dict(self._weights_mature)
        return dict(self._weights_initial)

    def _compute_similarity_score(
        self,
        strategy_name: str,
        signal: TradeSignal,
    ) -> float:
        """Compute historical similarity score (0-1 scale).

        Uses SimilarityMatcher to find historically similar signals
        and computes their win rate. Returns 0.5 if no history available.

        Args:
            strategy_name: Strategy name
            signal: Trade signal to score

        Returns:
            Similarity score (0-1), higher = more favorable historical similarity
        """
        if self._similarity_matcher is None:
            return 0.5  # Neutral score if no history available

        try:
            # Extract features from signal using market context
            market_context_dict = None
            if self._market_context:
                ctx = self._market_context.get_context()
                if ctx:
                    market_context_dict = ctx.to_dict()

            features = self._similarity_matcher.extract_features(
                signal,
                market_context_dict or {}
            )

            # Find similar historical signals
            similar_outcomes = self._similarity_matcher.find_similar(features, top_k=20)

            if not similar_outcomes:
                return 0.5  # No similar history

            # Compute win rate from similar outcomes
            historical_win_rate = self._similarity_matcher.compute_historical_win_rate(
                similar_outcomes
            )

            logger.info(
                "Similarity score computed | strategy={strategy} symbol={symbol} similar={count} hist_wr={wr:.0%}",
                strategy=strategy_name,
                symbol=signal.symbol,
                count=len(similar_outcomes),
                wr=historical_win_rate
            )

            return historical_win_rate
        except Exception as exc:
            logger.warning(
                "Failed to compute similarity score | strategy={strategy} symbol={symbol} error={error}",
                strategy=strategy_name,
                symbol=signal.symbol,
                error=exc
            )
            return 0.5  # Fallback to neutral

    def _compute_win_rate_score(
        self,
        stats: StrategyStats,
        published_win_rate: float,
    ) -> float:
        """Compute win rate score (0-1 scale).

        Uses published win rate initially, transitioning to live win rate
        as trade count increases.
        """
        if stats.trade_count == 0:
            return published_win_rate

        live_win_rate = stats.win_rate

        # Smooth transition: weight increases with trade count
        # At 0 trades: 100% published
        # At 30 trades: 50/50 blend
        # At 60+ trades: 100% live
        transition_factor = min(stats.trade_count / 60.0, 1.0)

        return (1 - transition_factor) * published_win_rate + transition_factor * live_win_rate

    def _compute_rr_score(self, signal: TradeSignal) -> float:
        """Compute risk/reward score (0-1 scale).

        Higher is better. Uses signal's risk_reward_ratio if available,
        otherwise defaults to 0.5 (neutral).
        """
        # Check if signal has risk_reward_ratio attribute
        rr_ratio = getattr(signal, "risk_reward_ratio", None)

        if rr_ratio is None or rr_ratio <= 0:
            return 0.5  # Neutral score if no data

        # Normalize: 1.0 = 0.5 score, 2.0 = 0.7 score, 3.0+ = 1.0 score
        # This creates a reasonable spread for typical R/R ratios
        if rr_ratio >= 3.0:
            return 1.0
        elif rr_ratio <= 1.0:
            return 0.3  # Minimum viable score
        else:
            # Linear interpolation between 1.0 and 3.0
            return 0.3 + (rr_ratio - 1.0) * 0.35

    def _compute_perf_score(self, stats: StrategyStats) -> float:
        """Compute live performance score (0-1 scale).

        Based on average P&L and consistency.
        """
        if stats.trade_count == 0:
            return 0.5  # Neutral for new strategies

        # Score based on win rate (component of performance)
        win_rate_component = stats.win_rate

        # Score based on average P&L (normalized)
        # Assuming typical trade P&L ranges from -$5 to +$5 per contract
        avg_pnl = stats.avg_pnl
        if avg_pnl >= 2.0:
            pnl_component = 1.0
        elif avg_pnl <= -2.0:
            pnl_component = 0.0
        else:
            pnl_component = 0.5 + (avg_pnl / 4.0)  # -2 to +2 maps to 0 to 1

        return 0.6 * win_rate_component + 0.4 * pnl_component

    def _compute_context_penalty(self, symbol: str) -> float:
        """Compute context penalty based on market conditions.

        Penalties:
        - Pre-earnings (within 5 days): -0.20
        - Economic blackout period: -0.15
        - High VIX (25-35): -0.15
        - Extreme VIX (>35): -0.30

        Args:
            symbol: Symbol to check for earnings proximity

        Returns:
            Total penalty value (0-1) to subtract from composite score
        """
        if not self._market_context:
            return 0.0

        return self._market_context.get_penalty(symbol)

    def _generate_reason(
        self,
        strategy_name: str,
        stats: StrategyStats,
        published_win_rate: float,
        win_rate_score: float,
        rr_score: float,
        perf_score: float,
        trade_count: int,
        context_penalty: float = 0.0,
        similarity_score: float = 0.5,
    ) -> str:
        """Generate human-readable ranking reason."""
        reasons: List[str] = []

        # Win rate assessment
        if trade_count >= 30:
            wr_label = f"Live win rate {stats.win_rate:.0%}"
        elif trade_count > 0:
            wr_label = f"Blend win rate {win_rate_score:.0%}"
        else:
            wr_label = f"Published win rate {published_win_rate:.0%}"

        if win_rate_score >= 0.65:
            reasons.append(f"High {wr_label}")
        elif win_rate_score >= 0.5:
            reasons.append(f"Moderate {wr_label}")
        else:
            reasons.append(f"Below avg {wr_label}")

        # R/R assessment
        if rr_score >= 0.7:
            reasons.append("favorable R/R")
        elif rr_score >= 0.4:
            reasons.append("acceptable R/R")
        else:
            reasons.append("tight R/R")

        # Performance assessment (for mature strategies)
        if trade_count >= 10:
            if perf_score >= 0.7:
                reasons.append(f"strong live perf ({stats.win_rate:.0%} win, ${stats.avg_pnl:.2f} avg)")
            elif perf_score >= 0.4:
                reasons.append(f"stable live perf ({stats.win_rate:.0%} win)")

        # Historical similarity assessment
        if similarity_score >= 0.65:
            reasons.append(f"favorable historical pattern ({similarity_score:.0%} similar win rate)")
        elif similarity_score >= 0.5:
            reasons.append("neutral historical pattern")
        else:
            reasons.append(f"unfavorable historical pattern ({similarity_score:.0%} similar win rate)")

        # Context penalties
        if context_penalty >= 0.3:
            reasons.append("HIGH RISK: extreme market conditions")
        elif context_penalty >= 0.15:
            reasons.append("elevated risk (market context)")
        elif context_penalty > 0:
            reasons.append("minor risk penalty")

        return "; ".join(reasons)


def load_strategy_configs(config: Dict[str, object]) -> Dict[str, StrategyConfig]:
    """Load strategy configurations from config.yaml structure.

    Expected format:
    strategies:
      VerticalSpreadStrategy:
        params:
          # ... strategy params
        published_win_rate: 0.65
    """
    configs: Dict[str, StrategyConfig] = {}
    strategies = config.get("strategies", {})

    for name, strategy_config in strategies.items():
        if not isinstance(strategy_config, dict):
            continue

        published_wr = float(strategy_config.get("published_win_rate", 0.5))

        configs[name] = StrategyConfig(
            name=name,
            published_win_rate=published_wr,
        )

    return configs


__all__ = [
    "SignalRanker",
    "SignalScore",
    "StrategyConfig",
    "load_strategy_configs",
]
