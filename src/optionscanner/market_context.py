"""Unified market context provider with VIX monitoring and scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from earnings_data import EarningsFetcher
from economic_calendar import EconomicEvent, EconomicEventTracker
from market_state import MarketState, MarketStateResult
from option_data import IBKRDataFetcher


@dataclass(slots=True)
class VIXState:
    """Current VIX level and interpretation."""

    level: float
    state: str  # LOW, NORMAL, HIGH, EXTREME
    change: Optional[float] = None  # Day change if available
    timestamp: Optional[datetime] = None

    @property
    def description(self) -> str:
        """Human-readable VIX state description."""
        descriptions = {
            "LOW": "Complacency (VIX < 15)",
            "NORMAL": "Normal conditions (15-25)",
            "HIGH": "Elevated fear (25-35)",
            "EXTREME": "Panic/Crisis (>35)",
        }
        return descriptions.get(self.state, f"Unknown ({self.level:.1f})")


@dataclass
class MarketContextResult:
    """Aggregated market context for signal generation and ranking."""

    # VIX state
    vix: Optional[VIXState] = None

    # Market states for key indices
    spy_state: Optional[MarketState] = None
    qqq_state: Optional[MarketState] = None
    iwm_state: Optional[MarketState] = None

    # Earnings context (symbol -> days until earnings)
    earnings_map: Dict[str, int] = field(default_factory=dict)

    # Upcoming economic events
    economic_events: List[EconomicEvent] = field(default_factory=list)

    # Computed context score (0-1, higher = more favorable)
    context_score: float = 1.0

    # Warnings and messages
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Slack formatting."""
        return {
            "vix_level": self.vix.level if self.vix else None,
            "vix_state": self.vix.state if self.vix else None,
            "spy_state": self.spy_state.value if self.spy_state else None,
            "qqq_state": self.qqq_state.value if self.qqq_state else None,
            "context_score": round(self.context_score, 2),
            "earnings_count": len(self.earnings_map),
            "upcoming_events": len(self.economic_events),
            "warnings": self.warnings,
        }


@dataclass
class MarketContextConfig:
    """Configuration for market context scoring."""

    # VIX thresholds
    vix_low_threshold: float = 15.0
    vix_high_threshold: float = 25.0
    vix_extreme_threshold: float = 35.0

    # Scoring weights
    vix_weight: float = 0.4
    market_state_weight: float = 0.3
    earnings_weight: float = 0.2
    economic_weight: float = 0.1

    # Earnings penalty
    earnings_penalty_days: int = 5
    earnings_penalty: float = 0.2

    # Economic event penalty
    economic_blackout_days: int = 2
    economic_penalty: float = 0.15


class MarketContextProvider:
    """Provides unified market context for signal generation and ranking.

    Aggregates:
    - VIX level and state
    - Market states for SPY, QQQ, IWM
    - Earnings calendar
    - Economic event calendar

    Computes a context score that can be used to:
    - Adjust signal rankings
    - Filter risky signals
    - Add warnings to Slack messages
    """

    def __init__(
        self,
        earnings_fetcher: Optional[EarningsFetcher] = None,
        economic_tracker: Optional[EconomicEventTracker] = None,
        config: Optional[MarketContextConfig] = None,
    ) -> None:
        self._earnings = earnings_fetcher or EarningsFetcher()
        self._economic = economic_tracker or EconomicEventTracker()
        self._config = config or MarketContextConfig()
        self._cache: Optional[MarketContextResult] = None
        self._cache_timestamp: Optional[datetime] = None

    def refresh_context(
        self,
        ibkr_fetcher: Optional[IBKRDataFetcher] = None,
        market_states: Optional[Dict[str, MarketState]] = None,
        symbols: Optional[List[str]] = None,
    ) -> MarketContextResult:
        """Refresh and return current market context.

        Args:
            ibkr_fetcher: IBKR data fetcher for VIX (optional, uses cached if not provided)
            market_states: Pre-computed market states for symbols
            symbols: Symbols to check for earnings

        Returns:
            MarketContextResult with all context data
        """
        symbols = symbols or ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "AMD"]

        result = MarketContextResult()

        # Get VIX state
        result.vix = self._get_vix_state(ibkr_fetcher)

        # Get market states for key indices
        if market_states:
            result.spy_state = market_states.get("SPY")
            result.qqq_state = market_states.get("QQQ")
            result.iwm_state = market_states.get("IWM")

        # Get earnings context
        if symbols and self._earnings.is_configured:
            earnings = self._earnings.get_upcoming_earnings(
                symbols,
                window_days=self._config.earnings_penalty_days,
            )
            result.earnings_map = earnings
            if earnings:
                earnings_list = ", ".join(f"{sym}({days}d)" for sym, days in sorted(earnings.items(), key=lambda x: x[1])[:5])
                result.info_messages.append(f"Earnings watch: {earnings_list}")

        # Get economic events
        result.economic_events = self._economic.get_high_impact_events(window_days=7)
        if result.economic_events:
            result.warnings.append(self._economic.get_warning_message(window_days=7))

        # Compute context score
        result.context_score = self._compute_context_score(result)

        self._cache = result
        self._cache_timestamp = datetime.now(timezone.utc)

        logger.info(
            "Market context refreshed | score={score} vix={vix} spy={spy} qqq={qqq}",
            score=round(result.context_score, 2),
            vix=result.vix.level if result.vix else "N/A",
            spy=result.spy_state.value if result.spy_state else "N/A",
            qqq=result.qqq_state.value if result.qqq_state else "N/A",
        )

        return result

    def get_context(self) -> Optional[MarketContextResult]:
        """Get cached context result."""
        return self._cache

    def get_context_score(self) -> float:
        """Get current context score (0-1)."""
        if self._cache:
            return self._cache.context_score
        return 1.0

    def should_warn(self, symbol: Optional[str] = None) -> bool:
        """Check if there are any active warnings."""
        if not self._cache:
            return False

        if self._cache.warnings:
            return True

        if symbol and symbol in self._cache.earnings_map:
            return True

        return False

    def get_penalty(self, symbol: Optional[str] = None) -> float:
        """Get total penalty to apply to signal score.

        Args:
            symbol: Optional symbol to check for earnings proximity

        Returns:
            Penalty value (0-1) to subtract from signal score
        """
        if not self._cache:
            return 0.0

        penalty = 0.0

        # VIX penalty
        if self._cache.vix:
            if self._cache.vix.state == "EXTREME":
                penalty += 0.3
            elif self._cache.vix.state == "HIGH":
                penalty += 0.15

        # Earnings penalty
        if symbol and symbol in self._cache.earnings_map:
            penalty += self._config.earnings_penalty

        # Economic blackout penalty
        if self._economic.is_blackout_period():
            penalty += self._config.economic_penalty

        return min(penalty, 1.0)

    def is_pre_earnings(self, symbol: str) -> bool:
        """Check if symbol is in pre-earnings window."""
        return symbol in (self._cache.earnings_map if self._cache else {})

    def get_earnings_phase(self, symbol: str) -> str:
        """Get earnings phase for a symbol."""
        if not self._cache:
            return "NONE"

        if symbol in self._cache.earnings_map:
            return "PRE"

        # Check if just reported (simplified - would need more data)
        return "NONE"

    def get_earnings_warning(self, symbol: str) -> Optional[str]:
        """Get earnings warning message for a symbol."""
        if not self._cache or symbol not in self._cache.earnings_map:
            return None

        days = self._cache.earnings_map[symbol]
        if days == 0:
            return "Earnings TODAY"
        elif days == 1:
            return "Earnings TOMORROW"
        else:
            return f"Earnings in {days} days"

    def _get_vix_state(
        self,
        ibkr_fetcher: Optional[IBKRDataFetcher] = None,
    ) -> Optional[VIXState]:
        """Get current VIX state."""
        # Try to get from IBKR if fetcher provided
        if ibkr_fetcher and hasattr(ibkr_fetcher, "ib") and ibkr_fetcher.ib:
            try:
                # This would require additional IBKR calls
                # For now, we'll use a placeholder
                pass
            except Exception as exc:
                logger.debug("Failed to fetch VIX from IBKR | error={error}", error=exc)

        # Fallback: try to get from cached data or return None
        # In production, you'd fetch VIX from a data source
        return self._estimate_vix_state()

    def _estimate_vix_state(self) -> Optional[VIXState]:
        """Estimate VIX state from available data.

        This is a placeholder. In production, fetch actual VIX data.
        For now, returns None to indicate VIX data is unavailable.
        """
        # TODO: Implement actual VIX fetch from IBKR or other source
        return None

    def _compute_context_score(self, context: MarketContextResult) -> float:
        """Compute overall context score (0-1)."""
        score = 1.0

        # VIX component (lower VIX = higher score)
        vix_score = self._compute_vix_score(context.vix)
        score -= (1 - vix_score) * self._config.vix_weight

        # Market state component
        market_score = self._compute_market_state_score(
            context.spy_state,
            context.qqq_state,
        )
        score -= (1 - market_score) * self._config.market_state_weight

        # Earnings component (pre-earnings symbols reduce score)
        if context.earnings_map:
            earnings_ratio = len(context.earnings_map) / 10.0  # Assume 10 symbols max
            score -= min(earnings_ratio, 1.0) * self._config.earnings_weight

        # Economic events component
        if context.economic_events:
            high_impact_count = sum(1 for e in context.economic_events if e.is_high_impact)
            score -= min(high_impact_count * 0.05, 0.2) * self._config.economic_weight

        return max(score, 0.0)

    def _compute_vix_score(self, vix: Optional[VIXState]) -> float:
        """Compute VIX component score (0-1, higher = better)."""
        if vix is None:
            return 0.5  # Neutral if no data

        level = vix.level

        if level < self._config.vix_low_threshold:
            return 1.0  # Complacency = good for risk-taking
        elif level < self._config.vix_high_threshold:
            # Linear interpolation: 15->0.8, 25->0.5
            return 0.8 - (level - 15) / 10 * 0.3
        elif level < self._config.vix_extreme_threshold:
            # Linear interpolation: 25->0.5, 35->0.2
            return 0.5 - (level - 25) / 10 * 0.3
        else:
            return 0.2  # Panic = avoid risk

    def _compute_market_state_score(
        self,
        spy_state: Optional[MarketState],
        qqq_state: Optional[MarketState],
    ) -> float:
        """Compute market state component score (0-1)."""
        if spy_state is None and qqq_state is None:
            return 0.5  # Neutral if no data

        state_scores = {
            MarketState.BULL: 1.0,
            MarketState.UPTREND: 0.7,
            MarketState.BEAR: 0.3,
        }

        score = 0.0
        count = 0

        if spy_state:
            score += state_scores.get(spy_state, 0.5)
            count += 1

        if qqq_state:
            score += state_scores.get(qqq_state, 0.5)
            count += 1

        return score / count if count > 0 else 0.5

    def format_slack_header(self) -> str:
        """Format market context for Slack message header."""
        if not self._cache:
            return ""

        lines = []

        # VIX line
        if self._cache.vix:
            vix_emoji = {
                "LOW": ":green_circle:",
                "NORMAL": ":white_circle:",
                "HIGH": ":orange_circle:",
                "EXTREME": ":red_circle:",
            }.get(self._cache.vix.state, ":white_circle:")
            lines.append(f"VIX: {self._cache.vix.level:.1f} {vix_emoji} ({self._cache.vix.state})")

        # Market states
        state_emoji = {
            "bull": ":green_circle:",
            "uptrend": ":large_blue_circle:",
            "bear": ":red_circle:",
        }

        if self._cache.spy_state:
            emoji = state_emoji.get(self._cache.spy_state.value, "")
            lines.append(f"SPY: {self._cache.spy_state.value.upper()} {emoji}")

        if self._cache.qqq_state:
            emoji = state_emoji.get(self._cache.qqq_state.value, "")
            lines.append(f"QQQ: {self._cache.qqq_state.value.upper()} {emoji}")

        # Context score
        score = self._cache.context_score
        if score >= 0.7:
            score_emoji = ":green_circle:"
        elif score >= 0.4:
            score_emoji = ":yellow_circle:"
        else:
            score_emoji = ":red_circle:"
        lines.append(f"Context Score: {score:.2f} {score_emoji}")

        # Earnings
        if self._cache.earnings_map:
            earnings_str = ", ".join(f"{s}({d}d)" for s, d in sorted(self._cache.earnings_map.items(), key=lambda x: x[1])[:3])
            lines.append(f"Earnings: {earnings_str}")

        # Economic events
        if self._cache.economic_events:
            events_str = ", ".join(f"{e.event_type}({(e.date - date.today()).days}d)" for e in self._cache.economic_events[:3])
            lines.append(f"Events: {events_str}")

        return " | ".join(lines)


__all__ = [
    "MarketContextProvider",
    "MarketContextResult",
    "MarketContextConfig",
    "VIXState",
]
