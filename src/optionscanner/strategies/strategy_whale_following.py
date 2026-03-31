"""Whale Following strategy - trades based on Reddit whale activity detection."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from optionscanner.whale_detector import WhaleActivity, WhaleDetector, WhaleDirection

from .base import BaseOptionStrategy, SignalLeg, TradeSignal


class WhaleFollowingStrategy(BaseOptionStrategy):
    """Generates trade signals based on detected Reddit whale activity.

    This strategy monitors Reddit posts for mentions of large options trades
    (whale activity) and generates corresponding trade signals to follow
    the detected smart money flow.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_upvotes: int = 50,
        max_positions: int = 3,
        min_days_to_expiry: int = 7,
        max_days_to_expiry: int = 30,
        target_delta: float = 0.35,
        whale_detector: Optional[WhaleDetector] = None,
        whale_activities: Optional[Dict[str, WhaleActivity]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Whale Following strategy.

        Args:
            min_confidence: Minimum confidence score for whale activity
            min_upvotes: Minimum upvotes required for source posts
            max_positions: Maximum concurrent positions to hold
            min_days_to_expiry: Minimum days to expiry for options
            max_days_to_expiry: Maximum days to expiry for options
            target_delta: Target delta for option selection
            whale_detector: Optional pre-configured WhaleDetector instance
            whale_activities: Pre-detected whale activities (for testing)
        """
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.min_upvotes = min_upvotes
        self.max_positions = max_positions
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry
        self.target_delta = target_delta
        self._whale_detector = whale_detector or WhaleDetector(
            min_confidence=min_confidence,
        )
        self._whale_activities = whale_activities or {}

    def set_whale_activities(self, activities: Dict[str, WhaleActivity]) -> None:
        """
        Update the detected whale activities.

        Args:
            activities: Dictionary mapping symbols to WhaleActivity
        """
        self._whale_activities = activities
        logger.info(
            "Updated whale activities | count={count} symbols={symbols}",
            count=len(activities),
            symbols=list(activities.keys())[:5],  # Show first 5
        )

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """
        Process option chain data and generate signals based on whale activity.

        Args:
            data: Iterable of OptionChainSnapshot objects

        Returns:
            List of TradeSignal objects
        """
        signals: List[TradeSignal] = []
        now = datetime.now(timezone.utc)

        # Convert data to accessible format
        snapshots = list(data)
        if not snapshots:
            return signals

        # If no whale activities detected, return empty
        if not self._whale_activities:
            logger.debug("No whale activities to act upon")
            return signals

        # Track how many signals we've generated
        signal_count = 0

        for snapshot in snapshots:
            if signal_count >= self.max_positions:
                break

            symbol = self._snapshot_value(snapshot, "symbol")
            if not symbol:
                continue

            symbol_upper = str(symbol).upper()

            # Check if we have whale activity for this symbol
            whale_activity = self._whale_activities.get(symbol_upper)
            if not whale_activity:
                continue

            # Validate whale activity meets thresholds
            if whale_activity.confidence < self.min_confidence:
                logger.debug(
                    "Skipping {symbol} - confidence {confidence:.2f} below threshold",
                    symbol=symbol_upper,
                    confidence=whale_activity.confidence,
                )
                continue

            if whale_activity.total_score < self.min_upvotes:
                logger.debug(
                    "Skipping {symbol} - upvotes {upvotes} below threshold",
                    symbol=symbol_upper,
                    upvotes=whale_activity.total_score,
                )
                continue

            # Get option chain data
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue

            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None:
                continue

            logger.info(
                "Processing whale activity | symbol={symbol} direction={direction} confidence={confidence:.2f} upvotes={upvotes}",
                symbol=symbol_upper,
                direction=whale_activity.direction.value,
                confidence=whale_activity.confidence,
                upvotes=whale_activity.total_score,
            )

            # Select expiry
            expiry = self._select_expiry(chain, now)
            if expiry is None:
                continue

            # Generate signal based on whale direction
            if whale_activity.direction == WhaleDirection.BULLISH:
                signal = self._build_bullish_signal(
                    chain=chain,
                    symbol=symbol_upper,
                    expiry=expiry,
                    underlying_price=underlying_price,
                    whale_activity=whale_activity,
                )
            elif whale_activity.direction == WhaleDirection.BEARISH:
                signal = self._build_bearish_signal(
                    chain=chain,
                    symbol=symbol_upper,
                    expiry=expiry,
                    underlying_price=underlying_price,
                    whale_activity=whale_activity,
                )
            else:
                continue

            if signal:
                signals.append(signal)
                signal_count += 1

        return signals

    def _build_bullish_signal(
        self,
        chain: Any,
        symbol: str,
        expiry: Any,
        underlying_price: float,
        whale_activity: WhaleActivity,
    ) -> Optional[TradeSignal]:
        """Build a bullish call option signal."""
        try:
            calls = chain[chain["option_type"] == "CALL"].copy()
            if calls.empty:
                return None

            # Filter by expiry
            calls = calls[calls["expiry"] == expiry]
            if calls.empty:
                return None

            # Find OTM calls (strike > underlying price)
            otm_calls = calls[calls["strike"] > underlying_price]
            if otm_calls.empty:
                # Fallback to ATM if no OTM available
                otm_calls = calls

            # Sort by delta approximation (closer to target delta is better)
            # Use moneyness as proxy for delta
            otm_calls["moneyness"] = otm_calls["strike"] / underlying_price
            target_moneyness = 1.0 + (1.0 - self.target_delta) * 0.1
            otm_calls["delta_diff"] = abs(otm_calls["moneyness"] - target_moneyness)

            selected = otm_calls.loc[otm_calls["delta_diff"].idxmin()]

            strike = float(selected["strike"])
            price_data = self._get_price_data(selected)

            rationale = (
                f"Whale Following: Detected {whale_activity.direction.value} whale activity | "
                f"confidence={whale_activity.confidence:.2f} | "
                f"sources={len(whale_activity.source_posts)} posts, {len(whale_activity.source_comments)} comments | "
                f"total upvotes={whale_activity.total_score} | "
                f"{whale_activity.details}"
            )

            return self.emit_signal(
                TradeSignal(
                    symbol=symbol,
                    expiry=expiry if hasattr(expiry, "to_pydatetime") else expiry,
                    strike=strike,
                    option_type="CALL",
                    direction="LONG_CALL",
                    rationale=rationale,
                    legs=(
                        SignalLeg(
                            action="BUY",
                            option_type="CALL",
                            strike=strike,
                            expiry=expiry if hasattr(expiry, "to_pydatetime") else expiry,
                        ),
                    ),
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to build bullish signal for {symbol} | error={error}",
                symbol=symbol,
                error=str(exc),
            )
            return None

    def _build_bearish_signal(
        self,
        chain: Any,
        symbol: str,
        expiry: Any,
        underlying_price: float,
        whale_activity: WhaleActivity,
    ) -> Optional[TradeSignal]:
        """Build a bearish put option signal."""
        try:
            puts = chain[chain["option_type"] == "PUT"].copy()
            if puts.empty:
                return None

            # Filter by expiry
            puts = puts[puts["expiry"] == expiry]
            if puts.empty:
                return None

            # Find OTM puts (strike < underlying price)
            otm_puts = puts[puts["strike"] < underlying_price]
            if otm_puts.empty:
                # Fallback to ATM if no OTM available
                otm_puts = puts

            # Sort by delta approximation
            otm_puts["moneyness"] = otm_puts["strike"] / underlying_price
            target_moneyness = 1.0 - (1.0 - self.target_delta) * 0.1
            otm_puts["delta_diff"] = abs(otm_puts["moneyness"] - target_moneyness)

            selected = otm_puts.loc[otm_puts["delta_diff"].idxmin()]

            strike = float(selected["strike"])
            price_data = self._get_price_data(selected)

            rationale = (
                f"Whale Following: Detected {whale_activity.direction.value} whale activity | "
                f"confidence={whale_activity.confidence:.2f} | "
                f"sources={len(whale_activity.source_posts)} posts, {len(whale_activity.source_comments)} comments | "
                f"total upvotes={whale_activity.total_score} | "
                f"{whale_activity.details}"
            )

            return self.emit_signal(
                TradeSignal(
                    symbol=symbol,
                    expiry=expiry if hasattr(expiry, "to_pydatetime") else expiry,
                    strike=strike,
                    option_type="PUT",
                    direction="LONG_PUT",
                    rationale=rationale,
                    legs=(
                        SignalLeg(
                            action="BUY",
                            option_type="PUT",
                            strike=strike,
                            expiry=expiry if hasattr(expiry, "to_pydatetime") else expiry,
                        ),
                    ),
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to build bearish signal for {symbol} | error={error}",
                symbol=symbol,
                error=str(exc),
            )
            return None

    def _select_expiry(
        self,
        chain: Any,
        now: datetime,
    ) -> Optional[Any]:
        """Select the best expiry date from the chain."""
        if "expiry" not in chain.columns:
            return None

        expiries = chain["expiry"].dropna().unique()
        if len(expiries) == 0:
            return None

        best_expiry = None
        best_distance = None

        for expiry in expiries:
            expiry_ts = expiry
            if hasattr(expiry, "to_pydatetime"):
                expiry_ts = expiry.to_pydatetime()
            elif not isinstance(expiry, datetime):
                try:
                    expiry_ts = datetime.fromisoformat(str(expiry))
                except (ValueError, TypeError):
                    continue

            if getattr(expiry_ts, "tzinfo", None) is None:
                expiry_ts = expiry_ts.replace(tzinfo=timezone.utc)

            days = (expiry_ts - now).days
            if days < self.min_days_to_expiry or days > self.max_days_to_expiry:
                continue

            # Prefer expiry closest to middle of our range
            target_days = (self.min_days_to_expiry + self.max_days_to_expiry) / 2
            distance = abs(days - target_days)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_expiry = expiry

        return best_expiry

    def _get_price_data(self, row: Any) -> Dict[str, float]:
        """Extract price data from a row."""
        return {
            "bid": float(row.get("bid", 0) or 0),
            "ask": float(row.get("ask", 0) or 0),
            "mark": float(row.get("mark", 0) or 0),
            "last": float(row.get("last", 0) or 0),
        }


__all__ = ["WhaleFollowingStrategy"]
