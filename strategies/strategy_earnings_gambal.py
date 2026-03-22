"""Pre-earnings 'gambal' strategy for aggressive directional plays."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

from earnings_data import EarningsFetcher
from market_context import MarketContextProvider

from .base import BaseOptionStrategy, SignalLeg, TradeSignal


@dataclass
class GambalAnalysis:
    """Analysis result for earnings gambal opportunity."""

    symbol: str
    days_to_earnings: int
    recommended_structure: str  # STRADDLE, STRANGLE, DIRECTIONAL_CALL, DIRECTIONAL_PUT
    rationale: str
    risk_reward_ratio: float
    max_profit: float
    max_loss: float
    probability_of_profit: float
    legs: List[SignalLeg]


class EarningsGambalStrategy(BaseOptionStrategy):
    """Aggressive pre-earnings directional options strategy.

    This strategy evaluates both:
    1. Direction-neutral volatility plays (straddles, strangles)
    2. Directional high-delta plays (calls or puts)

    And recommends the structure with the best risk/reward profile.

    Configuration:
        min_days_to_earnings: Minimum days before earnings to enter (default: 1)
        max_days_to_earnings: Maximum days before earnings to enter (default: 5)
        max_position_size: Max position size as fraction of normal (default: 0.5)
        target_delta: Target delta for directional plays (default: 0.40)
        min_expected_move: Minimum expected move to justify straddle (default: 0.05 = 5%)
        straddle_weight: Weight given to straddle analysis (default: 0.5)
        directional_weight: Weight given to directional analysis (default: 0.5)
    """

    def __init__(
        self,
        min_days_to_earnings: int = 1,
        max_days_to_earnings: int = 5,
        max_position_size: float = 0.5,
        target_delta: float = 0.40,
        min_expected_move: float = 0.05,
        straddle_weight: float = 0.5,
        directional_weight: float = 0.5,
        earnings_fetcher: Optional[EarningsFetcher] = None,
        market_context: Optional[MarketContextProvider] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_days_to_earnings = min_days_to_earnings
        self.max_days_to_earnings = max_days_to_earnings
        self.max_position_size = max_position_size
        self.target_delta = target_delta
        self.min_expected_move = min_expected_move
        self.straddle_weight = straddle_weight
        self.directional_weight = directional_weight
        self._earnings = earnings_fetcher
        self._context = market_context

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """Generate pre-earnings gambal signals."""
        now = datetime.now(timezone.utc)
        signals: List[TradeSignal] = []

        for snapshot in data:
            symbol = self._get_symbol(snapshot)
            if not symbol:
                continue

            # Check if in pre-earnings window
            days_to_earnings = self._get_days_to_earnings(symbol)
            if days_to_earnings is None:
                continue

            if not (self.min_days_to_earnings <= days_to_earnings <= self.max_days_to_earnings):
                continue

            # Analyze opportunity
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue

            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None:
                continue

            # Evaluate both straddle and directional
            straddle_analysis = self._analyze_straddle(
                chain, underlying_price, symbol, days_to_earnings
            )
            directional_analysis = self._analyze_directional(
                chain, underlying_price, symbol, days_to_earnings
            )

            # Select best structure
            best = self._select_best_structure(straddle_analysis, directional_analysis)
            if best is None:
                continue

            # Build signal
            signal = self._build_signal(best, underlying_price, snapshot)
            if signal:
                signals.append(signal)
                logger.info(
                    "Generated earnings gambal signal | symbol={symbol} structure={structure} dte_earnings={days}",
                    symbol=symbol,
                    structure=best.recommended_structure,
                    days=days_to_earnings,
                )

        return signals

    def _analyze_straddle(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        symbol: str,
        days_to_earnings: int,
    ) -> Optional[GambalAnalysis]:
        """Analyze long straddle opportunity."""
        try:
            # Get ATM options
            atm_strike = self._find_atm_strike(chain, underlying_price)
            if atm_strike is None:
                return None

            call_row = self._get_option(chain, atm_strike, "CALL")
            put_row = self._get_option(chain, atm_strike, "PUT")

            if call_row is None or put_row is None:
                return None

            call_price = self._get_price(call_row, "ask")
            put_price = self._get_price(put_row, "ask")

            if call_price is None or put_price is None:
                return None

            straddle_cost = call_price + put_price
            breakeven_upper = atm_strike + straddle_cost
            breakeven_lower = atm_strike - straddle_cost

            # Calculate expected move
            expected_move_pct = straddle_cost / underlying_price

            if expected_move_pct < self.min_expected_move:
                logger.info(
                    "Skipping straddle: expected move too small | symbol={symbol} expected={exp:.1%} min={min:.1%}",
                    symbol=symbol,
                    exp=expected_move_pct,
                    min=self.min_expected_move,
                )
                return None

            # Risk/reward: unlimited upside, limited to premium
            max_loss = straddle_cost
            # Assume 50% profit target at 2x premium
            max_profit = straddle_cost * 2.0
            rr_ratio = max_profit / max_loss if max_loss > 0 else 0

            # Probability of profit (simplified: based on expected move vs implied)
            # In production, use IV to calculate probability
            prob_profit = 0.35  # Simplified assumption for straddles

            return GambalAnalysis(
                symbol=symbol,
                days_to_earnings=days_to_earnings,
                recommended_structure="STRADDLE",
                rationale=f"Long straddle: expecting {expected_move_pct:.1%} move, cost ${straddle_cost:.2f}, breakevens ${breakeven_lower:.0f}/${breakeven_upper:.0f}",
                risk_reward_ratio=rr_ratio,
                max_profit=max_profit,
                max_loss=max_loss,
                probability_of_profit=prob_profit,
                legs=[
                    SignalLeg(
                        action="BUY",
                        option_type="CALL",
                        strike=atm_strike,
                        expiry=call_row["expiry"],
                        quantity=1,
                    ),
                    SignalLeg(
                        action="BUY",
                        option_type="PUT",
                        strike=atm_strike,
                        expiry=put_row["expiry"],
                        quantity=1,
                    ),
                ],
            )

        except Exception as exc:
            logger.exception("Error analyzing straddle | symbol={symbol}", symbol=symbol)
            return None

    def _analyze_strangle(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        symbol: str,
        days_to_earnings: int,
    ) -> Optional[GambalAnalysis]:
        """Analyze long strangle opportunity (OTM call + OTM put)."""
        try:
            # Get OTM strikes (one strike away from ATM)
            strikes = sorted(chain["strike"].unique())
            atm_idx = self._find_atm_index(strikes, underlying_price)

            if atm_idx is None or atm_idx <= 0 or atm_idx >= len(strikes) - 1:
                return None

            # One strike OTM for each side
            call_strike = strikes[atm_idx + 1]
            put_strike = strikes[atm_idx - 1]

            call_row = self._get_option(chain, call_strike, "CALL")
            put_row = self._get_option(chain, put_strike, "PUT")

            if call_row is None or put_row is None:
                return None

            call_price = self._get_price(call_row, "ask")
            put_price = self._get_price(put_row, "ask")

            if call_price is None or put_price is None:
                return None

            strangle_cost = call_price + put_price
            breakeven_upper = call_strike + strangle_cost
            breakeven_lower = put_strike - strangle_cost

            expected_move_pct = strangle_cost / underlying_price

            if expected_move_pct < self.min_expected_move * 0.7:  # Lower threshold for strangles
                return None

            max_loss = strangle_cost
            max_profit = strangle_cost * 2.5  # Higher potential than straddle
            rr_ratio = max_profit / max_loss if max_loss > 0 else 0
            prob_profit = 0.30  # Lower prob than straddle (wider breakevens)

            return GambalAnalysis(
                symbol=symbol,
                days_to_earnings=days_to_earnings,
                recommended_structure="STRANGLE",
                rationale=f"Long strangle: OTM {put_strike}/{call_strike}, cost ${strangle_cost:.2f}, expecting >{expected_move_pct:.1%} move",
                risk_reward_ratio=rr_ratio,
                max_profit=max_profit,
                max_loss=max_loss,
                probability_of_profit=prob_profit,
                legs=[
                    SignalLeg(
                        action="BUY",
                        option_type="CALL",
                        strike=call_strike,
                        expiry=call_row["expiry"],
                        quantity=1,
                    ),
                    SignalLeg(
                        action="BUY",
                        option_type="PUT",
                        strike=put_strike,
                        expiry=put_row["expiry"],
                        quantity=1,
                    ),
                ],
            )

        except Exception as exc:
            logger.exception("Error analyzing strangle | symbol={symbol}", symbol=symbol)
            return None

    def _analyze_directional(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        symbol: str,
        days_to_earnings: int,
    ) -> Optional[GambalAnalysis]:
        """Analyze directional high-delta call or put play."""
        try:
            # Determine bias from technical setup
            ma_value = self._get_ma_value(chain)
            bias = self._determine_directional_bias(chain, underlying_price, ma_value)

            if bias is None:
                return None

            is_bullish, confidence = bias

            # Find appropriate strike
            if is_bullish:
                # Slightly ITM call (higher delta)
                target_strike = self._find_strike_below(chain, underlying_price, delta_pct=0.10)
                option_type = "CALL"
            else:
                # Slightly ITM put (higher delta)
                target_strike = self._find_strike_above(chain, underlying_price, delta_pct=0.10)
                option_type = "PUT"

            if target_strike is None:
                return None

            option_row = self._get_option(chain, target_strike, option_type)
            if option_row is None:
                return None

            option_price = self._get_price(option_row, "ask")
            if option_price is None or option_price <= 0:
                return None

            # Risk/reward for directional
            max_loss = option_price * 100  # Per contract

            # Target: 50% gain
            target_price = option_price * 1.5
            max_profit = (target_price - option_price) * 100

            rr_ratio = max_profit / max_loss if max_loss > 0 else 0
            prob_profit = 0.45 + (confidence * 0.1)  # 45-55% based on confidence

            direction_str = "BULLISH" if is_bullish else "BEARISH"
            rationale = (
                f"{direction_str} earnings play: {'Call' if is_bullish else 'Put'} strike ${target_strike:.0f}, "
                f"cost ${option_price:.2f}, confidence {confidence:.0%}"
            )

            return GambalAnalysis(
                symbol=symbol,
                days_to_earnings=days_to_earnings,
                recommended_structure="DIRECTIONAL_CALL" if is_bullish else "DIRECTIONAL_PUT",
                rationale=rationale,
                risk_reward_ratio=rr_ratio,
                max_profit=max_profit,
                max_loss=max_loss,
                probability_of_profit=prob_profit,
                legs=[
                    SignalLeg(
                        action="BUY",
                        option_type=option_type,
                        strike=target_strike,
                        expiry=option_row["expiry"],
                        quantity=1,
                    ),
                ],
            )

        except Exception as exc:
            logger.exception("Error analyzing directional | symbol={symbol}", symbol=symbol)
            return None

    def _select_best_structure(
        self,
        straddle: Optional[GambalAnalysis],
        directional: Optional[GambalAnalysis],
    ) -> Optional[GambalAnalysis]:
        """Select best structure based on weighted scoring."""
        if straddle is None and directional is None:
            return None

        if straddle is None:
            return directional

        if directional is None:
            return straddle

        # Score each structure
        straddle_score = (
            self.straddle_weight * straddle.risk_reward_ratio * straddle.probability_of_profit
        )

        directional_score = (
            self.directional_weight * directional.risk_reward_ratio * directional.probability_of_profit
        )

        if straddle_score >= directional_score:
            return straddle
        return directional

    def _build_signal(
        self,
        analysis: GambalAnalysis,
        underlying_price: float,
        snapshot: Any,
    ) -> Optional[TradeSignal]:
        """Build TradeSignal from analysis."""
        try:
            expiry = analysis.legs[0].expiry if analysis.legs else None
            if expiry is None:
                return None

            strike = analysis.legs[0].strike
            option_type = analysis.legs[0].option_type

            # Determine overall direction
            if analysis.recommended_structure in ("STRADDLE", "STRANGLE"):
                direction = "EARNINGS_STRADDLE"
            elif "CALL" in analysis.recommended_structure:
                direction = "BULL"
            else:
                direction = "BEAR"

            rationale = (
                f"EARNINGS GAMBAL ({analysis.recommended_structure}): {analysis.rationale} | "
                f"R/R {analysis.risk_reward_ratio:.2f}, Max Profit ${analysis.max_profit:.0f}, "
                f"Max Loss ${analysis.max_loss:.0f}, Prob {analysis.probability_of_profit:.0%}"
            )

            return self.emit_signal(
                TradeSignal(
                    symbol=analysis.symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    direction=direction,
                    rationale=rationale,
                    legs=tuple(analysis.legs),
                    risk_reward_ratio=analysis.risk_reward_ratio,
                    max_profit=analysis.max_profit,
                    max_loss=analysis.max_loss,
                )
            )

        except Exception as exc:
            logger.exception("Error building gambal signal | symbol={symbol}", symbol=analysis.symbol)
            return None

    def _get_symbol(self, snapshot: Any) -> Optional[str]:
        """Extract symbol from snapshot."""
        if hasattr(snapshot, "symbol"):
            return str(snapshot.symbol)
        if isinstance(snapshot, dict) and "symbol" in snapshot:
            return str(snapshot["symbol"])
        return None

    def _get_days_to_earnings(self, symbol: str) -> Optional[int]:
        """Get days until earnings for symbol."""
        if self._context:
            if symbol in (self._context.get_context().earnings_map if self._context.get_context() else {}):
                return self._context.get_context().earnings_map[symbol]

        if self._earnings:
            return self._earnings.get_days_to_earnings(symbol)

        return None

    def _to_dataframe(self, snapshot: Any) -> pd.DataFrame:
        """Convert snapshot to DataFrame."""
        if isinstance(snapshot, pd.DataFrame):
            return snapshot.copy()
        if hasattr(snapshot, "to_pandas"):
            return snapshot.to_pandas()
        return pd.DataFrame(snapshot.options if hasattr(snapshot, "options") else [])

    def _find_atm_strike(self, chain: pd.DataFrame, underlying_price: float) -> Optional[float]:
        """Find at-the-money strike."""
        if chain.empty or "strike" not in chain.columns:
            return None

        strikes = chain["strike"].unique()
        idx = self._find_atm_index(strikes, underlying_price)
        if idx is None:
            return None

        return float(strikes[idx])

    def _find_atm_index(self, strikes: List[float], underlying_price: float) -> Optional[int]:
        """Find index of ATM strike."""
        if not strikes:
            return None

        idx = (pd.Series(strikes) - underlying_price).abs().argsort().iloc[0]
        return int(idx)

    def _get_option(
        self,
        chain: pd.DataFrame,
        strike: float,
        option_type: str,
    ) -> Optional[pd.Series]:
        """Get option row from chain."""
        subset = chain[
            (chain["strike"] == strike) &
            (chain["option_type"].str.upper() == option_type.upper())
        ]
        if subset.empty:
            return None
        return subset.iloc[0]

    def _get_price(self, row: pd.Series, price_type: str = "mark") -> Optional[float]:
        """Get option price from row."""
        # Try primary price type first
        if price_type in row and pd.notna(row[price_type]) and row[price_type] > 0:
            return float(row[price_type])

        # Fallbacks
        fallbacks = ["mark", "last", "bid", "ask"]
        for fallback in fallbacks:
            if fallback in row and pd.notna(row[fallback]) and row[fallback] > 0:
                return float(row[fallback])

        # Mid price from bid/ask
        if "bid" in row and "ask" in row:
            bid = float(row["bid"]) if pd.notna(row.get("bid")) else 0
            ask = float(row["ask"]) if pd.notna(row.get("ask")) else 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2

        return None

    def _get_ma_value(self, chain: pd.DataFrame) -> Optional[float]:
        """Get moving average value from chain."""
        for col in ["ma30", "ma50", "moving_average_30", "moving_average_50"]:
            if col in chain.columns:
                values = chain[col].dropna()
                if not values.empty:
                    return float(values.iloc[-1])
        return None

    def _determine_directional_bias(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        ma_value: Optional[float],
    ) -> Optional[Tuple[bool, float]]:
        """Determine directional bias and confidence.

        Returns:
            (is_bullish, confidence) or None if no clear bias
        """
        if ma_value is None:
            return None

        # Price above MA = bullish bias
        is_bullish = underlying_price > ma_value

        # Confidence based on distance from MA
        distance_pct = abs(underlying_price - ma_value) / ma_value
        confidence = min(distance_pct / 0.05, 1.0)  # 5% distance = 100% confidence

        return (is_bullish, confidence)

    def _find_strike_below(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        delta_pct: float = 0.10,
    ) -> Optional[float]:
        """Find strike below underlying price (for ITM calls)."""
        strikes = sorted(chain["strike"].unique())
        target = underlying_price * (1 - delta_pct)

        for strike in reversed(strikes):
            if strike <= target:
                return float(strike)

        return None

    def _find_strike_above(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        delta_pct: float = 0.10,
    ) -> Optional[float]:
        """Find strike above underlying price (for ITM puts)."""
        strikes = sorted(chain["strike"].unique())
        target = underlying_price * (1 + delta_pct)

        for strike in strikes:
            if strike >= target:
                return float(strike)

        return None


__all__ = ["EarningsGambalStrategy", "GambalAnalysis"]
