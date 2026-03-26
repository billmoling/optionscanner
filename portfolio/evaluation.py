"""Position grouping and evaluation data structures for multi-leg option position evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass(slots=True)
class GrouperConfig:
    """Configuration for position grouping logic."""

    group_by_strategy_column: bool = True  # Use strategy column if available
    max_days_expiry_difference: int = 2  # Max DTE difference to consider same group
    prefer_fewer_groups: bool = True  # Merge when ambiguous


class Recommendation(str, Enum):
    """Possible recommendations for a position group."""

    HOLD = "HOLD"
    SELL = "SELL"
    ROLL = "ROLL"
    CLOSE_HALF = "CLOSE_HALF"
    ADJUST = "ADJUST"


@dataclass(slots=True)
class PositionGroup:
    """Represents a group of related option legs forming a single position."""

    group_id: str
    underlying: str
    strategy_type: str
    legs: List[pd.Series]

    # Aggregated metrics
    net_quantity: float
    total_debit: float
    market_value: float
    unrealized_pnl: float
    pnl_pct: float

    # Risk metrics
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Position characteristics
    min_expiry: str = ""
    max_expiry: str = ""
    min_strike: float = 0.0
    max_strike: float = 0.0
    width: float = 0.0

    # Leg counts
    leg_counts: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_legs(cls, legs: List[pd.Series], group_id: str) -> PositionGroup:
        """Create a PositionGroup from a list of leg Series.

        Args:
            legs: List of pandas Series, each representing an option leg
            group_id: Unique identifier for this position group

        Returns:
            PositionGroup with aggregated metrics
        """
        if not legs:
            raise ValueError("At least one leg is required to create a PositionGroup")

        # Extract common attributes
        underlying = str(legs[0].get("underlying", legs[0].get("symbol", ""))).upper()

        # Calculate aggregated metrics
        net_quantity = 0.0
        total_debit = 0.0
        market_value = 0.0
        unrealized_pnl = 0.0

        # Greek aggregates
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0

        # Strike tracking
        strikes: List[float] = []
        expiries: List[str] = []
        leg_counts: Dict[str, int] = {"calls": 0, "puts": 0}

        for leg in legs:
            quantity = float(leg.get("quantity", 0.0) or 0.0)
            avg_price = float(leg.get("avg_price", 0.0) or 0.0)
            market_price = float(leg.get("market_price", 0.0) or 0.0)
            multiplier = float(leg.get("multiplier", 100.0) or 100.0)
            strike = float(leg.get("strike", 0.0) or 0.0)
            expiry = str(leg.get("expiry", ""))
            right = str(leg.get("right", "")).upper()

            net_quantity += quantity
            total_debit += quantity * avg_price * multiplier
            market_value += quantity * market_price * multiplier

            # PnL calculation: (market_price - avg_price) * quantity * multiplier
            # For short positions: profit when price goes down
            unrealized_pnl += (market_price - avg_price) * quantity * multiplier

            # Aggregate greeks if available
            delta = float(leg.get("delta", 0.0) or 0.0)
            gamma = float(leg.get("gamma", 0.0) or 0.0)
            theta = float(leg.get("theta", 0.0) or 0.0)
            vega = float(leg.get("vega", 0.0) or 0.0)

            net_delta += delta * quantity * multiplier
            net_gamma += gamma * quantity * multiplier
            net_theta += theta * quantity * multiplier
            net_vega += vega * quantity * multiplier

            # Track strikes and expiries
            if strike > 0:
                strikes.append(strike)
            if expiry:
                expiries.append(expiry)

            # Count calls and puts
            if right == "C":
                leg_counts["calls"] = leg_counts.get("calls", 0) + 1
            elif right == "P":
                leg_counts["puts"] = leg_counts.get("puts", 0) + 1

        # Calculate PnL percentage relative to initial debit/credit
        if total_debit != 0:
            pnl_pct = unrealized_pnl / abs(total_debit)
        else:
            pnl_pct = 0.0

        # Determine min/max expiry and strike
        min_expiry = min(expiries) if expiries else ""
        max_expiry = max(expiries) if expiries else ""
        min_strike = min(strikes) if strikes else 0.0
        max_strike = max(strikes) if strikes else 0.0
        width = max_strike - min_strike if strikes else 0.0

        # Infer strategy type from legs if not provided
        strategy = cls._infer_strategy_type(legs, leg_counts)

        logger.debug(
            "Created position group | group_id={group_id} underlying={underlying} strategy={strategy} legs={legs}",
            group_id=group_id,
            underlying=underlying,
            strategy=strategy,
            legs=len(legs),
        )

        return cls(
            group_id=group_id,
            underlying=underlying,
            strategy_type=strategy,
            legs=legs,
            net_quantity=net_quantity,
            total_debit=total_debit,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            pnl_pct=pnl_pct,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            min_expiry=min_expiry,
            max_expiry=max_expiry,
            min_strike=min_strike,
            max_strike=max_strike,
            width=width,
            leg_counts=leg_counts,
        )

    @staticmethod
    def _infer_strategy_type(legs: List[pd.Series], leg_counts: Dict[str, int]) -> str:
        """Infer the strategy type from the structure of legs.

        Args:
            legs: List of leg Series
            leg_counts: Dict with 'calls' and 'puts' counts

        Returns:
            Strategy type string
        """
        if not legs:
            return "unknown"

        # Check if strategy is already specified in legs
        for leg in legs:
            strat = leg.get("strategy")
            if strat and strat not in (None, "", "nan"):
                return str(strat).lower()

        num_legs = len(legs)
        calls = leg_counts.get("calls", 0)
        puts = leg_counts.get("puts", 0)

        # Collect expiry and right/strike info for analysis
        expiries: set[str] = set()
        rights: set[str] = set()
        strikes: list[float] = []

        for leg in legs:
            expiry = str(leg.get("expiry", ""))
            if expiry:
                expiries.add(expiry)
            right = str(leg.get("right", "")).upper()
            if right:
                rights.add(right)
            strike = float(leg.get("strike", 0.0) or 0.0)
            if strike > 0:
                strikes.append(strike)

        # Strategy inference logic
        if num_legs == 1:
            return "single_leg"

        if num_legs == 2:
            # Could be vertical spread, calendar spread, or straddle/strangle
            if len(expiries) == 1 and len(rights) == 1:
                # Same expiry, same right = vertical spread
                return "vertical_spread"
            if len(expiries) == 2 and len(rights) == 1:
                # Different expiry, same right = calendar spread
                return "calendar_spread"
            if len(expiries) == 1 and len(rights) == 2:
                # Same expiry, different rights = straddle or strangle
                if len(strikes) == 2 and strikes[0] == strikes[1]:
                    return "straddle"
                return "strangle"

        if num_legs == 3:
            # Could be butterfly, collar, or broken wing butterfly
            if len(expiries) == 1:
                if calls == 2 and puts == 1 or calls == 1 and puts == 2:
                    return "butterfly"
            return "three_leg"

        if num_legs == 4:
            # Could be iron condor, iron butterfly, or box spread
            if len(expiries) == 1 and calls == 2 and puts == 2:
                # Check for iron condor structure (4 different strikes)
                unique_strikes = sorted(set(strikes))
                if len(unique_strikes) == 4:
                    return "iron_condor"
                if len(unique_strikes) == 3:
                    return "iron_butterfly"
            return "four_leg"

        if num_legs > 4:
            return "complex"

        return "unknown"

    def days_to_expiry(self) -> int:
        """Calculate days until the nearest expiry.

        Returns:
            Number of days until min_expiry, or 0 if no expiry set
        """
        if not self.min_expiry:
            return 0

        try:
            expiry_date = self._parse_expiry(self.min_expiry)
            today = date.today()
            delta = expiry_date - today
            return max(0, delta.days)
        except (ValueError, TypeError):
            logger.warning(
                "Failed to parse expiry date | expiry={expiry}",
                expiry=self.min_expiry,
            )
            return 0

    def is_credit_position(self) -> bool:
        """Check if this is a credit position (received premium).

        Returns:
            True if total_debit is negative (net credit received)
        """
        return self.total_debit < 0

    def profit_factor(self) -> float:
        """Calculate profit factor as current market value relative to initial debit.

        Returns:
            Ratio of market_value to total_debit, or 0 if total_debit is 0
        """
        if self.total_debit == 0:
            return 0.0
        return self.market_value / self.total_debit

    @staticmethod
    def _parse_expiry(expiry_str: str) -> date:
        """Parse an expiry string to a date object.

        Args:
            expiry_str: Expiry string in YYYY-MM-DD or YYYYMMDD format

        Returns:
            date object
        """
        # Try YYYY-MM-DD format first
        if "-" in expiry_str:
            return datetime.strptime(expiry_str, "%Y-%m-%d").date()
        # Try YYYYMMDD format
        if len(expiry_str) == 8 and expiry_str.isdigit():
            return datetime.strptime(expiry_str, "%Y%m%d").date()
        # Fallback: try pandas Timestamp
        try:
            ts = pd.Timestamp(expiry_str)
            return ts.to_pydatetime().date()
        except Exception:
            raise ValueError(f"Unable to parse expiry: {expiry_str}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert PositionGroup to dictionary for serialization.

        Returns:
            Dict representation of the position group
        """
        return {
            "group_id": self.group_id,
            "underlying": self.underlying,
            "strategy_type": self.strategy_type,
            "net_quantity": self.net_quantity,
            "total_debit": self.total_debit,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "pnl_pct": self.pnl_pct,
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_theta": self.net_theta,
            "net_vega": self.net_vega,
            "min_expiry": self.min_expiry,
            "max_expiry": self.max_expiry,
            "min_strike": self.min_strike,
            "max_strike": self.max_strike,
            "width": self.width,
            "leg_counts": self.leg_counts,
            "days_to_expiry": self.days_to_expiry(),
            "is_credit": self.is_credit_position(),
        }


@dataclass(slots=True)
class EvaluationResult:
    """Result of evaluating a PositionGroup for potential actions."""

    group_id: str
    underlying: str
    strategy_type: str
    recommendation: Recommendation
    confidence: float
    rationale: str
    suggested_action: str

    # Snapshot of key metrics at evaluation time
    pnl_pct: float = 0.0
    days_to_expiry: int = 0
    net_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationResult to dictionary for CSV export.

        Returns:
            Dict representation suitable for CSV serialization
        """
        return {
            "group_id": self.group_id,
            "underlying": self.underlying,
            "strategy_type": self.strategy_type,
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "suggested_action": self.suggested_action,
            "pnl_pct": self.pnl_pct,
            "days_to_expiry": self.days_to_expiry,
            "net_delta": self.net_delta,
        }


__all__ = [
    "Recommendation",
    "PositionGroup",
    "EvaluationResult",
    "GrouperConfig",
    "PositionGrouper",
    "PositionEvaluator",
]


class PositionGrouper:
    """
    Groups individual option legs into strategic positions.

    Grouping logic:
    1. If strategy column is populated, use it
    2. Otherwise, group by underlying + expiry proximity
    3. Detect spread/condor patterns from leg characteristics
    """

    def __init__(self, config: Optional[GrouperConfig] = None) -> None:
        self._config = config or GrouperConfig()

    def group(self, positions: pd.DataFrame) -> List[PositionGroup]:
        """Group individual position legs into strategic positions.

        Args:
            positions: DataFrame with position legs

        Returns:
            List of PositionGroup objects
        """
        if positions.empty:
            return []

        # Filter to options only
        if "sec_type" in positions.columns:
            options = positions[positions["sec_type"] == "OPT"].copy()
        else:
            options = positions.copy()

        if options.empty:
            return []

        logger.debug(
            "Grouping {count} option positions",
            count=len(options),
        )

        # Create initial groups based on configuration
        if self._config.group_by_strategy_column and "strategy" in options.columns:
            raw_groups = self._group_by_strategy_column(options)
        else:
            raw_groups = self._group_by_underlying_expiry(options)

        # Build PositionGroup objects
        groups: List[PositionGroup] = []
        for group_id, legs in raw_groups.items():
            try:
                position_group = PositionGroup.from_legs(legs, group_id=group_id)
                groups.append(position_group)
            except Exception as e:
                logger.warning(
                    "Failed to create position group {group_id}: {error}",
                    group_id=group_id,
                    error=e,
                )

        logger.info(
            "Created {count} position groups",
            count=len(groups),
        )
        return groups

    def _create_raw_groups(
        self, positions: pd.DataFrame
    ) -> Dict[str, List[pd.Series]]:
        """Create initial grouping based on metadata.

        Args:
            positions: DataFrame with position legs

        Returns:
            Dict mapping group_id to list of leg Series
        """
        if self._config.group_by_strategy_column and "strategy" in positions.columns:
            return self._group_by_strategy_column(positions)
        return self._group_by_underlying_expiry(positions)

    def _group_by_strategy_column(
        self, positions: pd.DataFrame
    ) -> Dict[str, List[pd.Series]]:
        """Group positions using strategy column.

        Args:
            positions: DataFrame with position legs

        Returns:
            Dict mapping group_id to list of leg Series
        """
        groups: Dict[str, List[pd.Series]] = {}

        # Group by strategy + underlying + expiry proximity
        for (strategy, underlying), group in positions.groupby(
            ["strategy", "underlying"]
        ):
            if strategy in (None, "", "nan", float("nan")):
                # Fall back to underlying + expiry grouping
                continue

            # Further subdivide by expiry
            expiry_groups = self._subdivide_by_expiry(
                group, f"{strategy}_{underlying}"
            )
            groups.update(expiry_groups)

        # Handle positions without strategy
        no_strategy = positions[
            positions["strategy"].isin([None, "", "nan"])
            | positions["strategy"].isna()
        ]
        if not no_strategy.empty:
            expiry_groups = self._group_by_underlying_expiry(no_strategy)
            groups.update(expiry_groups)

        return groups

    def _group_by_underlying_expiry(
        self, positions: pd.DataFrame
    ) -> Dict[str, List[pd.Series]]:
        """Default grouping by underlying + expiry.

        Args:
            positions: DataFrame with position legs

        Returns:
            Dict mapping group_id to list of leg Series
        """
        groups: Dict[str, List[pd.Series]] = {}

        for underlying, group in positions.groupby("underlying"):
            expiry_groups = self._subdivide_by_expiry(group, underlying)
            groups.update(expiry_groups)

        return groups

    def _subdivide_by_expiry(
        self, group: pd.DataFrame, base_id: str
    ) -> Dict[str, List[pd.Series]]:
        """Subdivide a group by expiry proximity.

        Args:
            group: DataFrame subset for a single underlying
            base_id: Base identifier for group IDs

        Returns:
            Dict mapping group_id to list of leg Series
        """
        if group.empty:
            return {}

        # Parse expiry dates
        expiries = []
        for _, row in group.iterrows():
            expiry_str = str(row.get("expiry", ""))
            if expiry_str:
                try:
                    expiry_date = self._parse_expiry(expiry_str)
                    expiries.append(expiry_date)
                except (ValueError, TypeError):
                    expiries.append(None)
            else:
                expiries.append(None)

        group = group.copy()
        group["_expiry_date"] = expiries

        # Cluster by expiry proximity
        expiry_clusters: Dict[date, List[int]] = {}
        for idx, expiry_date in enumerate(expiries):
            if expiry_date is None:
                continue

            # Find existing cluster within max_days_expiry_difference
            found_cluster = None
            for cluster_date in expiry_clusters.keys():
                if abs((expiry_date - cluster_date).days) <= self._config.max_days_expiry_difference:
                    found_cluster = cluster_date
                    break

            if found_cluster is not None:
                expiry_clusters[found_cluster].append(idx)
            else:
                expiry_clusters[expiry_date] = [idx]

        # Build groups
        groups: Dict[str, List[pd.Series]] = {}
        for cluster_date, indices in expiry_clusters.items():
            cluster_rows = group.iloc[indices]
            group_id = f"{base_id}_{cluster_date.strftime('%Y%m%d')}"
            groups[group_id] = [
                cluster_rows.iloc[i].drop("_expiry_date") for i in range(len(cluster_rows))
            ]

        return groups

    @staticmethod
    def _parse_expiry(expiry_str: str) -> date:
        """Parse an expiry string to a date object.

        Args:
            expiry_str: Expiry string in YYYY-MM-DD or YYYYMMDD format

        Returns:
            date object
        """
        if "-" in expiry_str:
            return datetime.strptime(expiry_str, "%Y-%m-%d").date()
        if len(expiry_str) == 8 and expiry_str.isdigit():
            return datetime.strptime(expiry_str, "%Y%m%d").date()
        try:
            ts = pd.Timestamp(expiry_str)
            return ts.to_pydatetime().date()
        except Exception:
            raise ValueError(f"Unable to parse expiry: {expiry_str}")


class PositionEvaluator:
    """
    Evaluates grouped positions and generates recommendations.

    Evaluation criteria:
    - PnL thresholds (take profit, stop loss)
    - DTE thresholds (roll windows)
    - Delta/risk thresholds
    - Strategy-specific rules
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._default_rules = {
            "take_profit_pct": self._config.get("take_profit_pct", 0.5),
            "stop_loss_pct": self._config.get("stop_loss_pct", 2.0),
            "roll_dte_min": self._config.get("roll_dte_min", 14),
            "roll_dte_target": self._config.get("roll_dte_target", 30),
            "close_winner_dte": self._config.get("close_winner_dte", 7),
        }

    def evaluate(self, groups: List[PositionGroup]) -> List[EvaluationResult]:
        """Evaluate all position groups and return recommendations.

        Args:
            groups: List of PositionGroup objects to evaluate

        Returns:
            List of EvaluationResult objects
        """
        results: List[EvaluationResult] = []

        for group in groups:
            result = self._evaluate_group(group)
            results.append(result)

        logger.info(
            "Evaluated {count} position groups",
            count=len(results),
        )
        return results

    def _evaluate_group(self, group: PositionGroup) -> EvaluationResult:
        """Evaluate a single position group.

        Args:
            group: PositionGroup to evaluate

        Returns:
            EvaluationResult with recommendation
        """
        days_to_expiry = group.days_to_expiry()
        pnl_pct = group.pnl_pct
        is_credit = group.is_credit_position()

        recommendation, confidence, rationale, suggested_action = self._apply_rules(
            group, days_to_expiry, pnl_pct, is_credit
        )

        logger.debug(
            "Evaluated group {group_id}: {recommendation} ({rationale})",
            group_id=group.group_id,
            recommendation=recommendation.value,
            rationale=rationale,
        )

        return EvaluationResult(
            group_id=group.group_id,
            underlying=group.underlying,
            strategy_type=group.strategy_type,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            suggested_action=suggested_action,
            pnl_pct=pnl_pct,
            days_to_expiry=days_to_expiry,
            net_delta=group.net_delta,
        )

    def _apply_rules(
        self, group: PositionGroup, days_to_expiry: int, pnl_pct: float, is_credit: bool
    ) -> tuple[Recommendation, float, str, str]:
        """Apply evaluation rules and return recommendation.

        Args:
            group: PositionGroup to evaluate
            days_to_expiry: Days until nearest expiry
            pnl_pct: PnL percentage (positive = profit)
            is_credit: True if this is a credit position

        Returns:
            Tuple of (recommendation, confidence, rationale, suggested_action)
        """
        take_profit_pct = self._default_rules["take_profit_pct"]
        stop_loss_pct = self._default_rules["stop_loss_pct"]
        roll_dte_min = self._default_rules["roll_dte_min"]
        close_winner_dte = self._default_rules["close_winner_dte"]

        # Rule 1: Take profit on credit spreads (50%+ profit, or 80% max profit)
        if is_credit and pnl_pct >= take_profit_pct:
            confidence = min(1.0, 0.7 + (pnl_pct - take_profit_pct) * 0.3)
            if pnl_pct >= 0.8:
                rationale = f"Max profit reached ({pnl_pct:.0%} gain)"
                confidence = 0.95
            else:
                rationale = f"Profit target reached ({pnl_pct:.0%} gain)"
            if days_to_expiry <= close_winner_dte:
                return (
                    Recommendation.SELL,
                    confidence,
                    rationale,
                    f"Close {group.strategy_type} position at {pnl_pct:.0%} profit",
                )
            return (
                Recommendation.SELL,
                confidence,
                rationale,
                f"Close {group.strategy_type} position at {pnl_pct:.0%} profit",
            )

        # Rule 2: Stop loss (200%+ loss)
        if pnl_pct <= -stop_loss_pct:
            return (
                Recommendation.SELL,
                0.85,
                f"Stop loss triggered ({pnl_pct:.0%} loss)",
                f"Close {group.strategy_type} to limit losses",
            )

        # Rule 3: Roll short options near expiry (<14 DTE)
        if days_to_expiry > 0 and days_to_expiry <= roll_dte_min:
            # Check if there are short options
            has_short = any(
                float(leg.get("quantity", 0) or 0) < 0 for leg in group.legs
            )
            if has_short:
                if pnl_pct >= take_profit_pct:
                    # Profitable position near expiry - close it
                    return (
                        Recommendation.SELL,
                        0.9,
                        f"Close profitable position near expiry ({pnl_pct:.0%} gain, {days_to_expiry} DTE)",
                        f"Close {group.strategy_type} at {pnl_pct:.0%} profit before expiry",
                    )
                else:
                    # Not profitable, roll it
                    return (
                        Recommendation.ROLL,
                        0.75,
                        f"Roll short options ({days_to_expiry} DTE remaining)",
                        f"Roll {group.strategy_type} to {self._default_rules['roll_dte_target']}DTE",
                    )

        # Rule 4: Close profitable positions near expiry (<7 DTE)
        if days_to_expiry > 0 and days_to_expiry <= close_winner_dte:
            if pnl_pct >= 0.3:  # At least 30% profit
                return (
                    Recommendation.SELL,
                    0.85,
                    f"Close profitable position near expiry ({pnl_pct:.0%} gain, {days_to_expiry} DTE)",
                    f"Close {group.strategy_type} at {pnl_pct:.0%} profit before expiry",
                )

        # Rule 5: Hold if no rules triggered
        rationale = "Position within acceptable parameters"
        if pnl_pct > 0:
            rationale = f"Position profitable ({pnl_pct:.0%} gain), maintain"
        elif pnl_pct < 0:
            rationale = f"Position underwater ({pnl_pct:.0%} loss), monitor"

        return (
            Recommendation.HOLD,
            0.6,
            rationale,
            "Maintain current position",
        )
