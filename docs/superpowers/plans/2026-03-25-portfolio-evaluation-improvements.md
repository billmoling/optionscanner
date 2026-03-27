# Portfolio Evaluation Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance portfolio management to read positions from IBKR, group multi-leg strategies together, and provide actionable suggestions (hold/sell/roll) for each position or strategy group.

**Architecture:** Extend the existing `PortfolioManager` flow by adding a new `PositionEvaluator` class that:
1. Groups individual legs into strategy positions using contract metadata and heuristics
2. Computes PnL, risk metrics, and position characteristics per group
3. Applies rule-based evaluation to generate hold/sell/roll recommendations
4. Integrates with existing Gemini AI for final review and Slack notifications

**Tech Stack:** Python, pandas, ib_async, existing portfolio module patterns

---

## Phase 1: Multi-Leg Position Grouping

### Task 1: Create Position Grouping Data Structures

**Files:**
- Create: `portfolio/evaluation.py`
- Test: `tests/test_portfolio_evaluation.py`

- [ ] **Step 1: Write data structure tests**

```python
# tests/test_portfolio_evaluation.py
import pytest
from portfolio.evaluation import PositionGroup, EvaluationResult

def test_position_group_from_single_leg():
    """Single-leg positions should create a valid group."""
    import pandas as pd
    leg = pd.Series({
        "underlying": "AAPL",
        "symbol": "AAPL260327C00150000",
        "expiry": "2026-03-27",
        "right": "C",
        "strike": 150.0,
        "quantity": -1.0,
        "avg_price": 5.0,
        "market_price": 3.0,
        "sec_type": "OPT",
        "strategy": None,
    })
    group = PositionGroup.from_legs([leg], group_id="test_1")
    assert group.group_id == "test_1"
    assert group.underlying == "AAPL"
    assert len(group.legs) == 1
    assert group.net_quantity == -1.0

def test_position_group_multi_leg_spread():
    """Multi-leg spreads should aggregate correctly."""
    import pandas as pd
    legs = [
        pd.Series({
            "underlying": "AAPL", "symbol": "AAPL260327C00150000",
            "expiry": "2026-03-27", "right": "C", "strike": 150.0,
            "quantity": -1.0, "avg_price": 5.0, "market_price": 3.0,
            "sec_type": "OPT", "strategy": None,
        }),
        pd.Series({
            "underlying": "AAPL", "symbol": "AAPL260327C00155000",
            "expiry": "2026-03-27", "right": "C", "strike": 155.0,
            "quantity": 1.0, "avg_price": 2.0, "market_price": 1.5,
            "sec_type": "OPT", "strategy": None,
        }),
    ]
    group = PositionGroup.from_legs(legs, group_id="spread_1")
    assert group.group_id == "spread_1"
    assert group.underlying == "AAPL"
    assert len(group.legs) == 2
    assert group.net_quantity == 0.0  # Spread is market-neutral
    assert group.total_debit == 3.0  # -5 + 2 = -3 (debit spread)
    assert group.market_value == -1.5  # -3 + 1.5 = -1.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_portfolio_evaluation.py::test_position_group_from_single_leg -v
# Expected: FAIL - ModuleNotFoundError: No module named 'portfolio.evaluation'
```

- [ ] **Step 3: Create evaluation.py with data structures**

```python
"""Position evaluation for hold/sell/roll recommendations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional

import pandas as pd
from loguru import logger


class Recommendation(Enum):
    """Action recommendation for a position or group."""
    HOLD = "hold"
    SELL = "sell"
    ROLL = "roll"
    CLOSE_HALF = "close_half"
    ADJUST = "adjust"


@dataclass(slots=True)
class PositionGroup:
    """
    Groups related option legs into a single strategic position.

    Used to evaluate multi-leg strategies (spreads, condors, PMCCs) as a unit
    rather than individual legs.
    """
    group_id: str
    underlying: str
    strategy_type: Optional[str]  # e.g., "vertical_spread", "condor", "pmcc"
    legs: List[pd.Series] = field(default_factory=list)

    # Aggregated metrics
    net_quantity: float = 0.0
    total_debit: float = 0.0  # Net premium paid (positive) or received (negative)
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0

    # Risk metrics
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Position characteristics
    min_expiry: Optional[date] = None
    max_expiry: Optional[date] = None
    min_strike: float = 0.0
    max_strike: float = 0.0
    width: float = 0.0  # For spreads: max_strike - min_strike

    # Legs metadata
    leg_counts: dict = field(default_factory=dict)  # {"call_short": 1, "call_long": 1}

    @classmethod
    def from_legs(cls, legs: List[pd.Series], group_id: str) -> PositionGroup:
        """Create a PositionGroup from individual leg data."""
        if not legs:
            raise ValueError("At least one leg is required")

        underlying = str(legs[0].get("underlying", legs[0].get("symbol", ""))).upper()
        strategy_type = cls._infer_strategy_type(legs)

        # Aggregate quantities
        quantities = [float(leg.get("quantity", 0.0) or 0.0) for leg in legs]
        avg_prices = [float(leg.get("avg_price", 0.0) or 0.0) for leg in legs]
        market_prices = [float(leg.get("market_price", 0.0) or 0.0) for leg in legs]
        multipliers = [float(leg.get("multiplier", 100.0) or 100.0) for leg in legs]

        net_quantity = sum(quantities)
        total_debit = sum(q * p * m for q, p, m in zip(quantities, avg_prices, multipliers))
        market_value = sum(q * p * m for q, p, m in zip(quantities, market_prices, multipliers))
        unrealized_pnl = -market_value - total_debit  # Short: profit if value decreases
        pnl_pct = (unrealized_pnl / abs(total_debit) * 100) if total_debit != 0 else 0.0

        # Aggregate Greeks
        net_delta = sum(float(leg.get("delta", 0.0) or 0.0) * float(leg.get("quantity", 0.0) or 0.0)
                       for leg in legs)
        net_gamma = sum(float(leg.get("gamma", 0.0) or 0.0) * float(leg.get("quantity", 0.0) or 0.0)
                       for leg in legs)
        net_theta = sum(float(leg.get("theta", 0.0) or 0.0) * float(leg.get("quantity", 0.0) or 0.0)
                       for leg in legs)
        net_vega = sum(float(leg.get("vega", 0.0) or 0.0) * float(leg.get("quantity", 0.0) or 0.0)
                      for leg in legs)

        # Expiry and strike analysis
        expiries = []
        strikes = []
        leg_counts = {"call_short": 0, "call_long": 0, "put_short": 0, "put_long": 0}

        for leg in legs:
            qty = float(leg.get("quantity", 0.0) or 0.0)
            right = str(leg.get("right", "")).upper()
            strike = float(leg.get("strike", 0.0) or 0.0)

            strikes.append(strike)

            expiry_raw = leg.get("expiry", "")
            try:
                expiry = pd.to_datetime(expiry_raw).date()
                expiries.append(expiry)
            except Exception:
                pass

            # Count leg types
            if right == "C":
                if qty < 0:
                    leg_counts["call_short"] += 1
                else:
                    leg_counts["call_long"] += 1
            elif right == "P":
                if qty < 0:
                    leg_counts["put_short"] += 1
                else:
                    leg_counts["put_long"] += 1

        min_expiry = min(expiries) if expiries else None
        max_expiry = max(expiries) if expiries else None
        min_strike = min(strikes) if strikes else 0.0
        max_strike = max(strikes) if strikes else 0.0
        width = max_strike - min_strike

        return cls(
            group_id=group_id,
            underlying=underlying,
            strategy_type=strategy_type,
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
    def _infer_strategy_type(legs: List[pd.Series]) -> Optional[str]:
        """Infer strategy type from leg characteristics."""
        if len(legs) == 1:
            return "single_leg"

        leg_counts = {"call_short": 0, "call_long": 0, "put_short": 0, "put_long": 0}
        expiries = set()
        strikes = set()

        for leg in legs:
            qty = float(leg.get("quantity", 0.0) or 0.0)
            right = str(leg.get("right", "")).upper()
            strike = float(leg.get("strike", 0.0) or 0.0)
            expiry = str(leg.get("expiry", ""))

            strikes.add(strike)
            expiries.add(expiry)

            if right == "C":
                leg_counts["call_short" if qty < 0 else "call_long"] += 1
            elif right == "P":
                leg_counts["put_short" if qty < 0 else "put_long"] += 1

        # Vertical spread: same expiry, 2 strikes, same direction
        if len(expiries) == 1 and len(strikes) == 2:
            if leg_counts["call_short"] == 1 and leg_counts["call_long"] == 1:
                return "vertical_call_spread"
            if leg_counts["put_short"] == 1 and leg_counts["put_long"] == 1:
                return "vertical_put_spread"

        # Iron condor: 4 legs, 4 strikes, both calls and puts
        if len(strikes) == 4 and all(v >= 1 for v in leg_counts.values()):
            return "iron_condor"

        # Straddle/strangle: same expiry, both calls and puts
        if len(expiries) == 1 and leg_counts["call_short"] > 0 and leg_counts["put_short"] > 0:
            if len(strikes) == 1:
                return "short_straddle"
            return "short_strangle"

        # Calendar/diagonal: different expiries
        if len(expiries) > 1:
            if leg_counts["call_short"] == 1 and leg_counts["call_long"] == 1:
                return "calendar_call" if len(strikes) == 1 else "diagonal_call"
            if leg_counts["put_short"] == 1 and leg_counts["put_long"] == 1:
                return "calendar_put" if len(strikes) == 1 else "diagonal_put"

        return "multi_leg_unknown"

    def days_to_expiry(self) -> Optional[int]:
        """Return days until nearest expiry."""
        if self.min_expiry is None:
            return None
        return (self.min_expiry - date.today()).days

    def is_credit_position(self) -> bool:
        """True if position received net credit (short premium)."""
        return self.total_debit < 0

    def profit_factor(self) -> float:
        """Return profit factor (1.0 = breakeven, >1 = profitable)."""
        if self.total_debit == 0:
            return 1.0
        if self.is_credit_position():
            # Credit: profit if unrealized_pnl > 0
            return 1.0 + (self.unrealized_pnl / abs(self.total_debit))
        else:
            # Debit: profit if unrealized_pnl > 0
            return 1.0 + (self.unrealized_pnl / abs(self.total_debit))


@dataclass(slots=True)
class EvaluationResult:
    """Result of evaluating a position group."""
    group_id: str
    underlying: str
    strategy_type: Optional[str]
    recommendation: Recommendation
    confidence: float  # 0.0 to 1.0
    rationale: str
    suggested_action: Optional[str] = None  # e.g., "Roll to AAPL260424C155/160"

    # Metrics snapshot
    pnl_pct: float = 0.0
    days_to_expiry: Optional[int] = None
    net_delta: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "group_id": self.group_id,
            "underlying": self.underlying,
            "strategy_type": self.strategy_type or "unknown",
            "recommendation": self.recommendation.value,
            "confidence": f"{self.confidence:.0%}",
            "rationale": self.rationale,
            "suggested_action": self.suggested_action or "",
            "pnl_pct": f"{self.pnl_pct:.1f}%",
            "days_to_expiry": self.days_to_expiry,
            "net_delta": f"{self.net_delta:.2f}",
        }


__all__ = ["Recommendation", "PositionGroup", "EvaluationResult"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_portfolio_evaluation.py::test_position_group_from_single_leg tests/test_portfolio_evaluation.py::test_position_group_multi_leg_spread -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add portfolio/evaluation.py tests/test_portfolio_evaluation.py
git commit -m "feat: Add PositionGroup and EvaluationResult data structures for multi-leg evaluation"
```

---

### Task 2: Implement Position Grouper

**Files:**
- Modify: `portfolio/evaluation.py` (add PositionGrouper class)
- Test: `tests/test_portfolio_evaluation.py` (add grouping tests)

- [ ] **Step 1: Write grouping tests**

```python
# tests/test_portfolio_evaluation.py (add to file)
from portfolio.evaluation import PositionGrouper

def test_grouper_groups_by_underlying_and_expiry():
    """Legs with same underlying and expiry should be grouped."""
    import pandas as pd
    positions = pd.DataFrame([
        {"underlying": "AAPL", "symbol": "AAPL260327C00150000", "expiry": "2026-03-27",
         "right": "C", "strike": 150.0, "quantity": -1.0, "avg_price": 5.0,
         "market_price": 3.0, "sec_type": "OPT", "strategy": None},
        {"underlying": "AAPL", "symbol": "AAPL260327C00155000", "expiry": "2026-03-27",
         "right": "C", "strike": 155.0, "quantity": 1.0, "avg_price": 2.0,
         "market_price": 1.5, "sec_type": "OPT", "strategy": None},
        {"underlying": "AAPL", "symbol": "AAPL260327P00145000", "expiry": "2026-03-27",
         "right": "P", "strike": 145.0, "quantity": -1.0, "avg_price": 3.0,
         "market_price": 2.0, "sec_type": "OPT", "strategy": None},
    ])
    grouper = PositionGrouper()
    groups = grouper.group(positions)
    assert len(groups) == 1  # All AAPL same expiry should be one group
    assert groups[0].underlying == "AAPL"
    assert len(groups[0].legs) == 3

def test_grouper_separates_different_underlyings():
    """Positions in different underlyings should be separate groups."""
    import pandas as pd
    positions = pd.DataFrame([
        {"underlying": "AAPL", "symbol": "AAPL260327C00150000", "expiry": "2026-03-27",
         "right": "C", "strike": 150.0, "quantity": -1.0, "avg_price": 5.0,
         "market_price": 3.0, "sec_type": "OPT", "strategy": None},
        {"underlying": "TSLA", "symbol": "TSLA260327C00200000", "expiry": "2026-03-27",
         "right": "C", "strike": 200.0, "quantity": -1.0, "avg_price": 10.0,
         "market_price": 8.0, "sec_type": "OPT", "strategy": None},
    ])
    grouper = PositionGrouper()
    groups = grouper.group(positions)
    assert len(groups) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_portfolio_evaluation.py::test_grouper_groups_by_underlying_and_expiry -v
# Expected: FAIL - ImportError: cannot import name 'PositionGrouper'
```

- [ ] **Step 3: Add PositionGrouper class**

```python
# Add to portfolio/evaluation.py

@dataclass(slots=True)
class GrouperConfig:
    """Configuration for position grouping logic."""
    group_by_strategy_column: bool = True  # Use strategy column if available
    max_days_expiry_difference: int = 2  # Max DTE difference to consider same group
    prefer_fewer_groups: bool = True  # Merge when ambiguous


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
        """
        Group individual position legs into strategic positions.

        Args:
            positions: DataFrame with columns from PositionLoader

        Returns:
            List of PositionGroup objects
        """
        if positions.empty:
            return []

        # Filter to options only
        options = positions[positions.get("sec_type", "OPT").astype(str).str.upper() == "OPT"].copy()
        if options.empty:
            return []

        # Create initial groups
        raw_groups = self._create_raw_groups(options)

        # Build PositionGroup objects
        groups = []
        for idx, (group_id, legs) in enumerate(raw_groups.items(), start=1):
            try:
                group = PositionGroup.from_legs(legs, group_id=f"group_{idx}_{group_id}")
                groups.append(group)
            except Exception as exc:
                logger.warning(
                    "Failed to create PositionGroup | group_id={gid} reason={error}",
                    gid=group_id, error=exc
                )

        logger.info("Grouped {count} legs into {groups} positions",
                   count=len(options), groups=len(groups))
        return groups

    def _create_raw_groups(self, options: pd.DataFrame) -> dict:
        """Create initial grouping based on available metadata."""
        groups = {}

        # Check if strategy column has useful data
        if self._config.group_by_strategy_column and "strategy" in options.columns:
            strategy_counts = options["strategy"].value_counts()
            if len(strategy_counts) > 1 and strategy_counts.dropna().sum() > 0:
                return self._group_by_strategy_column(options)

        # Default: group by underlying + expiry
        return self._group_by_underlying_expiry(options)

    def _group_by_strategy_column(self, options: pd.DataFrame) -> dict:
        """Group positions using the strategy column."""
        groups = {}
        for idx, row in options.iterrows():
            strategy = row.get("strategy")
            underlying = str(row.get("underlying", row.get("symbol", ""))).upper()
            group_key = f"{underlying}_{strategy or 'single'}_{idx}"
            groups.setdefault(group_key, []).append(row)
        return groups

    def _group_by_underlying_expiry(self, options: pd.DataFrame) -> dict:
        """Group positions by underlying and expiry proximity."""
        groups = {}

        for idx, row in options.iterrows():
            underlying = str(row.get("underlying", row.get("symbol", ""))).upper()
            expiry_raw = row.get("expiry", "")

            try:
                expiry = pd.to_datetime(expiry_raw).date()
                expiry_key = expiry.isoformat()
            except Exception:
                expiry_key = str(expiry_raw)

            group_key = f"{underlying}_{expiry_key}"
            groups.setdefault(group_key, []).append(row)

        return groups


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
            "take_profit_pct": self._config.get("take_profit_pct", 0.5),  # 50% profit
            "stop_loss_pct": self._config.get("stop_loss_pct", 2.0),  # 200% loss
            "roll_dte_min": self._config.get("roll_dte_min", 14),  # Roll if <14 DTE
            "roll_dte_target": self._config.get("roll_dte_target", 30),  # Roll to 30 DTE
            "close_winnner_dte": self._config.get("close_winner_dte", 7),  # Close if <7 DTE and profitable
        }

    def evaluate(self, groups: List[PositionGroup]) -> List[EvaluationResult]:
        """Evaluate all position groups and return recommendations."""
        results = []
        for group in groups:
            result = self._evaluate_group(group)
            results.append(result)
        return results

    def _evaluate_group(self, group: PositionGroup) -> EvaluationResult:
        """Evaluate a single position group."""
        days_to_expiry = group.days_to_expiry()
        pnl_pct = group.pnl_pct
        is_credit = group.is_credit_position()

        # Rule-based evaluation
        recommendation, confidence, rationale, action = self._apply_rules(
            group, days_to_expiry, pnl_pct, is_credit
        )

        return EvaluationResult(
            group_id=group.group_id,
            underlying=group.underlying,
            strategy_type=group.strategy_type,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            suggested_action=action,
            pnl_pct=pnl_pct,
            days_to_expiry=days_to_expiry,
            net_delta=group.net_delta,
        )

    def _apply_rules(
        self,
        group: PositionGroup,
        days_to_expiry: Optional[int],
        pnl_pct: float,
        is_credit: bool
    ) -> tuple:
        """Apply evaluation rules and return recommendation."""
        rules = self._default_rules

        # Rule 1: Take profit on credit spreads
        if is_credit and pnl_pct >= rules["take_profit_pct"] * 100:
            if days_to_expiry is not None and days_to_expiry < rules["close_winnner_dte"]:
                return (
                    Recommendation.SELL,
                    0.9,
                    f"Profit target reached ({pnl_pct:.0f}%) and near expiry ({days_to_expiry} DTE)",
                    f"Close {group.underlying} position for {pnl_pct:.0f}% profit"
                )
            elif pnl_pct >= 0.8:  # 80% profit rule
                return (
                    Recommendation.SELL,
                    0.8,
                    f"Profit target reached ({pnl_pct:.0f}%)",
                    f"Close {group.underlying} for 80% max profit"
                )

        # Rule 2: Stop loss
        if pnl_pct <= -rules["stop_loss_pct"] * 100:
            return (
                Recommendation.SELL,
                0.7,
                f"Stop loss triggered ({pnl_pct:.0f}%)",
                f"Close {group.underlying} to limit losses"
            )

        # Rule 3: Roll short options near expiry
        if is_credit and days_to_expiry is not None:
            if days_to_expiry < rules["roll_dte_min"]:
                return (
                    Recommendation.ROLL,
                    0.75,
                    f"Near expiry ({days_to_expiry} DTE) with short premium",
                    f"Roll {group.underlying} to {rules['roll_dte_target']} DTE"
                )

        # Rule 4: Close profitable positions near expiry
        if pnl_pct > 0 and days_to_expiry is not None and days_to_expiry < 7:
            return (
                Recommendation.SELL,
                0.85,
                f"Profitable ({pnl_pct:.0f}%) and near expiry ({days_to_expiry} DTE)",
                f"Close {group.underlying} before expiry"
            )

        # Rule 5: Hold if no other rules triggered
        return (
            Recommendation.HOLD,
            0.6,
            f"Holding: {pnl_pct:.0f}% PnL, {days_to_expiry or '?'} DTE, {group.strategy_type or 'position'}",
            None
        )


__all__ = [
    "Recommendation",
    "PositionGroup",
    "EvaluationResult",
    "GrouperConfig",
    "PositionGrouper",
    "PositionEvaluator",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_portfolio_evaluation.py -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add portfolio/evaluation.py tests/test_portfolio_evaluation.py
git commit -m "feat: Add PositionGrouper and PositionEvaluator with rule-based recommendations"
```

---

## Phase 2: Integration with Portfolio Manager

### Task 3: Integrate Evaluator into PortfolioManager

**Files:**
- Modify: `portfolio/manager.py`
- Test: `tests/test_portfolio_manager.py` (integration tests)

- [ ] **Step 1: Write integration test**

```python
# tests/test_portfolio_manager.py
from unittest.mock import Mock
from portfolio.manager import PortfolioManager
from portfolio.evaluation import Recommendation

def test_portfolio_manager_generates_evaluations():
    """PortfolioManager should evaluate positions and generate recommendations."""
    ib_mock = Mock()
    manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

    # Verify evaluator is initialized
    assert hasattr(manager, "evaluator")
    assert manager.evaluator is not None

def test_portfolio_manager_loads_and_groups_positions():
    """PortfolioManager.load_positions() should return grouped positions."""
    ib_mock = Mock()
    manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

    # Mock position loading
    import pandas as pd
    manager.positions = pd.DataFrame([
        {"underlying": "AAPL", "symbol": "AAPL260327C00150000", "expiry": "2026-03-27",
         "right": "C", "strike": 150.0, "quantity": -1.0, "avg_price": 5.0,
         "market_price": 3.0, "sec_type": "OPT", "strategy": None,
         "delta": -0.3, "gamma": 0.02, "theta": 0.1, "vega": 5.0},
    ])

    # Group positions
    groups = manager.group_positions()
    assert len(groups) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_portfolio_manager.py::test_portfolio_manager_generates_evaluations -v
# Expected: FAIL - AttributeError: 'PortfolioManager' object has no attribute 'evaluator'
```

- [ ] **Step 3: Modify PortfolioManager**

```python
# portfolio/manager.py - add imports and modify class

from .evaluation import (
    PositionGrouper,
    PositionEvaluator,
    GrouperConfig,
    EvaluationResult,
)

class PortfolioManager:
    """High level coordinator for portfolio risk management."""

    def __init__(
        self,
        ib: IB,
        config_path: str = "risk.yaml",
        *,
        slack_config: Optional[dict] = None,
        enable_gemini: bool = True,
    ) -> None:
        # ... existing init code ...

        # Add evaluator
        eval_config = self._config_data.get("evaluation", {})
        grouper_config = GrouperConfig(
            group_by_strategy_column=eval_config.get("group_by_strategy", True),
            max_days_expiry_difference=eval_config.get("max_dte_diff", 2),
        )
        self._grouper = PositionGrouper(grouper_config)
        self._evaluator = PositionEvaluator(eval_config)

        # Store evaluation results
        self.position_groups: List[PositionGroup] = []
        self.evaluations: List[EvaluationResult] = []

    # ... existing methods ...

    def group_positions(self) -> List[PositionGroup]:
        """Group loaded positions into strategic positions."""
        logger.info(
            "Grouping positions into strategic units",
            component="portfolio_manager",
            event_type="group_positions_start",
        )
        self.position_groups = self._grouper.group(self.positions)
        logger.info(
            "Grouped {count} positions into {groups} strategic units",
            count=len(self.positions),
            groups=len(self.position_groups),
            component="portfolio_manager",
            event_type="group_positions_complete",
        )
        return self.position_groups

    def evaluate_positions(self) -> List[EvaluationResult]:
        """Evaluate grouped positions and generate recommendations."""
        logger.info(
            "Evaluating positions for hold/sell/roll recommendations",
            component="portfolio_manager",
            event_type="evaluate_positions_start",
        )
        if not self.position_groups:
            logger.warning("No position groups to evaluate; run group_positions() first")
            self.group_positions()

        self.evaluations = self._evaluator.evaluate(self.position_groups)

        # Log summary
        hold_count = sum(1 for e in self.evaluations if e.recommendation == Recommendation.HOLD)
        sell_count = sum(1 for e in self.evaluations if e.recommendation == Recommendation.SELL)
        roll_count = sum(1 for e in self.evaluations if e.recommendation == Recommendation.ROLL)

        logger.info(
            "Position evaluation complete | hold={hold} sell={sell} roll={roll}",
            hold=hold_count, sell=sell_count, roll=roll_count,
            component="portfolio_manager",
            event_type="evaluate_positions_complete",
        )
        return self.evaluations

    def notify(self) -> str:
        logger.info(
            "Preparing portfolio notifications",
            component="portfolio_manager",
            event_type="notify_start",
        )
        message = self._reporter.build_summary_message(
            self.greek_summary.totals,
            self.concentration,
            self.breaches,
            self.actions,
        )

        # Add evaluation results to message
        if self.evaluations:
            eval_lines = self._format_evaluations(self.evaluations)
            message += "\n\nPosition Recommendations:\n" + "\n".join(eval_lines)

        csv_path = None
        json_path = None
        try:
            csv_path, _json_unused, timestamp = self._reporter.write_outputs(
                self.positions, self.greek_summary.per_symbol
            )
            # Also write evaluation results
            eval_csv_path = self._reporter.write_evaluation_results(
                self.evaluations, timestamp
            )
            logger.info(
                "Portfolio outputs written | csv={path} eval_csv={eval_path}",
                path=csv_path, eval_path=eval_csv_path,
                component="portfolio_manager",
                event_type="outputs_written",
            )
            self.last_gemini_response = self._reporter.evaluate_positions_with_gemini(
                self.positions, timestamp
            )
            if self.last_gemini_response:
                logger.info(
                    "Gemini portfolio evaluation completed",
                    component="portfolio_manager",
                    event_type="gemini_evaluation_complete",
                )
        except Exception as exc:
            logger.warning("Failed to write portfolio CSV | reason={error}", error=exc)
            csv_path = None
        self._reporter.log_details(message)
        self._reporter.send_notifications(message, csv_path)
        logger.info(
            "Portfolio notification sent",
            component="portfolio_manager",
            event_type="notify_complete",
        )
        return message

    def _format_evaluations(self, evaluations: List[EvaluationResult]) -> List[str]:
        """Format evaluation results for the summary message."""
        lines = []
        for eval in sorted(evaluations, key=lambda e: e.confidence, reverse=True):
            action_icon = {
                Recommendation.HOLD: "•",
                Recommendation.SELL: "⚠",
                Recommendation.ROLL: "🔄",
            }.get(eval.recommendation, "•")

            pnl_str = f"{eval.pnl_pct:+.0f}%" if eval.pnl_pct else "N/A"
            dte_str = f"{eval.days_to_expiry}d" if eval.days_to_expiry else "?"

            lines.append(
                f"{action_icon} {eval.underlying} [{eval.strategy_type or 'POS'}]: "
                f"{eval.recommendation.value.upper()} ({pnl_str}, {dte_str})"
            )
            if eval.suggested_action:
                lines.append(f"  → {eval.suggested_action}")
        return lines
```

- [ ] **Step 4: Add reporter method for evaluations**

```python
# portfolio/report.py - add to PortfolioReporter

def write_evaluation_results(self, evaluations: Iterable[EvaluationResult], timestamp: str) -> Path:
    """Write evaluation results to CSV."""
    csv_path = self._results_dir / f"position_evaluations_{timestamp}.csv"

    rows = [eval.to_dict() for eval in evaluations]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    logger.info(
        "Wrote position evaluations to {path}",
        path=str(csv_path),
        component="portfolio_reporter",
    )
    return csv_path
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_portfolio_manager.py -v
# Expected: PASS
```

- [ ] **Step 6: Commit**

```bash
git add portfolio/manager.py portfolio/report.py portfolio/evaluation.py tests/test_portfolio_manager.py
git commit -m "feat: Integrate position evaluation into PortfolioManager with reporting"
```

---

## Phase 3: Roll Recommendation Engine

### Task 4: Implement Roll Recommendation Logic

**Files:**
- Modify: `portfolio/evaluation.py` (add RollRecommender class)
- Test: `tests/test_portfolio_evaluation.py` (add roll tests)

- [ ] **Step 1: Write roll recommendation tests**

```python
# tests/test_portfolio_evaluation.py
from portfolio.evaluation import RollRecommender, RollRecommendation

def test_roll_recommender_vertical_spread():
    """Roll recommendations for vertical spreads."""
    import pandas as pd
    group = PositionGroup.from_legs([
        pd.Series({
            "underlying": "AAPL", "expiry": "2026-03-27", "right": "C",
            "strike": 150.0, "quantity": -1.0, "delta": -0.4,
        }),
        pd.Series({
            "underlying": "AAPL", "expiry": "2026-03-27", "right": "C",
            "strike": 155.0, "quantity": 1.0, "delta": 0.2,
        }),
    ], group_id="test")

    recommender = RollRecommender()
    roll_rec = recommender.recommend_roll(group, current_dte=5)

    assert roll_rec.should_roll is True
    assert roll_rec.target_dte == 30  # Default target
    assert "short delta" in roll_rec.rationale.lower() or "expiry" in roll_rec.rationale.lower()

def test_roll_recommender_no_roll_needed():
    """No roll recommendation when position is fine."""
    import pandas as pd
    group = PositionGroup.from_legs([
        pd.Series({
            "underlying": "AAPL", "expiry": "2026-04-17", "right": "C",
            "strike": 150.0, "quantity": -1.0, "delta": -0.2,
        }),
    ], group_id="test")

    recommender = RollRecommender()
    roll_rec = recommender.recommend_roll(group, current_dte=23)

    assert roll_rec.should_roll is False
```

- [ ] **Step 2: Add RollRecommender class**

```python
# Add to portfolio/evaluation.py

@dataclass(slots=True)
class RollRecommendation:
    """Details of a roll recommendation."""
    should_roll: bool
    target_expiry: Optional[date] = None
    target_dte: int = 30
    rationale: str = ""
    adjustment_type: Optional[str] = None  # "roll_up", "roll_down", "roll_out", "roll_in"


class RollRecommender:
    """
    Determines optimal roll timing and strikes for short option positions.

    Roll triggers:
    - DTE < min_dte (typically 7-14 days)
    - Short delta > threshold (typically 0.40-0.50)
    - Profit target reached but DTE still elevated
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._min_dte = self._config.get("min_dte", 7)
        self._target_dte = self._config.get("target_dte", 30)
        self._roll_up_delta = self._config.get("roll_up_delta", 0.40)
        self._profit_roll_threshold = self._config.get("profit_threshold", 0.5)

    def recommend_roll(
        self,
        group: PositionGroup,
        current_dte: Optional[int] = None
    ) -> RollRecommendation:
        """Determine if and how a position should be rolled."""
        if current_dte is None:
            current_dte = group.days_to_expiry()

        reasons_to_roll = []
        adjustment_type = None

        # Check DTE-based roll
        if current_dte is not None and current_dte < self._min_dte:
            reasons_to_roll.append(f"Near expiry ({current_dte} DTE)")
            adjustment_type = "roll_out"

        # Check delta-based roll for short positions
        short_delta = self._get_max_short_delta(group)
        if abs(short_delta) > self._roll_up_delta:
            reasons_to_roll.append(f"Short delta elevated ({short_delta:.2f})")
            adjustment_type = "roll_up" if adjustment_type == "roll_out" else "roll_up"

        # Check profit-based roll
        if group.pnl_pct > self._profit_roll_threshold * 100:
            if current_dte is not None and current_dte > 14:
                reasons_to_roll.append(f"Profit target reached ({group.pnl_pct:.0f}%)")

        if not reasons_to_roll:
            return RollRecommendation(
                should_roll=False,
                rationale="No roll triggers met",
            )

        # Calculate target expiry
        target_expiry = None
        if current_dte is not None:
            from datetime import timedelta
            target_expiry = date.today() + timedelta(days=self._target_dte)

        return RollRecommendation(
            should_roll=True,
            target_expiry=target_expiry,
            target_dte=self._target_dte,
            rationale="; ".join(reasons_to_roll),
            adjustment_type=adjustment_type,
        )

    def _get_max_short_delta(self, group: PositionGroup) -> float:
        """Get the maximum absolute delta among short legs."""
        max_delta = 0.0
        for leg in group.legs:
            qty = float(leg.get("quantity", 0.0) or 0.0)
            if qty < 0:  # Short leg
                delta = abs(float(leg.get("delta", 0.0) or 0.0))
                max_delta = max(max_delta, delta)
        return max_delta


# Update PositionEvaluator to use RollRecommender
class PositionEvaluator:
    # ... existing code ...

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._default_rules = {
            # ... existing rules ...
        }
        self._roll_recommender = RollRecommender(config)

    def _evaluate_group(self, group: PositionGroup) -> EvaluationResult:
        # ... existing evaluation logic ...

        # Enhance roll recommendations with RollRecommender
        if recommendation == Recommendation.ROLL:
            roll_rec = self._roll_recommender.recommend_roll(group, days_to_expiry)
            if roll_rec.should_roll and roll_rec.adjustment_type:
                action = f"Roll {group.underlying} {roll_rec.adjustment_type.replace('_', ' ')}"
                if roll_rec.target_dte:
                    action += f" to {roll_rec.target_dte} DTE"
                return EvaluationResult(
                    group_id=group.group_id,
                    underlying=group.underlying,
                    strategy_type=group.strategy_type,
                    recommendation=recommendation,
                    confidence=confidence,
                    rationale=f"{rationale}; {roll_rec.rationale}",
                    suggested_action=action,
                    pnl_pct=pnl_pct,
                    days_to_expiry=days_to_expiry,
                    net_delta=group.net_delta,
                )

        # ... rest of existing method ...
```

- [ ] **Step 3: Commit**

```bash
git add portfolio/evaluation.py tests/test_portfolio_evaluation.py
git commit -m "feat: Add RollRecommender with delta and DTE-based roll logic"
```

---

## Phase 4: Reporting and Output

### Task 5: Add Evaluation Summary Report

**Files:**
- Modify: `portfolio/report.py`
- Create: `portfolio/templates/evaluation_summary.html` (optional)

- [ ] **Step 1: Add evaluation summary to CSV output**

```python
# portfolio/report.py - modify write_outputs

def write_outputs(self, positions: pd.DataFrame, greek_summary: pd.DataFrame,
                  evaluations: Optional[List[EvaluationResult]] = None) -> (Path, Path, str):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = self._results_dir / f"portfolio_summary_{timestamp}.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        positions.to_csv(fh, index=False)
        fh.write("\n")
        greek_summary.to_csv(fh, index=False)
        if evaluations:
            fh.write("\n")
            eval_df = pd.DataFrame([e.to_dict() for e in evaluations])
            eval_df.to_csv(fh, index=False)

    logger.info("Wrote portfolio summary CSV to {path}", path=str(csv_path))
    return csv_path, None, timestamp
```

- [ ] **Step 2: Commit**

```bash
git add portfolio/report.py
git commit -m "feat: Add evaluation results to portfolio summary CSV output"
```

---

## Configuration Example

Add to `risk.yaml`:

```yaml
evaluation:
  group_by_strategy: true
  max_dte_diff: 2
  take_profit_pct: 0.5
  stop_loss_pct: 2.0
  roll_dte_min: 14
  roll_dte_target: 30
  close_winner_dte: 7
  roll_rules:
    min_dte: 7
    target_dte: 30
    roll_up_delta: 0.40
    profit_threshold: 0.5
```

---

## Testing Checklist

- [ ] Run full test suite: `pytest tests/test_portfolio_evaluation.py -v`
- [ ] Run integration tests: `pytest tests/test_portfolio_manager.py -v`
- [ ] Verify CSV output includes evaluations
- [ ] Verify Slack notifications include recommendations
- [ ] Manual test with live IBKR positions (optional)

---

## Summary

This plan creates:

1. **PositionGroup** - Groups multi-leg strategies into cohesive units
2. **PositionGrouper** - Automatically groups legs by underlying/expiry/strategy
3. **PositionEvaluator** - Rule-based hold/sell/roll recommendations
4. **RollRecommender** - Intelligent roll timing and strike selection
5. **Integration** - Seamless integration with existing PortfolioManager flow

The implementation follows existing patterns in the codebase and uses the same logging, configuration, and reporting infrastructure.
