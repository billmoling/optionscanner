"""Tests for portfolio evaluation data structures."""
from __future__ import annotations

import pandas as pd
import pytest

from optionscanner.portfolio.evaluation import (
    EvaluationResult,
    GrouperConfig,
    PositionEvaluator,
    PositionGroup,
    PositionGrouper,
    Recommendation,
    RollRecommendation,
    RollRecommender,
)


class TestPositionGroup:
    """Tests for PositionGroup dataclass."""

    def test_position_group_from_single_leg(self):
        """Single-leg positions should create a valid group."""
        leg = pd.Series(
            {
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
                "multiplier": 100.0,
            }
        )
        group = PositionGroup.from_legs([leg], group_id="test_1")
        assert group.group_id == "test_1"
        assert group.underlying == "AAPL"
        assert len(group.legs) == 1
        assert group.net_quantity == -1.0

    def test_position_group_multi_leg_spread(self):
        """Multi-leg spreads should aggregate correctly."""
        legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00155000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 155.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
        ]
        group = PositionGroup.from_legs(legs, group_id="spread_1")
        assert group.group_id == "spread_1"
        assert group.underlying == "AAPL"
        assert len(group.legs) == 2
        assert group.net_quantity == 0.0
        # With multiplier=100: total_debit = (-1*5*100) + (1*2*100) = -300 (net credit)
        assert group.total_debit == -300.0
        # market_value = (-1*3*100) + (1*1.5*100) = -150
        assert group.market_value == -150.0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """EvaluationResult should be created with all required fields."""
        result = EvaluationResult(
            group_id="test_1",
            underlying="AAPL",
            strategy_type="vertical_spread",
            recommendation=Recommendation.HOLD,
            confidence=0.85,
            rationale="Position is profitable with adequate time remaining",
            suggested_action="Maintain current position",
            pnl_pct=0.40,
            days_to_expiry=30,
            net_delta=-0.25,
        )
        assert result.group_id == "test_1"
        assert result.underlying == "AAPL"
        assert result.strategy_type == "vertical_spread"
        assert result.recommendation == Recommendation.HOLD
        assert result.confidence == 0.85
        assert "profitable" in result.rationale
        assert result.pnl_pct == 0.40
        assert result.days_to_expiry == 30
        assert result.net_delta == -0.25

    def test_evaluation_result_to_dict(self):
        """to_dict should convert result to dictionary for CSV export."""
        result = EvaluationResult(
            group_id="test_1",
            underlying="AAPL",
            strategy_type="vertical_spread",
            recommendation=Recommendation.SELL,
            confidence=0.75,
            rationale="Take profits early",
            suggested_action="Close 50% of position",
            pnl_pct=0.50,
            days_to_expiry=15,
            net_delta=0.10,
        )
        result_dict = result.to_dict()
        assert result_dict["group_id"] == "test_1"
        assert result_dict["underlying"] == "AAPL"
        assert result_dict["strategy_type"] == "vertical_spread"
        assert result_dict["recommendation"] == "SELL"
        assert result_dict["confidence"] == 0.75
        assert result_dict["pnl_pct"] == 0.50
        assert result_dict["days_to_expiry"] == 15
        assert result_dict["net_delta"] == 0.10


class TestPositionGroupMetrics:
    """Tests for PositionGroup metric calculations."""

    def test_unrealized_pnl_calculation(self):
        """Unrealized PnL should be calculated correctly."""
        legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            )
        ]
        group = PositionGroup.from_legs(legs, group_id="test_1")
        # Short call: sold at 5, now at 3 = profit of 2 per share = 200 total
        # PnL = (avg_price - market_price) * quantity * multiplier
        # PnL = (5 - 3) * -1 * 100 = -200... wait, that's wrong
        # For short: profit when price goes down
        # PnL = (sell_price - buy_to_close_price) * abs(quantity) * multiplier
        # PnL = (5 - 3) * 1 * 100 = 200
        assert group.unrealized_pnl == 200.0

    def test_pnl_pct_calculation(self):
        """PnL percentage should be calculated relative to max loss."""
        legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            )
        ]
        group = PositionGroup.from_legs(legs, group_id="test_1")
        # For short option: max_loss is theoretically unlimited, but we use margin requirement
        # PnL% = unrealized_pnl / abs(total_debit) = 200 / 500 = 0.40 = 40%
        assert group.pnl_pct == 0.40

    def test_is_credit_position(self):
        """Credit positions have negative total_debit."""
        credit_legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            )
        ]
        credit_group = PositionGroup.from_legs(credit_legs, group_id="credit_1")
        assert credit_group.is_credit_position() is True

        debit_legs = [
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00150000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 150.0,
                    "quantity": 1.0,
                    "avg_price": 5.0,
                    "market_price": 3.0,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            )
        ]
        debit_group = PositionGroup.from_legs(debit_legs, group_id="debit_1")
        assert debit_group.is_credit_position() is False

    def test_profit_factor_calculation(self):
        """Profit factor should be market_value / total_debit."""
        legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            )
        ]
        group = PositionGroup.from_legs(legs, group_id="test_1")
        # profit_factor = market_value / total_debit = -300 / -500 = 0.6
        assert group.profit_factor() == 0.6

    def test_days_to_expiry(self):
        """Days to expiry should be calculated from min expiry date."""
        from datetime import date, timedelta

        today = date.today()
        expiry_date = today + timedelta(days=30)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        legs = [
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00150000",
                    "expiry": expiry_str,
                    "right": "C",
                    "strike": 150.0,
                    "quantity": -1.0,
                    "avg_price": 5.0,
                    "market_price": 3.0,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            )
        ]
        group = PositionGroup.from_legs(legs, group_id="test_1")
        assert group.days_to_expiry() == 30


class TestPositionGroupCharacteristics:
    """Tests for PositionGroup position characteristics."""

    def test_min_max_expiry(self):
        """Min and max expiry should be calculated from legs."""
        from datetime import date, timedelta

        today = date.today()
        expiry_1 = (today + timedelta(days=10)).strftime("%Y-%m-%d")
        expiry_2 = (today + timedelta(days=30)).strftime("%Y-%m-%d")

        legs = [
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00150000",
                    "expiry": expiry_1,
                    "right": "C",
                    "strike": 150.0,
                    "quantity": -1.0,
                    "avg_price": 5.0,
                    "market_price": 3.0,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260427C00155000",
                    "expiry": expiry_2,
                    "right": "C",
                    "strike": 155.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
        ]
        group = PositionGroup.from_legs(legs, group_id="calendar_1")
        assert group.min_expiry == expiry_1
        assert group.max_expiry == expiry_2

    def test_min_max_strike(self):
        """Min and max strike should be calculated from legs."""
        legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00155000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 155.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
        ]
        group = PositionGroup.from_legs(legs, group_id="spread_1")
        assert group.min_strike == 150.0
        assert group.max_strike == 155.0

    def test_width_calculation(self):
        """Width should be max_strike - min_strike."""
        legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00155000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 155.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
        ]
        group = PositionGroup.from_legs(legs, group_id="spread_1")
        assert group.width == 5.0

    def test_leg_counts(self):
        """Leg counts should track calls and puts."""
        legs = [
            pd.Series(
                {
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
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327P00145000",
                    "expiry": "2026-03-27",
                    "right": "P",
                    "strike": 145.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
        ]
        group = PositionGroup.from_legs(legs, group_id="combo_1")
        assert group.leg_counts["calls"] == 1
        assert group.leg_counts["puts"] == 1


class TestPositionGroupStrategyInference:
    """Tests for strategy type inference."""

    def test_infer_vertical_spread(self):
        """Vertical spreads should be inferred from same expiry, different strikes."""
        from datetime import date, timedelta

        today = date.today()
        expiry_str = (today + timedelta(days=30)).strftime("%Y-%m-%d")

        legs = [
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00150000",
                    "expiry": expiry_str,
                    "right": "C",
                    "strike": 150.0,
                    "quantity": -1.0,
                    "avg_price": 5.0,
                    "market_price": 3.0,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00155000",
                    "expiry": expiry_str,
                    "right": "C",
                    "strike": 155.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
        ]
        group = PositionGroup.from_legs(legs, group_id="vertical_1")
        assert group.strategy_type == "vertical_spread"

    def test_infer_iron_condor(self):
        """Iron condors should be inferred from 4 legs with specific structure."""
        from datetime import date, timedelta

        today = date.today()
        expiry_str = (today + timedelta(days=30)).strftime("%Y-%m-%d")

        legs = [
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327P00145000",
                    "expiry": expiry_str,
                    "right": "P",
                    "strike": 145.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327P00150000",
                    "expiry": expiry_str,
                    "right": "P",
                    "strike": 150.0,
                    "quantity": -1.0,
                    "avg_price": 3.0,
                    "market_price": 2.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00155000",
                    "expiry": expiry_str,
                    "right": "C",
                    "strike": 155.0,
                    "quantity": -1.0,
                    "avg_price": 3.0,
                    "market_price": 2.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
            pd.Series(
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00160000",
                    "expiry": expiry_str,
                    "right": "C",
                    "strike": 160.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                }
            ),
        ]
        group = PositionGroup.from_legs(legs, group_id="condor_1")
        assert group.strategy_type == "iron_condor"


class TestRecommendation:
    """Tests for Recommendation enum."""

    def test_recommendation_values(self):
        """Recommendation enum should have expected values."""
        assert Recommendation.HOLD.value == "HOLD"
        assert Recommendation.SELL.value == "SELL"
        assert Recommendation.ROLL.value == "ROLL"
        assert Recommendation.CLOSE_HALF.value == "CLOSE_HALF"
        assert Recommendation.ADJUST.value == "ADJUST"


class TestPositionGrouper:
    """Tests for PositionGrouper class."""

    def test_grouper_groups_by_underlying_and_expiry(self):
        """Legs with same underlying and expiry should be grouped."""
        positions = pd.DataFrame(
            [
                {
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
                    "multiplier": 100.0,
                },
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00155000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 155.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                },
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327P00145000",
                    "expiry": "2026-03-27",
                    "right": "P",
                    "strike": 145.0,
                    "quantity": -1.0,
                    "avg_price": 3.0,
                    "market_price": 2.0,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                },
            ]
        )
        grouper = PositionGrouper()
        groups = grouper.group(positions)
        assert len(groups) == 1  # All AAPL same expiry should be one group
        assert groups[0].underlying == "AAPL"
        assert len(groups[0].legs) == 3

    def test_grouper_separates_different_underlyings(self):
        """Positions in different underlyings should be separate groups."""
        positions = pd.DataFrame(
            [
                {
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
                    "multiplier": 100.0,
                },
                {
                    "underlying": "TSLA",
                    "symbol": "TSLA260327C00200000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 200.0,
                    "quantity": -1.0,
                    "avg_price": 10.0,
                    "market_price": 8.0,
                    "sec_type": "OPT",
                    "strategy": None,
                    "multiplier": 100.0,
                },
            ]
        )
        grouper = PositionGrouper()
        groups = grouper.group(positions)
        assert len(groups) == 2

    def test_grouper_with_strategy_column(self):
        """Grouper should use strategy column when available."""
        positions = pd.DataFrame(
            [
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00150000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 150.0,
                    "quantity": -1.0,
                    "avg_price": 5.0,
                    "market_price": 3.0,
                    "sec_type": "OPT",
                    "strategy": "PutCreditSpread",
                    "multiplier": 100.0,
                },
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00155000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 155.0,
                    "quantity": 1.0,
                    "avg_price": 2.0,
                    "market_price": 1.5,
                    "sec_type": "OPT",
                    "strategy": "PutCreditSpread",
                    "multiplier": 100.0,
                },
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260417C00160000",
                    "expiry": "2026-04-17",
                    "right": "C",
                    "strike": 160.0,
                    "quantity": -1.0,
                    "avg_price": 4.0,
                    "market_price": 3.5,
                    "sec_type": "OPT",
                    "strategy": "CoveredCall",
                    "multiplier": 100.0,
                },
            ]
        )
        grouper = PositionGrouper(config=GrouperConfig(group_by_strategy_column=True))
        groups = grouper.group(positions)
        # Should have 2 groups: one for PutCreditSpread, one for CoveredCall
        assert len(groups) == 2

    def test_grouper_empty_positions(self):
        """Grouper should handle empty dataframe."""
        positions = pd.DataFrame()
        grouper = PositionGrouper()
        groups = grouper.group(positions)
        assert len(groups) == 0

    def test_grouper_config_defaults(self):
        """GrouperConfig should have correct defaults."""
        config = GrouperConfig()
        assert config.group_by_strategy_column is True
        assert config.max_days_expiry_difference == 2
        assert config.prefer_fewer_groups is True


class TestPositionEvaluator:
    """Tests for PositionEvaluator class."""

    def test_evaluator_hold_recommendation(self):
        """Hold when no rules triggered."""
        from datetime import date, timedelta

        # Use expiry > 14 days away to avoid roll trigger (roll_dte_min=14)
        # Using relative date ensures test remains valid as time passes
        expiry_date = date.today() + timedelta(days=30)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        group = PositionGroup.from_legs(
            [
                pd.Series(
                    {
                        "underlying": "AAPL",
                        "expiry": expiry_str,
                        "right": "C",
                        "strike": 150.0,
                        "quantity": -1.0,
                        "avg_price": 5.0,
                        "market_price": 4.0,
                        "sec_type": "OPT",
                        "multiplier": 100.0,
                        "delta": -0.3,
                        "gamma": 0.02,
                        "theta": 0.1,
                        "vega": 5.0,
                    }
                ),
            ],
            group_id="test",
        )
        evaluator = PositionEvaluator()
        results = evaluator.evaluate([group])
        assert len(results) == 1
        assert results[0].recommendation == Recommendation.HOLD

    def test_evaluator_sell_profit_target(self):
        """Sell when profit target reached (80%+)."""
        from datetime import date, timedelta

        # Credit spread: sold at 5.0, now worth 1.0 = 80% profit
        # Use relative date to ensure test remains valid as time passes
        expiry_date = date.today() + timedelta(days=30)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        group = PositionGroup.from_legs(
            [
                pd.Series(
                    {
                        "underlying": "AAPL",
                        "expiry": expiry_str,
                        "right": "C",
                        "strike": 150.0,
                        "quantity": -1.0,
                        "avg_price": 5.0,
                        "market_price": 1.0,
                        "sec_type": "OPT",
                        "multiplier": 100.0,
                        "delta": -0.1,
                        "gamma": 0.01,
                        "theta": 0.05,
                        "vega": 2.0,
                    }
                ),
            ],
            group_id="test",
        )
        evaluator = PositionEvaluator()
        results = evaluator.evaluate([group])
        assert len(results) == 1
        assert results[0].recommendation == Recommendation.SELL

    def test_evaluator_stop_loss(self):
        """Stop loss when loss exceeds 200%."""
        from datetime import date, timedelta

        # Credit position: sold at 5.0, now worth 15.0 (loss on short)
        # PnL = (5 - 15) * -1 * 100 = -1000 (loss)
        # PnL% = -1000 / 500 = -2.0 = -200%
        # Use relative date to ensure test remains valid as time passes
        expiry_date = date.today() + timedelta(days=30)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        group = PositionGroup.from_legs(
            [
                pd.Series(
                    {
                        "underlying": "AAPL",
                        "expiry": expiry_str,
                        "right": "C",
                        "strike": 150.0,
                        "quantity": -1.0,
                        "avg_price": 5.0,
                        "market_price": 15.0,
                        "sec_type": "OPT",
                        "multiplier": 100.0,
                        "delta": -0.5,
                        "gamma": 0.02,
                        "theta": 0.1,
                        "vega": 5.0,
                    }
                ),
            ],
            group_id="test",
        )
        evaluator = PositionEvaluator()
        results = evaluator.evaluate([group])
        assert len(results) == 1
        assert results[0].recommendation == Recommendation.SELL

    def test_evaluator_roll_near_expiry(self):
        """Roll when short option near expiry (<14 DTE)."""
        from datetime import date, timedelta

        # Position expiring in 10 days with short call
        expiry_date = date.today() + timedelta(days=10)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        group = PositionGroup.from_legs(
            [
                pd.Series(
                    {
                        "underlying": "AAPL",
                        "expiry": expiry_str,
                        "right": "C",
                        "strike": 150.0,
                        "quantity": -1.0,
                        "avg_price": 5.0,
                        "market_price": 4.0,
                        "sec_type": "OPT",
                        "multiplier": 100.0,
                        "delta": -0.3,
                        "gamma": 0.02,
                        "theta": 0.1,
                        "vega": 5.0,
                    }
                ),
            ],
            group_id="test",
        )
        evaluator = PositionEvaluator()
        results = evaluator.evaluate([group])
        assert len(results) == 1
        assert results[0].recommendation == Recommendation.ROLL

    def test_evaluator_result_contains_rationale(self):
        """Evaluation result should contain rationale."""
        from datetime import date, timedelta

        # Use relative date to ensure test remains valid as time passes
        expiry_date = date.today() + timedelta(days=30)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        group = PositionGroup.from_legs(
            [
                pd.Series(
                    {
                        "underlying": "AAPL",
                        "expiry": expiry_str,
                        "right": "C",
                        "strike": 150.0,
                        "quantity": -1.0,
                        "avg_price": 5.0,
                        "market_price": 1.0,
                        "sec_type": "OPT",
                        "multiplier": 100.0,
                        "delta": -0.1,
                        "gamma": 0.01,
                        "theta": 0.05,
                        "vega": 2.0,
                    }
                ),
            ],
            group_id="test",
        )
        evaluator = PositionEvaluator()
        results = evaluator.evaluate([group])
        assert len(results) == 1
        assert results[0].rationale
        assert "profit" in results[0].rationale.lower()

    def test_evaluator_config_custom_thresholds(self):
        """Evaluator should accept custom thresholds."""
        from datetime import date, timedelta

        # Use relative date to ensure test remains valid as time passes
        expiry_date = date.today() + timedelta(days=30)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        group = PositionGroup.from_legs(
            [
                pd.Series(
                    {
                        "underlying": "AAPL",
                        "expiry": expiry_str,
                        "right": "C",
                        "strike": 150.0,
                        "quantity": -1.0,
                        "avg_price": 5.0,
                        "market_price": 4.0,
                        "sec_type": "OPT",
                        "multiplier": 100.0,
                        "delta": -0.3,
                        "gamma": 0.02,
                        "theta": 0.1,
                        "vega": 5.0,
                    }
                ),
            ],
            group_id="test",
        )
        # Custom config with 20% take profit
        evaluator = PositionEvaluator(
            config={"take_profit_pct": 0.2, "stop_loss_pct": 2.0}
        )
        results = evaluator.evaluate([group])
        assert len(results) == 1
        # 20% profit should trigger sell with 20% threshold
        assert results[0].recommendation == Recommendation.SELL


class TestRollRecommender:
    """Tests for RollRecommender class."""

    def test_roll_recommender_near_expiry(self):
        """Roll recommendation when near expiry."""
        from datetime import date, timedelta

        # Create a position expiring in 5 days
        future_expiry = date.today() + timedelta(days=5)
        group = PositionGroup.from_legs([
            pd.Series({
                "underlying": "AAPL", "expiry": future_expiry.isoformat(), "right": "C",
                "strike": 150.0, "quantity": -1.0, "avg_price": 5.0,
                "market_price": 3.0, "sec_type": "OPT", "multiplier": 100.0,
                "delta": -0.4,
            }),
        ], group_id="test")

        recommender = RollRecommender()
        roll_rec = recommender.recommend_roll(group, current_dte=5)

        assert roll_rec.should_roll is True
        assert roll_rec.target_dte == 30  # Default target
        assert "expiry" in roll_rec.rationale.lower()

    def test_roll_recommender_elevated_delta(self):
        """Roll recommendation when short delta is elevated."""
        from datetime import date, timedelta

        # Use relative date to ensure test remains valid as time passes
        future_expiry = date.today() + timedelta(days=30)

        group = PositionGroup.from_legs([
            pd.Series({
                "underlying": "AAPL", "expiry": future_expiry.isoformat(), "right": "C",
                "strike": 150.0, "quantity": -1.0, "avg_price": 5.0,
                "market_price": 8.0, "sec_type": "OPT", "multiplier": 100.0,
                "delta": -0.55,  # Elevated delta
            }),
        ], group_id="test")

        recommender = RollRecommender()
        roll_rec = recommender.recommend_roll(group, current_dte=30)

        assert roll_rec.should_roll is True
        assert "delta" in roll_rec.rationale.lower()

    def test_roll_recommender_no_roll_needed(self):
        """No roll recommendation when position is fine."""
        from datetime import date, timedelta

        # Use relative date to ensure test remains valid as time passes
        future_expiry = date.today() + timedelta(days=30)

        group = PositionGroup.from_legs([
            pd.Series({
                "underlying": "AAPL", "expiry": future_expiry.isoformat(), "right": "C",
                "strike": 150.0, "quantity": -1.0, "avg_price": 5.0,
                "market_price": 4.0, "sec_type": "OPT", "multiplier": 100.0,
                "delta": -0.2,  # Low delta
            }),
        ], group_id="test")

        recommender = RollRecommender()
        roll_rec = recommender.recommend_roll(group, current_dte=30)

        assert roll_rec.should_roll is False
        assert roll_rec.rationale == "No roll triggers met"

    def test_roll_recommender_adjustment_type(self):
        """Roll recommendation should include adjustment type."""
        from datetime import date, timedelta

        # Both near expiry AND elevated delta = roll_out + roll_up
        future_expiry = date.today() + timedelta(days=5)
        group = PositionGroup.from_legs([
            pd.Series({
                "underlying": "AAPL", "expiry": future_expiry.isoformat(), "right": "C",
                "strike": 150.0, "quantity": -1.0, "avg_price": 5.0,
                "market_price": 10.0, "sec_type": "OPT", "multiplier": 100.0,
                "delta": -0.55,
            }),
        ], group_id="test")

        recommender = RollRecommender()
        roll_rec = recommender.recommend_roll(group, current_dte=5)

        assert roll_rec.should_roll is True
        assert roll_rec.adjustment_type is not None


class TestEvaluationResultFormatting:
    """Tests for evaluation result formatting for messages and CSV output."""

    def test_evaluation_result_to_dict(self):
        """to_dict() should convert EvaluationResult to dictionary for CSV."""
        result = EvaluationResult(
            group_id="test_1",
            underlying="AAPL",
            strategy_type="vertical_spread",
            recommendation=Recommendation.SELL,
            confidence=0.8,
            rationale="Profit target reached",
            suggested_action="Close AAPL for 80% profit",
            pnl_pct=80.0,
            days_to_expiry=15,
            net_delta=-0.2,
        )
        result_dict = result.to_dict()
        assert result_dict["group_id"] == "test_1"
        assert result_dict["underlying"] == "AAPL"
        assert result_dict["strategy_type"] == "vertical_spread"
        assert result_dict["recommendation"] == "SELL"
        assert result_dict["confidence"] == 0.8
        assert result_dict["rationale"] == "Profit target reached"
        assert result_dict["suggested_action"] == "Close AAPL for 80% profit"
        assert result_dict["pnl_pct"] == 80.0
        assert result_dict["days_to_expiry"] == 15
        assert result_dict["net_delta"] == -0.2

    def test_reporter_write_outputs_with_evaluations(self):
        """write_outputs() should optionally include evaluations in CSV."""
        import tempfile
        from pathlib import Path
        from optionscanner.portfolio.report import PortfolioReporter, ReporterConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReporterConfig(
                results_dir=Path(tmpdir),
                logs_dir=Path(tmpdir),
            )
            reporter = PortfolioReporter(config)

            # Create sample data
            positions = pd.DataFrame([
                {"symbol": "AAPL", "quantity": 100, "market_price": 150.0},
            ])
            greek_summary = pd.DataFrame([
                {"underlying": "AAPL", "delta": -0.3, "gamma": 0.02},
            ])
            evaluations = [
                EvaluationResult(
                    group_id="test_1",
                    underlying="AAPL",
                    strategy_type="vertical_spread",
                    recommendation=Recommendation.SELL,
                    confidence=0.8,
                    rationale="Profit target reached",
                    suggested_action="Close position",
                    pnl_pct=50.0,
                    days_to_expiry=15,
                    net_delta=-0.25,
                ),
            ]

            # Call write_outputs with evaluations
            csv_path, json_path, timestamp = reporter.write_outputs(
                positions, greek_summary, evaluations
            )

            # Verify main CSV exists
            assert csv_path.exists()
            assert "portfolio_summary" in csv_path.name

            # Verify evaluation CSV also created
            eval_csv_path = reporter._results_dir / f"position_evaluations_{timestamp}.csv"
            assert eval_csv_path.exists()

            # Verify main CSV contains evaluation data
            content = csv_path.read_text()
            assert "AAPL" in content
            assert "vertical_spread" in content
            assert "SELL" in content

    def test_manager_format_evaluations_for_message(self):
        """PortfolioManager should format evaluations for messages."""
        import tempfile
        from unittest.mock import Mock
        from pathlib import Path
        from optionscanner.portfolio.manager import PortfolioManager

        with tempfile.TemporaryDirectory() as tmpdir:
            ib_mock = Mock()
            # Mock config to use temp directory
            import yaml
            config_path = Path(tmpdir) / "risk.yaml"
            config_path.write_text(yaml.dump({
                "limits": {},
                "roll_rules": {},
                "results_dir": tmpdir,
                "logs_dir": tmpdir,
            }))

            manager = PortfolioManager(ib=ib_mock, config_path=str(config_path))

            # Test the formatting
            evaluations = [
                EvaluationResult(
                    group_id="test_1",
                    underlying="AAPL",
                    strategy_type="vertical_spread",
                    recommendation=Recommendation.SELL,
                    confidence=0.8,
                    rationale="Profit target reached",
                    suggested_action="Close for profit",
                    pnl_pct=80.0,
                    days_to_expiry=15,
                    net_delta=-0.2,
                ),
                EvaluationResult(
                    group_id="test_2",
                    underlying="TSLA",
                    strategy_type="single_leg",
                    recommendation=Recommendation.HOLD,
                    confidence=0.6,
                    rationale="No triggers",
                    suggested_action="Maintain position",
                    pnl_pct=10.0,
                    days_to_expiry=30,
                    net_delta=-0.15,
                ),
            ]

            # Verify _format_evaluations works
            lines = manager._format_evaluations(evaluations)

            assert isinstance(lines, list)
            assert len(lines) >= 2
            # Should contain underlying symbols
            assert any("AAPL" in line for line in lines)
            assert any("TSLA" in line for line in lines)
            # TSLA (0.8 confidence) should appear before AAPL (0.6 confidence)
            # Actually AAPL has 0.8, TSLA has 0.6, so AAPL first
            aapl_line = next(i for i, line in enumerate(lines) if "AAPL" in line)
            tsla_line = next(i for i, line in enumerate(lines) if "TSLA" in line)
            assert aapl_line < tsla_line  # AAPL (higher confidence) first
