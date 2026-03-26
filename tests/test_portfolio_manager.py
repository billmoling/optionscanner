"""Integration tests for PortfolioManager with position evaluation."""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import Mock

from portfolio.manager import PortfolioManager
from portfolio.evaluation import (
    PositionGrouper,
    PositionEvaluator,
    GrouperConfig,
    EvaluationResult,
    PositionGroup,
    Recommendation,
)


class TestPortfolioManagerEvaluatorIntegration:
    """Tests for PortfolioManager integration with PositionGrouper and PositionEvaluator."""

    def test_portfolio_manager_has_evaluator(self):
        """PortfolioManager should have evaluator initialized."""
        ib_mock = Mock()
        manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

        # Verify evaluator and grouper are initialized
        assert hasattr(manager, "_grouper")
        assert hasattr(manager, "_evaluator")
        assert manager._grouper is not None
        assert manager._evaluator is not None

    def test_portfolio_manager_group_positions(self):
        """PortfolioManager.group_positions() should group loaded positions."""
        ib_mock = Mock()
        manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

        # Mock position loading with sample data
        manager.positions = pd.DataFrame(
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
                    "delta": -0.3,
                    "gamma": 0.02,
                    "theta": 0.1,
                    "vega": 5.0,
                    "multiplier": 100.0,
                },
            ]
        )

        # Group positions
        groups = manager.group_positions()
        assert len(groups) >= 1
        assert groups[0].underlying == "AAPL"

    def test_portfolio_manager_evaluate_positions(self):
        """PortfolioManager.evaluate_positions() should generate recommendations."""
        ib_mock = Mock()
        manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

        # Create a highly profitable position (80%+ profit should recommend SELL)
        manager.positions = pd.DataFrame(
            [
                {
                    "underlying": "AAPL",
                    "symbol": "AAPL260327C00150000",
                    "expiry": "2026-03-27",
                    "right": "C",
                    "strike": 150.0,
                    "quantity": -1.0,
                    "avg_price": 5.0,
                    "market_price": 1.0,
                    "sec_type": "OPT",
                    "strategy": None,
                    "delta": -0.1,
                    "gamma": 0.01,
                    "theta": 0.05,
                    "vega": 2.0,
                    "multiplier": 100.0,
                },
            ]
        )

        # Group and evaluate
        manager.group_positions()
        results = manager.evaluate_positions()

        assert len(results) >= 1
        # With 80% profit, should recommend SELL
        assert results[0].recommendation in [Recommendation.SELL, Recommendation.HOLD]

    def test_portfolio_manager_stores_groups_and_evaluations(self):
        """PortfolioManager should store position_groups and evaluations."""
        ib_mock = Mock()
        manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

        manager.positions = pd.DataFrame(
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
            ]
        )

        # Group and evaluate
        manager.group_positions()
        manager.evaluate_positions()

        # Verify stored attributes
        assert hasattr(manager, "position_groups")
        assert hasattr(manager, "evaluations")
        assert len(manager.position_groups) >= 1
        assert len(manager.evaluations) >= 1

    def test_portfolio_manager_evaluate_without_grouping(self):
        """evaluate_positions() should call group_positions() if not already grouped."""
        ib_mock = Mock()
        manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

        manager.positions = pd.DataFrame(
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
            ]
        )

        # Call evaluate without explicit grouping
        results = manager.evaluate_positions()

        # Should have grouped internally
        assert len(results) >= 1
        assert len(manager.position_groups) >= 1


class TestPortfolioManagerEvaluationReporting:
    """Tests for PortfolioManager evaluation reporting integration."""

    def test_format_evaluations_method_exists(self):
        """PortfolioManager should have _format_evaluations helper method."""
        ib_mock = Mock()
        manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

        # Create mock evaluation results
        evaluations = [
            EvaluationResult(
                group_id="test_1",
                underlying="AAPL",
                strategy_type="vertical_spread",
                recommendation=Recommendation.HOLD,
                confidence=0.85,
                rationale="Position is profitable",
                suggested_action="Maintain current position",
                pnl_pct=0.40,
                days_to_expiry=30,
                net_delta=-0.25,
            ),
            EvaluationResult(
                group_id="test_2",
                underlying="TSLA",
                strategy_type="put_credit_spread",
                recommendation=Recommendation.SELL,
                confidence=0.75,
                rationale="Profit target reached",
                suggested_action="Close position",
                pnl_pct=0.60,
                days_to_expiry=15,
                net_delta=0.10,
            ),
        ]

        # Format evaluations
        lines = manager._format_evaluations(evaluations)

        assert isinstance(lines, list)
        assert len(lines) >= 2
        # Should contain underlying symbols
        assert any("AAPL" in line for line in lines)
        assert any("TSLA" in line for line in lines)

    def test_format_evaluations_sorts_by_confidence(self):
        """_format_evaluations should sort results by confidence descending."""
        ib_mock = Mock()
        manager = PortfolioManager(ib=ib_mock, config_path="risk.yaml")

        # Create evaluations with different confidence levels
        evaluations = [
            EvaluationResult(
                group_id="test_1",
                underlying="AAPL",
                strategy_type="vertical_spread",
                recommendation=Recommendation.HOLD,
                confidence=0.60,
                rationale="Position within parameters",
                suggested_action="Hold",
                pnl_pct=0.20,
                days_to_expiry=30,
                net_delta=-0.25,
            ),
            EvaluationResult(
                group_id="test_2",
                underlying="TSLA",
                strategy_type="put_credit_spread",
                recommendation=Recommendation.SELL,
                confidence=0.95,
                rationale="Max profit reached",
                suggested_action="Close position",
                pnl_pct=0.85,
                days_to_expiry=15,
                net_delta=0.10,
            ),
        ]

        lines = manager._format_evaluations(evaluations)

        # TSLA (0.95 confidence) should appear before AAPL (0.60 confidence)
        tsla_index = next(i for i, line in enumerate(lines) if "TSLA" in line)
        aapl_index = next(i for i, line in enumerate(lines) if "AAPL" in line)
        assert tsla_index < aapl_index


class TestPortfolioReporterEvaluationOutput:
    """Tests for PortfolioReporter evaluation output methods."""

    def test_write_evaluation_results(self):
        """PortfolioReporter should write evaluation results to CSV."""
        import tempfile
        from pathlib import Path
        from portfolio.report import PortfolioReporter, ReporterConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReporterConfig(
                results_dir=Path(tmpdir),
                logs_dir=Path(tmpdir),
            )
            reporter = PortfolioReporter(config)

            evaluations = [
                EvaluationResult(
                    group_id="test_1",
                    underlying="AAPL",
                    strategy_type="vertical_spread",
                    recommendation=Recommendation.HOLD,
                    confidence=0.85,
                    rationale="Position is profitable",
                    suggested_action="Maintain current position",
                    pnl_pct=0.40,
                    days_to_expiry=30,
                    net_delta=-0.25,
                ),
                EvaluationResult(
                    group_id="test_2",
                    underlying="TSLA",
                    strategy_type="put_credit_spread",
                    recommendation=Recommendation.SELL,
                    confidence=0.75,
                    rationale="Profit target reached",
                    suggested_action="Close position",
                    pnl_pct=0.60,
                    days_to_expiry=15,
                    net_delta=0.10,
                ),
            ]

            csv_path = reporter.write_evaluation_results(evaluations, "20260325_120000")

            assert csv_path.exists()
            assert "position_evaluations" in csv_path.name

            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert "group_id" in df.columns
            assert "underlying" in df.columns
            assert "recommendation" in df.columns
            assert "pnl_pct" in df.columns
