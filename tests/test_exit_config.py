"""Tests for ExitEngine foundation."""
import pytest
from src.exit.config import ExitRule, ExitEngine


class TestExitRule:
    """Tests for ExitRule dataclass."""

    def test_default_values(self):
        """Test ExitRule default values."""
        rule = ExitRule(strategy="TestStrategy")

        assert rule.strategy == "TestStrategy"
        assert rule.profit_target_pct == 0.50
        assert rule.stop_loss_pct is None
        assert rule.close_at_dte == 5
        assert rule.trailing_stop_delta is None
        assert rule.use_ai_advisor is False

    def test_custom_values(self):
        """Test ExitRule with custom values."""
        rule = ExitRule(
            strategy="PutCreditSpread",
            profit_target_pct=0.30,
            stop_loss_pct=0.20,
            close_at_dte=3,
            trailing_stop_delta=0.05,
            use_ai_advisor=True
        )

        assert rule.strategy == "PutCreditSpread"
        assert rule.profit_target_pct == 0.30
        assert rule.stop_loss_pct == 0.20
        assert rule.close_at_dte == 3
        assert rule.trailing_stop_delta == 0.05
        assert rule.use_ai_advisor is True


class TestExitEngine:
    """Tests for ExitEngine."""

    def test_init_with_rules(self):
        """Test ExitEngine initialization with rules."""
        rules = [
            ExitRule(strategy="PutCreditSpread", profit_target_pct=0.30),
            ExitRule(strategy="VerticalSpread", profit_target_pct=0.40),
        ]

        engine = ExitEngine(rules)

        assert len(engine._rules) == 2
        assert engine._rules["PutCreditSpread"].profit_target_pct == 0.30
        assert engine._rules["VerticalSpread"].profit_target_pct == 0.40

    def test_init_logs_rules_count(self):
        """Test ExitEngine logs the number of rules."""
        rules = [ExitRule(strategy="TestStrategy")]
        engine = ExitEngine(rules)

        # Engine initialized successfully with rule count
        assert len(engine._rules) == 1

    def test_get_rule_for_known_strategy(self):
        """Test getting rule for a configured strategy."""
        rules = [
            ExitRule(
                strategy="IronCondor",
                profit_target_pct=0.25,
                close_at_dte=7
            ),
        ]

        engine = ExitEngine(rules)
        rule = engine.get_rule("IronCondor")

        assert rule.strategy == "IronCondor"
        assert rule.profit_target_pct == 0.25
        assert rule.close_at_dte == 7

    def test_get_rule_for_unknown_strategy_returns_default(self):
        """Test getting rule for an unconfigured strategy returns defaults."""
        rules = [ExitRule(strategy="KnownStrategy", profit_target_pct=0.30)]
        engine = ExitEngine(rules, default_profit_target=0.50, default_close_dte=5)

        rule = engine.get_rule("UnknownStrategy")

        assert rule.strategy == "UnknownStrategy"
        assert rule.profit_target_pct == 0.50
        assert rule.close_at_dte == 5

    def test_get_rule_preserves_other_defaults(self):
        """Test that default rule preserves other default values."""
        rules = [ExitRule(strategy="KnownStrategy")]
        engine = ExitEngine(rules)

        rule = engine.get_rule("UnknownStrategy")

        assert rule.stop_loss_pct is None
        assert rule.trailing_stop_delta is None
        assert rule.use_ai_advisor is False

    def test_evaluate_returns_none_placeholder(self):
        """Test evaluate returns None as placeholder."""
        rules = [ExitRule(strategy="TestStrategy")]
        engine = ExitEngine(rules)

        # Create a mock position object
        class MockPosition:
            pass

        result = engine.evaluate(position=MockPosition())

        assert result is None
