import unittest
from datetime import datetime, timedelta, timezone

from ai_agents import (
    GeminiClientError,
    SignalValidationAgent,
)
from option_data import OptionChainSnapshot
from strategies.base import TradeSignal


class DummyGeminiClient:
    def __init__(self, response: str | None = None, should_raise: bool = False) -> None:
        self.response = response or ""
        self.should_raise = should_raise
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.should_raise:
            raise GeminiClientError("boom")
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


def build_signal() -> TradeSignal:
    return TradeSignal(
        symbol="NVDA",
        expiry=datetime.now(timezone.utc) + timedelta(days=30),
        strike=500.0,
        option_type="CALL",
        direction="LONG_CALL",
        rationale="Breakout through resistance",
    )


def build_snapshot() -> OptionChainSnapshot:
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(days=30)
    return OptionChainSnapshot(
        symbol="NVDA",
        underlying_price=490.0,
        timestamp=now,
        options=[
            {
                "expiry": expiry,
                "strike": 500.0,
                "option_type": "CALL",
                "mark": 12.5,
                "delta": 0.55,
                "theta": -0.02,
                "implied_volatility": 0.45,
            },
            {
                "expiry": expiry,
                "strike": 480.0,
                "option_type": "PUT",
                "mark": 8.1,
                "delta": -0.48,
                "theta": -0.015,
                "implied_volatility": 0.5,
            },
        ],
    )


class SignalValidationAgentTests(unittest.TestCase):
    def test_validation_agent_uses_gemini(self) -> None:
        client = DummyGeminiClient(response="Validation guidance")
        agent = SignalValidationAgent(client=client)
        signal = build_signal()
        snapshot = build_snapshot()

        review = agent.review(signal, snapshot, [signal])

        self.assertEqual(review, "Validation guidance")
        self.assertIn(signal.symbol, client.last_user_prompt or "")

    def test_validation_agent_fallback_when_error(self) -> None:
        client = DummyGeminiClient(should_raise=True)
        agent = SignalValidationAgent(client=client)
        signal = build_signal()

        review = agent.review(signal, None, [])

        self.assertTrue(review)

    def test_validation_agent_can_disable_gemini(self) -> None:
        client = DummyGeminiClient(response="should not be used")
        agent = SignalValidationAgent(client=client, enable_gemini=False)
        signal = build_signal()

        review = agent.review(signal, None, [])

        self.assertTrue(review)
        self.assertIsNone(client.last_user_prompt)


if __name__ == "__main__":
    unittest.main()
