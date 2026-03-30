"""Utility helpers for building template-based trade explanations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from option_data import OptionChainSnapshot
from strategies.base import TradeSignal


@dataclass(slots=True)
class TemplateExplanationBuilder:
    """Generate deterministic explanations without calling external services."""

    bullish_templates: Sequence[str] = (
        "The setup anticipates strength in {symbol}, so the {direction} {option_type} looks to capture upside once the price clears the {strike:.2f} strike.",
        "A supportive backdrop suggests momentum could build above {strike:.2f}, making a {direction.lower()} on the {option_type.lower()} attractive for upside participation.",
    )
    bearish_templates: Sequence[str] = (
        "The thesis expects weakness in {symbol}; the {direction} {option_type} aims to profit if price slips beneath {strike:.2f}.",
        "Downside pressure is the primary risk in focus, so {direction.lower()} the {option_type.lower()} offers protection should price drop through {strike:.2f}.",
    )

    def build(self, signal: TradeSignal, snapshot: Optional[OptionChainSnapshot]) -> str:
        template = self._choose_template(signal.direction)
        base = template.format(
            symbol=signal.symbol,
            direction=signal.direction,
            option_type=signal.option_type,
            strike=signal.strike,
        )
        rationale_note = f" Rationale: {signal.rationale}." if signal.rationale else ""
        market_note = self._market_scenarios(signal, snapshot)
        return f"{base}{market_note}{rationale_note}".strip()

    def _choose_template(self, direction: str) -> str:
        direction_upper = (direction or "").upper()
        if any(keyword in direction_upper for keyword in ("CALL", "BULL", "LONG")):
            return self.bullish_templates[0]
        if any(keyword in direction_upper for keyword in ("PUT", "BEAR", "SHORT")):
            return self.bearish_templates[0]
        return "The strategy produced a {direction} idea on {symbol} around the {strike:.2f} level."

    def _market_scenarios(
        self,
        signal: TradeSignal,
        snapshot: Optional[OptionChainSnapshot],
    ) -> str:
        if snapshot is None or not snapshot.options:
            return ""
        price = snapshot.underlying_price
        upside = price * 1.02
        downside = price * 0.98
        return (
            f" If price rallies toward {upside:.2f}, the position should benefit as it moves deeper in-the-money."
            f" If price retreats toward {downside:.2f}, reassess the thesis because the option could lose premium."
        )


__all__ = ["TemplateExplanationBuilder"]
