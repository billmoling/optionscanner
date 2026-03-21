"""Strategy playbook helpers for generating adjustment actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from .rules import RiskBreach


@dataclass(slots=True)
class PlaybookContext:
    """Provides context for generating playbook actions."""

    roll_rules: Dict[str, Dict[str, float]]


class PlaybookEngine:
    """Dispatches playbook logic per strategy type."""

    def __init__(self, context: PlaybookContext) -> None:
        self._context = context

    def generate(self, positions: pd.DataFrame, breaches: Iterable[RiskBreach]) -> List[str]:
        if positions.empty:
            return []
        actions: List[str] = []
        breach_by_symbol: Dict[str, List[RiskBreach]] = {}
        for breach in breaches:
            symbol = (breach.symbol or "portfolio").upper()
            breach_by_symbol.setdefault(symbol, []).append(breach)
        for strategy, group in positions.groupby("strategy"):
            if not strategy:
                continue
            strategy = str(strategy).lower()
            handler = getattr(self, f"_handle_{strategy}", None)
            if handler is None:
                continue
            symbol = str(group["underlying"].iloc[0])
            symbol_breaches = breach_by_symbol.get(symbol.upper(), [])
            actions.extend(handler(symbol, group, symbol_breaches))
        return actions

    def _handle_pmcc(
        self,
        symbol: str,
        positions: pd.DataFrame,
        breaches: Iterable[RiskBreach],
    ) -> List[str]:
        rules = self._context.roll_rules.get("pmcc", {})
        take_profit_pct = float(rules.get("take_profit_pct", 0.6))
        roll_delta = float(rules.get("roll_up_if_short_delta_gt", 0.45))
        roll_days = int(rules.get("roll_out_days", 21))
        short_legs = positions[pd.to_numeric(positions["quantity"], errors="coerce") < 0]
        if short_legs.empty:
            return []
        short = short_legs.iloc[0]
        realized_pct = float(short.get("pnl_pct", take_profit_pct))
        delta = abs(float(short.get("delta", 0.0) or 0.0))
        reason = []
        if realized_pct >= take_profit_pct:
            reason.append(f"{realized_pct:.0%} credit captured")
        for breach in breaches:
            if "vega" in breach.metric.lower():
                reason.append("vega limit breach")
                break
        if delta >= roll_delta:
            reason.append(f"short delta {delta:.2f}")
        if not reason:
            reason.append("maintain core collar")
        description = (
            f"PMCC {symbol}: close short {short.get('symbol')} (+{take_profit_pct:.0%} credit), "
            f"re-sell near-{int(roll_delta * 100)}Δ call {roll_days}DTE"
        )
        return [f"{description} ({'; '.join(reason)})"]

    def _handle_condor(
        self,
        symbol: str,
        positions: pd.DataFrame,
        breaches: Iterable[RiskBreach],
    ) -> List[str]:
        rules = self._context.roll_rules.get("condor", {})
        take_profit_pct = float(rules.get("take_profit_pct", 0.6))
        strikes = pd.to_numeric(positions.get("strike"), errors="coerce")
        width = float(strikes.max() - strikes.min()) if not strikes.empty else 0.0
        reason = [f"width {width:.0f}"] if width == width else []  # NaN guard
        for breach in breaches:
            if "gamma" in breach.metric.lower():
                reason.append("gamma elevated")
                break
        description = f"{symbol} Condor: harvest profits near {take_profit_pct:.0%} credit"
        if reason:
            description += f" ({'; '.join(reason)})"
        return [description]


__all__ = ["PlaybookEngine", "PlaybookContext"]
