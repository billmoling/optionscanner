"""Iron condor strategy implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, List

import pandas as pd

from .base import BaseOptionStrategy, TradeSignal


class IronCondorStrategy(BaseOptionStrategy):
    """Identifies neutral volatility setups for iron condors."""

    def __init__(
        self,
        target_prob_itm: float = 0.15,
        premium_threshold: float = 1.0,
        min_credit_pct: float = 0.01,
        max_expiries_per_symbol: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_prob_itm = target_prob_itm
        self.premium_threshold = premium_threshold
        self.min_credit_pct = min_credit_pct
        self.max_expiries_per_symbol = max_expiries_per_symbol

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        signals: List[TradeSignal] = []
        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue
            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None or underlying_price <= 0:
                continue
            chain = chain.sort_values("strike")
            expiry_groups = chain.groupby("expiry")
            processed_expiries = 0
            for expiry, subset in expiry_groups:
                if (expiry - datetime.utcnow()).days < 21:
                    continue
                if self.max_expiries_per_symbol and processed_expiries >= self.max_expiries_per_symbol:
                    break
                calls = subset[subset["option_type"] == "CALL"]
                puts = subset[subset["option_type"] == "PUT"]
                if calls.empty or puts.empty:
                    continue
                call_candidate = calls.iloc[int(len(calls) * (1 - self.target_prob_itm)) - 1]
                put_candidate = puts.iloc[int(len(puts) * self.target_prob_itm)]
                total_credit = call_candidate.get("bid", 0.0) + put_candidate.get("bid", 0.0)
                if total_credit < self.premium_threshold:
                    continue
                credit_pct = total_credit / underlying_price
                if credit_pct < self.min_credit_pct:
                    continue
                rationale = (
                    f"Iron condor credit {total_credit:.2f} with delta {call_candidate.get('delta', 0):.2f}/"
                    f"{put_candidate.get('delta', 0):.2f}"
                )
                signals.append(
                    self.emit_signal(
                        TradeSignal(
                            symbol=subset["symbol"].iloc[0],
                            expiry=expiry,
                            strike=float(call_candidate["strike"]),
                            option_type="CALL",
                            direction="SHORT_CONDOR_CALL",
                            rationale=rationale,
                        )
                    )
                )
                signals.append(
                    self.emit_signal(
                        TradeSignal(
                            symbol=subset["symbol"].iloc[0],
                            expiry=expiry,
                            strike=float(put_candidate["strike"]),
                            option_type="PUT",
                            direction="SHORT_CONDOR_PUT",
                            rationale=rationale,
                        )
                    )
                )
                processed_expiries += 1
        return signals

    def _to_dataframe(self, snapshot: Any) -> pd.DataFrame:
        if isinstance(snapshot, pd.DataFrame):
            df = snapshot.copy()
        elif hasattr(snapshot, "to_pandas"):
            df = snapshot.to_pandas()
        else:
            df = pd.DataFrame(self._snapshot_options(snapshot))
        if df.empty:
            return df
        if "expiry" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["expiry"]):
            df["expiry"] = pd.to_datetime(df["expiry"])
        symbol = self._snapshot_value(snapshot, "symbol")
        if "symbol" not in df.columns and symbol is not None:
            df["symbol"] = symbol
        return df
