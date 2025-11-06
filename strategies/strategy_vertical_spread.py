"""Vertical spread option strategy implementation."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, List

import pandas as pd

from .base import BaseOptionStrategy, TradeSignal


class VerticalSpreadStrategy(BaseOptionStrategy):
    """Constructs bullish or bearish vertical spreads based on IV skew."""

    def __init__(self, spread_width: float = 5.0, min_days_to_expiry: int = 14, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spread_width = spread_width
        self.min_days_to_expiry = min_days_to_expiry

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        signals: List[TradeSignal] = []
        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue

            underlying_price = float(snapshot.get("underlying_price", chain["underlying_price"].iloc[0]))
            expiry_candidates = sorted(chain["expiry"].unique())
            for expiry in expiry_candidates:
                if (expiry - datetime.utcnow()).days < self.min_days_to_expiry:
                    continue
                subset = chain[chain["expiry"] == expiry]
                atm = subset.iloc[(subset["strike"] - underlying_price).abs().argsort()[:1]].iloc[0]
                otm = subset[(subset["strike"] >= atm["strike"] + self.spread_width) & (subset["option_type"] == "CALL")]
                if not otm.empty:
                    selected = otm.iloc[0]
                    rationale = (
                        f"Bull call spread targeting IV {atm['implied_volatility']:.2%} -> "
                        f"{selected['implied_volatility']:.2%}"
                    )
                    signals.append(
                        self.emit_signal(
                            TradeSignal(
                                symbol=subset["symbol"].iloc[0],
                                expiry=expiry,
                                strike=float(selected["strike"]),
                                option_type="CALL",
                                direction="LONG_SPREAD",
                                rationale=rationale,
                            )
                        )
                    )
                put_subset = subset[subset["option_type"] == "PUT"]
                otm_puts = put_subset[put_subset["strike"] <= atm["strike"] - self.spread_width]
                if not otm_puts.empty:
                    selected = otm_puts.iloc[-1]
                    rationale = (
                        f"Bear put spread for downside hedge with theta {selected['theta']:.4f}"
                    )
                    signals.append(
                        self.emit_signal(
                            TradeSignal(
                                symbol=subset["symbol"].iloc[0],
                                expiry=expiry,
                                strike=float(selected["strike"]),
                                option_type="PUT",
                                direction="LONG_SPREAD",
                                rationale=rationale,
                            )
                        )
                    )
        return signals

    def _to_dataframe(self, snapshot: Any) -> pd.DataFrame:
        if isinstance(snapshot, pd.DataFrame):
            return snapshot
        if hasattr(snapshot, "to_pandas"):
            df = snapshot.to_pandas()
        else:
            df = pd.DataFrame(snapshot.get("options", []))
        if df.empty:
            return df
        if "expiry" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["expiry"]):
            df["expiry"] = pd.to_datetime(df["expiry"])
        if "symbol" not in df.columns and "symbol" in snapshot:
            df["symbol"] = snapshot["symbol"]
        if "underlying_price" not in df.columns and "underlying_price" in snapshot:
            df["underlying_price"] = snapshot["underlying_price"]
        return df
