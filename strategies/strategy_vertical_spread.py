"""Vertical spread option strategy implementation."""
from __future__ import annotations

from datetime import datetime, timezone
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

            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None:
                continue
            expiry_candidates = sorted(chain["expiry"].unique())
            for expiry in expiry_candidates:
                if (expiry - datetime.now(timezone.utc)).days < self.min_days_to_expiry:
                    continue
                subset = chain[chain["expiry"] == expiry]
                call_subset = subset[subset["option_type"] == "CALL"].sort_values("strike")
                put_subset = subset[subset["option_type"] == "PUT"].sort_values("strike")

                if not call_subset.empty:
                    atm_call = call_subset.iloc[
                        (call_subset["strike"] - underlying_price).abs().argsort()[:1]
                    ].iloc[0]
                    otm_calls = call_subset[
                        call_subset["strike"] >= atm_call["strike"] + self.spread_width
                    ].sort_values("strike")
                    if not otm_calls.empty:
                        short_call = otm_calls.iloc[0]
                        rationale = (
                            "Bull call debit spread: long "
                            f"{atm_call['strike']:.2f}C IV {atm_call['implied_volatility']:.2%}, short "
                            f"{short_call['strike']:.2f}C IV {short_call['implied_volatility']:.2%}"
                        )
                        signals.append(
                            self.emit_signal(
                                TradeSignal(
                                    symbol=subset["symbol"].iloc[0],
                                    expiry=expiry,
                                    strike=float(short_call["strike"]),
                                    option_type="CALL",
                                    direction="BULL_CALL_DEBIT_SPREAD",
                                    rationale=rationale,
                                )
                            )
                        )

                if not put_subset.empty:
                    atm_put = put_subset.iloc[
                        (put_subset["strike"] - underlying_price).abs().argsort()[:1]
                    ].iloc[0]
                    otm_puts = put_subset[
                        put_subset["strike"] <= atm_put["strike"] - self.spread_width
                    ].sort_values("strike", ascending=False)
                    if not otm_puts.empty:
                        short_put = otm_puts.iloc[0]
                        rationale = (
                            "Bear put debit spread: long "
                            f"{atm_put['strike']:.2f}P theta {atm_put['theta']:.4f}, short "
                            f"{short_put['strike']:.2f}P theta {short_put['theta']:.4f}"
                        )
                        signals.append(
                            self.emit_signal(
                                TradeSignal(
                                    symbol=subset["symbol"].iloc[0],
                                    expiry=expiry,
                                    strike=float(short_put["strike"]),
                                    option_type="PUT",
                                    direction="BEAR_PUT_DEBIT_SPREAD",
                                    rationale=rationale,
                                )
                            )
                        )
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
            df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
        symbol = self._snapshot_value(snapshot, "symbol")
        if "symbol" not in df.columns and symbol is not None:
            df["symbol"] = symbol
        underlying = self._snapshot_value(snapshot, "underlying_price")
        if "underlying_price" not in df.columns and underlying is not None:
            df["underlying_price"] = underlying
        return df
