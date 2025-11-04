"""Poor Man's Covered Call (PMCC) strategy implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, List, Optional, Tuple

import pandas as pd

from .base import BaseOptionStrategy, TradeSignal


class PoorMansCoveredCallStrategy(BaseOptionStrategy):
    """Identify Poor Man's Covered Call (PMCC) opportunities."""

    def __init__(
        self,
        leaps_min_days: int = 240,
        leaps_delta_threshold: float = 0.7,
        leaps_max_strike_pct: float = 0.9,
        max_leaps_extrinsic_pct: float = 0.35,
        short_min_days: int = 21,
        short_max_days: int = 60,
        short_otm_pct: float = 0.05,
        min_return_on_capital: float = 0.12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.leaps_min_days = leaps_min_days
        self.leaps_delta_threshold = leaps_delta_threshold
        self.leaps_max_strike_pct = leaps_max_strike_pct
        self.max_leaps_extrinsic_pct = max_leaps_extrinsic_pct
        self.short_min_days = short_min_days
        self.short_max_days = short_max_days
        self.short_otm_pct = short_otm_pct
        self.min_return_on_capital = min_return_on_capital

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        now = pd.Timestamp(datetime.utcnow())
        signals: List[TradeSignal] = []

        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue

            underlying_price = float(
                snapshot.get("underlying_price", chain["underlying_price"].iloc[0])
            )
            if underlying_price <= 0:
                continue

            prepared = chain.copy()
            required_cols = {"option_type", "expiry", "strike"}
            if not required_cols.issubset(prepared.columns):
                continue
            prepared["option_type"] = prepared["option_type"].str.upper()
            prepared["expiry"] = pd.to_datetime(prepared["expiry"])
            prepared["days_to_expiry"] = (prepared["expiry"] - now).dt.days

            calls = prepared[prepared["option_type"] == "CALL"].copy()
            if calls.empty:
                continue

            leaps_candidates = self._filter_leaps(calls, underlying_price)
            if leaps_candidates.empty:
                continue

            short_candidates = self._filter_short_calls(calls, underlying_price)
            if short_candidates.empty:
                continue

            for _, leaps in leaps_candidates.iterrows():
                best = self._select_short_for_leaps(leaps, short_candidates)
                if best is None:
                    continue
                short_call, net_debit, credit, roc = best
                rationale = (
                    "PMCC net debit {net:.2f} | credit {credit:.2f} | ROC {roc:.2%} "
                    "LEAPS {leaps_exp:%Y-%m-%d} {leaps_strike:.2f} vs short {short_exp:%Y-%m-%d} "
                    "{short_strike:.2f}"
                ).format(
                    net=net_debit,
                    credit=credit,
                    roc=roc,
                    leaps_exp=pd.Timestamp(leaps["expiry"]).to_pydatetime(),
                    leaps_strike=float(leaps["strike"]),
                    short_exp=pd.Timestamp(short_call["expiry"]).to_pydatetime(),
                    short_strike=float(short_call["strike"]),
                )

                leaps_expiry = pd.Timestamp(leaps["expiry"]).to_pydatetime()
                short_expiry = pd.Timestamp(short_call["expiry"]).to_pydatetime()
                symbol = str(leaps.get("symbol", short_call.get("symbol", "")))

                signals.append(
                    self.emit_signal(
                        TradeSignal(
                            symbol=symbol,
                            expiry=leaps_expiry,
                            strike=float(leaps["strike"]),
                            option_type="CALL",
                            direction="LONG_PMCC_LEAPS",
                            rationale=rationale,
                        )
                    )
                )
                signals.append(
                    self.emit_signal(
                        TradeSignal(
                            symbol=symbol,
                            expiry=short_expiry,
                            strike=float(short_call["strike"]),
                            option_type="CALL",
                            direction="SHORT_PMCC_CALL",
                            rationale=rationale,
                        )
                    )
                )
        return signals

    def _filter_leaps(self, calls: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
        leaps = calls[calls["days_to_expiry"] >= self.leaps_min_days].copy()
        if leaps.empty:
            return leaps
        leaps = leaps[leaps["strike"] <= underlying_price * self.leaps_max_strike_pct]
        if leaps.empty:
            return leaps
        if "delta" in leaps.columns:
            delta_series = leaps["delta"].fillna(0.0).abs()
            leaps = leaps[delta_series >= self.leaps_delta_threshold]
        leaps["price"] = leaps.apply(self._option_price_long, axis=1)
        leaps = leaps[leaps["price"] > 0]
        if leaps.empty:
            return leaps
        intrinsic = (underlying_price - leaps["strike"]).clip(lower=0.0)
        extrinsic = leaps["price"] - intrinsic
        extrinsic_pct = extrinsic / underlying_price
        leaps = leaps[extrinsic_pct <= self.max_leaps_extrinsic_pct]
        leaps = leaps.assign(intrinsic=intrinsic, extrinsic=extrinsic)
        return leaps

    def _filter_short_calls(self, calls: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
        shorts = calls[
            (calls["days_to_expiry"] >= self.short_min_days)
            & (calls["days_to_expiry"] <= self.short_max_days)
            & (calls["strike"] >= underlying_price * (1 + self.short_otm_pct))
        ].copy()
        if shorts.empty:
            return shorts
        shorts["price"] = shorts.apply(self._option_price_short, axis=1)
        shorts = shorts[shorts["price"] > 0]
        return shorts

    def _select_short_for_leaps(
        self,
        leaps: pd.Series,
        short_candidates: pd.DataFrame,
    ) -> Optional[Tuple[pd.Series, float, float, float]]:
        eligible = short_candidates[
            (short_candidates["expiry"] < leaps["expiry"])
            & (short_candidates["strike"] >= leaps["strike"])
        ]
        if eligible.empty:
            return None

        best: Optional[Tuple[pd.Series, float, float, float]] = None
        leaps_price = float(leaps.get("price", 0.0))
        for _, short in eligible.iterrows():
            credit = float(short.get("price", 0.0))
            if credit <= 0:
                continue
            net_debit = leaps_price - credit
            if net_debit <= 0:
                continue
            roc = credit / net_debit
            if roc < self.min_return_on_capital:
                continue
            if best is None or roc > best[3]:
                best = (short, net_debit, credit, roc)
        return best

    @staticmethod
    def _option_price_long(row: pd.Series) -> float:
        return PoorMansCoveredCallStrategy._option_price(row, prefer="ask")

    @staticmethod
    def _option_price_short(row: pd.Series) -> float:
        return PoorMansCoveredCallStrategy._option_price(row, prefer="bid")

    @staticmethod
    def _option_price(row: pd.Series, prefer: str = "bid") -> float:
        if prefer == "ask":
            priority = ("ask", "mark", "mid", "last", "bid")
        else:
            priority = ("bid", "mark", "mid", "last", "ask")
        for key in priority:
            if key in row:
                value = row.get(key, 0.0)
                if pd.notna(value) and float(value) > 0:
                    return float(value)
        return 0.0

    def _to_dataframe(self, snapshot: Any) -> pd.DataFrame:
        if isinstance(snapshot, pd.DataFrame):
            df = snapshot.copy()
        elif hasattr(snapshot, "to_pandas"):
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


__all__ = ["PoorMansCoveredCallStrategy"]
