"""Iron condor strategy implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from .base import BaseOptionStrategy, TradeSignal


class IronCondorStrategy(BaseOptionStrategy):
    """Identifies neutral volatility setups for iron condors."""

    def __init__(
        self,
        target_prob_itm: float = 0.15,  # delta/probability target for short legs (e.g., 15% ITM ≈ 0.15 delta)
        premium_threshold: float = 1.0,  # minimum absolute credit (per condor) worth trading, in dollars
        min_credit_pct: float = 0.01,  # minimum credit as % of underlying (guards against low premium vs. risk)
        max_expiries_per_symbol: int = 3,  # cap how many expirations per symbol we evaluate each run
        allowed_symbols: Optional[Iterable[str]] = None,  # whitelist underlyings (defaults to SPY/QQQ/IWM)
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_prob_itm = target_prob_itm
        self.premium_threshold = premium_threshold
        self.min_credit_pct = min_credit_pct
        self.max_expiries_per_symbol = max_expiries_per_symbol
        default_symbols: Set[str] = {"SPY", "QQQ", "IWM"}
        self.allowed_symbols = {symbol.upper() for symbol in (allowed_symbols or default_symbols)}

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        best_by_symbol: Dict[str, Dict[str, Any]] = {}
        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue
            symbol = str(chain["symbol"].iloc[0]).upper()
            if symbol not in self.allowed_symbols:
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
                condor = self._build_condor(subset["symbol"].iloc[0], expiry, calls, puts, underlying_price)
                if condor is None:
                    continue
                symbol = condor["symbol"]
                current_best = best_by_symbol.get(symbol)
                if current_best is None or condor["total_credit"] > current_best["total_credit"]:
                    best_by_symbol[symbol] = condor
                processed_expiries += 1
        signals: List[TradeSignal] = []
        for condor in best_by_symbol.values():
            rationale = condor["rationale"]
            expiry = condor["expiry"]
            symbol = condor["symbol"]

            def add_signal(option_row: pd.Series, direction: str) -> None:
                signals.append(
                    self.emit_signal(
                        TradeSignal(
                            symbol=symbol,
                            expiry=expiry,
                            strike=float(option_row["strike"]),
                            option_type=str(option_row["option_type"]),
                            direction=direction,
                            rationale=rationale,
                        )
                    )
                )

            add_signal(condor["short_call"], "SHORT_CONDOR_CALL")
            add_signal(condor["long_call"], "LONG_CONDOR_CALL")
            add_signal(condor["short_put"], "SHORT_CONDOR_PUT")
            add_signal(condor["long_put"], "LONG_CONDOR_PUT")
        return signals

    def _build_condor(
        self,
        symbol: str,
        expiry: pd.Timestamp,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        underlying_price: float,
    ) -> Optional[Dict[str, Any]]:
        calls = calls.sort_values("strike").reset_index(drop=True)
        puts = puts.sort_values("strike").reset_index(drop=True)

        if len(calls) < 2 or len(puts) < 2:
            return None

        call_index = max(int(len(calls) * (1 - self.target_prob_itm)) - 1, 0)
        if call_index >= len(calls) - 1:
            return None
        long_call_index = call_index + 1

        put_index = int(len(puts) * self.target_prob_itm)
        if put_index <= 0:
            put_index = 1
        if put_index >= len(puts):
            put_index = len(puts) - 1
        long_put_index = put_index - 1

        short_call = calls.iloc[call_index]
        long_call = calls.iloc[long_call_index]
        short_put = puts.iloc[put_index]
        long_put = puts.iloc[long_put_index]

        credit_call = self._price(short_call, "bid") - self._price(long_call, "ask")
        credit_put = self._price(short_put, "bid") - self._price(long_put, "ask")
        total_credit = credit_call + credit_put

        if total_credit < self.premium_threshold:
            return None
        credit_pct = total_credit / underlying_price
        if credit_pct < self.min_credit_pct:
            return None

        rationale = (
            f"Iron condor credit {total_credit:.2f} ({credit_pct:.2%}) | "
            f"call delta {short_call.get('delta', 0):.2f} / put delta {short_put.get('delta', 0):.2f} | "
            f"{expiry.date()} strikes {short_put['strike']}-{long_put['strike']} / "
            f"{short_call['strike']}-{long_call['strike']}"
        )

        return {
            "symbol": symbol,
            "expiry": expiry,
            "short_call": short_call,
            "long_call": long_call,
            "short_put": short_put,
            "long_put": long_put,
            "total_credit": float(total_credit),
            "rationale": rationale,
        }

    def _price(self, row: pd.Series, preference: str) -> float:
        fields = ("bid", "ask") if preference == "bid" else ("ask", "bid")
        candidates = list(fields) + ["mark", "price"]
        for field in candidates:
            if field in row and pd.notna(row[field]):
                value = float(row[field])
                if value > 0:
                    return value
        return 0.0

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
