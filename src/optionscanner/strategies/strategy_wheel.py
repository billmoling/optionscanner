"""Wheel Strategy implementation for Cash-Secured Put scanning."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List

from loguru import logger

from optionscanner.analytics.bs_model import BSModel
from optionscanner.filters.options_filters import (
    AnnualizedROIFilter,
    IVRankFilter,
    OptionFilterPipeline,
    OTMProbabilityFilter,
    VolumeFilter,
)
from optionscanner.scanners.put_scanner import PutScanResult, PutScanner
from optionscanner.strategies.base import BaseOptionStrategy, SignalLeg, TradeSignal


class WheelStrategy(BaseOptionStrategy):
    """Wheel Strategy: Cash-Secured Put scanner.

    Scans for attractive put-selling opportunities based on:
    - IV Rank >= 30% (high implied volatility)
    - Volume >= 500 (liquid options)
    - Annualized ROI >= 30% (attractive returns)
    - OTM Probability >= 60% (favorable win rate)

    When a put is assigned, the strategy would transition to holding stock
    and selling calls (not implemented in this scanner version).
    """

    def __init__(
        self,
        min_iv_rank: float = 0.30,
        min_volume: int = 500,
        min_annualized_roi: float = 0.30,
        min_otm_probability: float = 0.60,
        max_days_to_expiry: int = 15,
        min_days_to_expiry: int = 0,
        risk_free_rate: float = 0.05,
        **kwargs: Any,
    ) -> None:
        """Initialize the Wheel Strategy.

        Args:
            min_iv_rank: Minimum IV Rank threshold (default 30%)
            min_volume: Minimum daily volume (default 500)
            min_annualized_roi: Minimum annualized ROI (default 30%)
            min_otm_probability: Minimum OTM probability (default 60%)
            max_days_to_expiry: Maximum days to expiration (default 15)
            min_days_to_expiry: Minimum days to expiration (default 0)
            risk_free_rate: Risk-free rate for BS calculations
            **kwargs: Passed to BaseOptionStrategy
        """
        super().__init__(**kwargs)

        filter_pipeline = OptionFilterPipeline([
            IVRankFilter(min_iv_rank=min_iv_rank),
            VolumeFilter(min_volume=min_volume),
            AnnualizedROIFilter(min_annualized_roi=min_annualized_roi),
            OTMProbabilityFilter(min_otm_probability=min_otm_probability, option_type="PUT"),
        ])

        bs_model = BSModel(risk_free_rate=risk_free_rate)

        self.scanner = PutScanner(
            bs_model=bs_model,
            filter_pipeline=filter_pipeline,
            risk_free_rate=risk_free_rate,
        )
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """Process option chain snapshots and generate trade signals.

        Args:
            data: Iterable of OptionChainSnapshot objects

        Returns:
            List of TradeSignal objects for top CSP opportunities
        """
        signals: List[TradeSignal] = []
        now = datetime.now(timezone.utc)

        for snapshot in data:
            symbol = self._snapshot_value(snapshot, "symbol")
            if symbol is None:
                continue

            underlying_price = self._snapshot_value(snapshot, "underlying_price")
            if underlying_price is None:
                continue

            puts_df = self._extract_puts(snapshot)
            if puts_df.empty:
                continue

            puts_df = self._filter_by_expiry(puts_df, now)
            if puts_df.empty:
                continue

            results = self.scanner.scan(
                symbol=str(symbol),
                puts_df=puts_df,
                underlying_price=float(underlying_price),
                now=now,
            )

            for result in results[:1]:
                signal = self._result_to_signal(result)
                if signal:
                    signals.append(signal)
                    logger.info(
                        "Wheel Strategy signal | symbol={symbol} strike={strike} exp={exp} "
                        "roi={roi:.0%} otm_prob={prob:.0%} iv_rank={iv:.0%}",
                        symbol=result.symbol,
                        strike=result.strike,
                        exp=result.expiry.strftime("%Y-%m-%d"),
                        roi=result.annualized_roi,
                        prob=result.otm_probability,
                        iv=result.iv_rank,
                    )

        return signals

    def _extract_puts(self, snapshot: Any) -> "pd.DataFrame":
        """Extract put options from snapshot."""
        import pandas as pd

        options = self._snapshot_options(snapshot)
        if not options:
            return pd.DataFrame()

        df = pd.DataFrame(options)
        if df.empty:
            return df

        if "option_type" in df.columns:
            df["option_type"] = df["option_type"].str.upper()
            puts = df[df["option_type"] == "PUT"].copy()
        else:
            puts = df.copy()

        return puts

    def _filter_by_expiry(
        self,
        df: "pd.DataFrame",
        now: datetime,
    ) -> "pd.DataFrame":
        """Filter DataFrame by days to expiry range."""
        import pandas as pd

        if "expiry" not in df.columns:
            return df

        df = df.copy()
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
        df["days_to_expiry"] = (df["expiry"] - now).dt.days

        mask = (
            (df["days_to_expiry"] >= self.min_days_to_expiry) &
            (df["days_to_expiry"] <= self.max_days_to_expiry)
        )

        return df[mask]

    def _result_to_signal(self, result: PutScanResult) -> TradeSignal | None:
        """Convert PutScanResult to TradeSignal."""
        from pandas import Timestamp

        rationale = (
            f"Wheel Strategy CSP: {result.symbol} {result.strike:.0f}P "
            f"exp {result.expiry.strftime('%Y-%m-%d')} "
            f"credit {result.mid_price:.2f} "
            f"ROI {result.annualized_roi:.0%} "
            f"OTM prob {result.otm_probability:.0%} "
            f"IV Rank {result.iv_rank:.0%} "
            f"vol {result.volume}"
        )

        return TradeSignal(
            symbol=result.symbol,
            expiry=Timestamp(result.expiry),
            strike=result.strike,
            option_type="PUT",
            direction="SELL_PUT",
            rationale=rationale,
            risk_reward_ratio=result.annualized_roi,
            legs=(
                SignalLeg(
                    action="SELL",
                    option_type="PUT",
                    strike=result.strike,
                    expiry=Timestamp(result.expiry),
                ),
            ),
        )
