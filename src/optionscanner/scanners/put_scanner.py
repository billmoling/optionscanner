"""Put option scanner for Cash-Secured Put opportunities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from optionscanner.analytics.bs_model import BSModel
from optionscanner.filters.options_filters import (
    AnnualizedROIFilter,
    IVRankFilter,
    OptionFilterPipeline,
    OTMProbabilityFilter,
    VolumeFilter,
)


@dataclass
class PutScanResult:
    """Result of a put option scan."""

    symbol: str
    strike: float
    expiry: pd.Timestamp
    bid: float
    ask: float
    last: float
    volume: int
    iv_rank: float
    days_to_expiry: int
    underlying_price: float
    otm_probability: float
    annualized_roi: float

    @property
    def mid_price(self) -> float:
        """Return mid price between bid and ask."""
        return (self.bid + self.ask) / 2


class PutScanner:
    """Scanner for Cash-Secured Put opportunities.

    Applies a filter pipeline to put option chains and calculates
    risk/reward metrics for each candidate.
    """

    def __init__(
        self,
        bs_model: BSModel | None = None,
        filter_pipeline: OptionFilterPipeline | None = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        """Initialize the put scanner.

        Args:
            bs_model: Black-Scholes model instance (created if not provided)
            filter_pipeline: Filter pipeline to apply (default Wheel Strategy filters)
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.bs_model = bs_model or BSModel(risk_free_rate=risk_free_rate)
        self.filter_pipeline = filter_pipeline or self._default_wheel_strategy_filters()
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def _default_wheel_strategy_filters() -> OptionFilterPipeline:
        """Create default filter pipeline for Wheel Strategy."""
        return OptionFilterPipeline([
            IVRankFilter(min_iv_rank=0.30),
            VolumeFilter(min_volume=500),
            AnnualizedROIFilter(min_annualized_roi=0.30),
            OTMProbabilityFilter(min_otm_probability=0.60, option_type="PUT"),
        ])

    def scan(
        self,
        symbol: str,
        puts_df: pd.DataFrame,
        underlying_price: float,
        now: datetime | None = None,
    ) -> list[PutScanResult]:
        """Scan put options for suitable Cash-Secured Put candidates.

        Args:
            symbol: Underlying symbol
            puts_df: DataFrame of put options with columns:
                     strike, bid, ask, last, volume, iv_rank, expiry
            underlying_price: Current underlying stock price
            now: Current datetime (uses UTC now if not provided)

        Returns:
            List of PutScanResult sorted by annualized_roi descending
        """
        if puts_df.empty:
            return []

        now = now or datetime.now(timezone.utc)
        results: list[PutScanResult] = []

        for _, row in puts_df.iterrows():
            strike = self._safe_float(row.get("strike"))
            if strike is None or strike <= 0:
                continue

            expiry = self._parse_expiry(row.get("expiry"))
            if expiry is None:
                continue

            days_to_expiry = max((expiry - now).days, 0)
            if days_to_expiry <= 0:
                continue

            time_to_expiry = days_to_expiry / 365.0

            bid = self._safe_float(row.get("bid"))
            ask = self._safe_float(row.get("ask"))
            if bid is None or ask is None or bid <= 0:
                continue

            mid_price = (bid + ask) / 2

            iv_rank = self._safe_float(row.get("iv_rank")) or 0.0
            volume = int(row.get("volume") or 0)

            sigma = self._estimate_volatility(iv_rank, mid_price, underlying_price, strike, time_to_expiry)

            otm_probability = self.bs_model.calculate_otm_probability(
                underlying_price, strike, time_to_expiry, sigma, "PUT"
            )

            annualized_roi = self._calculate_annualized_roi(
                strike=strike,
                premium=mid_price,
                days_to_expiry=days_to_expiry,
            )

            result = PutScanResult(
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                bid=bid,
                ask=ask,
                last=self._safe_float(row.get("last")) or mid_price,
                volume=volume,
                iv_rank=iv_rank,
                days_to_expiry=days_to_expiry,
                underlying_price=underlying_price,
                otm_probability=otm_probability,
                annualized_roi=annualized_roi,
            )

            row_for_filter = self._to_filter_row(result)
            if self.filter_pipeline.matches(row_for_filter):
                results.append(result)

        results.sort(key=lambda r: r.annualized_roi, reverse=True)
        return results

    def _to_filter_row(self, result: PutScanResult) -> pd.Series:
        """Convert PutScanResult to a row for filter matching."""
        return pd.Series({
            "iv_rank": result.iv_rank,
            "volume": result.volume,
            "annualized_roi": result.annualized_roi,
            "otm_probability_put": result.otm_probability,
        })

    def _calculate_annualized_roi(
        self,
        strike: float,
        premium: float,
        days_to_expiry: int,
    ) -> float:
        """Calculate annualized ROI for a cash-secured put.

        ROI = (Premium / Collateral) * (365 / Days to Expiry)

        For a CSP, collateral = strike (cash secured).

        Args:
            strike: Strike price (collateral required)
            premium: Premium received
            days_to_expiry: Days until expiration

        Returns:
            Annualized ROI as a decimal (e.g., 0.30 for 30%)
        """
        if strike <= 0 or days_to_expiry <= 0:
            return 0.0

        roi = premium / strike
        annualized = roi * (365.0 / days_to_expiry)
        return annualized

    @staticmethod
    def _estimate_volatility(
        iv_rank: float,
        option_price: float,
        underlying: float,
        strike: float,
        time_to_expiry: float,
    ) -> float:
        """Estimate implied volatility from IV Rank or option price.

        Uses IV Rank to estimate sigma if available, otherwise back-solves
        from option price.

        Args:
            iv_rank: IV Rank percentile (0.0 to 1.0)
            option_price: Option mid price
            underlying: Underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years

        Returns:
            Estimated volatility (sigma)
        """
        if iv_rank > 0:
            iv_rank_to_sigma = 0.20 + (iv_rank * 0.60)
            return iv_rank_to_sigma

        if time_to_expiry <= 0 or underlying <= 0:
            return 0.30

        return 0.30

    @staticmethod
    def _safe_float(value: object) -> float | None:
        """Safely convert a value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_expiry(value: object) -> pd.Timestamp | None:
        """Parse expiry value to pd.Timestamp."""
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            expiry = value
        else:
            try:
                expiry = pd.Timestamp(value)
            except (ValueError, TypeError):
                return None

        if getattr(expiry, "tzinfo", None) is None:
            expiry = expiry.tz_localize(timezone.utc)
        else:
            expiry = expiry.tz_convert(timezone.utc)

        return expiry
