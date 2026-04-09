"""Unit tests for options filter pipeline."""
from __future__ import annotations

import pandas as pd
import pytest

from optionscanner.filters.options_filters import (
    AnnualizedROIFilter,
    IVRankFilter,
    OptionFilterPipeline,
    OTMProbabilityFilter,
    VolumeFilter,
)


class TestIVRankFilter:
    """Tests for IV Rank filter."""

    def test_matches_when_iv_rank_above_threshold(self) -> None:
        """Filter passes when IV Rank meets minimum."""
        filter_ = IVRankFilter(min_iv_rank=0.30)
        row = pd.Series({"iv_rank": 0.45})
        assert filter_.matches(row) is True

    def test_rejects_when_iv_rank_below_threshold(self) -> None:
        """Filter fails when IV Rank below minimum."""
        filter_ = IVRankFilter(min_iv_rank=0.30)
        row = pd.Series({"iv_rank": 0.20})
        assert filter_.matches(row) is False

    def test_rejects_when_iv_rank_missing(self) -> None:
        """Filter fails when IV Rank not available."""
        filter_ = IVRankFilter(min_iv_rank=0.30)
        row = pd.Series({"volume": 1000})
        assert filter_.matches(row) is False

    def test_description(self) -> None:
        """Verify filter description."""
        filter_ = IVRankFilter(min_iv_rank=0.30)
        assert filter_.description() == "IV Rank >= 30%"


class TestVolumeFilter:
    """Tests for Volume filter."""

    def test_matches_when_volume_above_threshold(self) -> None:
        """Filter passes when volume meets minimum."""
        filter_ = VolumeFilter(min_volume=500)
        row = pd.Series({"volume": 1000})
        assert filter_.matches(row) is True

    def test_rejects_when_volume_below_threshold(self) -> None:
        """Filter fails when volume below minimum."""
        filter_ = VolumeFilter(min_volume=500)
        row = pd.Series({"volume": 200})
        assert filter_.matches(row) is False

    def test_rejects_when_volume_missing(self) -> None:
        """Filter fails when volume not available."""
        filter_ = VolumeFilter(min_volume=500)
        row = pd.Series({"iv_rank": 0.45})
        assert filter_.matches(row) is False

    def test_description(self) -> None:
        """Verify filter description."""
        filter_ = VolumeFilter(min_volume=500)
        assert filter_.description() == "Volume >= 500"


class TestAnnualizedROIFilter:
    """Tests for Annualized ROI filter."""

    def test_matches_when_roi_above_threshold(self) -> None:
        """Filter passes when ROI meets minimum."""
        filter_ = AnnualizedROIFilter(min_annualized_roi=0.30)
        row = pd.Series({"annualized_roi": 0.45})
        assert filter_.matches(row) is True

    def test_rejects_when_roi_below_threshold(self) -> None:
        """Filter fails when ROI below minimum."""
        filter_ = AnnualizedROIFilter(min_annualized_roi=0.30)
        row = pd.Series({"annualized_roi": 0.20})
        assert filter_.matches(row) is False

    def test_rejects_when_roi_missing(self) -> None:
        """Filter fails when ROI not available."""
        filter_ = AnnualizedROIFilter(min_annualized_roi=0.30)
        row = pd.Series({"volume": 1000})
        assert filter_.matches(row) is False

    def test_description(self) -> None:
        """Verify filter description."""
        filter_ = AnnualizedROIFilter(min_annualized_roi=0.30)
        assert filter_.description() == "Annualized ROI >= 30%"


class TestOTMProbabilityFilter:
    """Tests for OTM Probability filter."""

    def test_matches_when_probability_above_threshold(self) -> None:
        """Filter passes when OTM probability meets minimum."""
        filter_ = OTMProbabilityFilter(min_otm_probability=0.60, option_type="PUT")
        row = pd.Series({"otm_probability_put": 0.75})
        assert filter_.matches(row) is True

    def test_rejects_when_probability_below_threshold(self) -> None:
        """Filter fails when OTM probability below minimum."""
        filter_ = OTMProbabilityFilter(min_otm_probability=0.60, option_type="PUT")
        row = pd.Series({"otm_probability_put": 0.45})
        assert filter_.matches(row) is False

    def test_rejects_when_probability_missing(self) -> None:
        """Filter fails when OTM probability not available."""
        filter_ = OTMProbabilityFilter(min_otm_probability=0.60, option_type="PUT")
        row = pd.Series({"volume": 1000})
        assert filter_.matches(row) is False

    def test_description(self) -> None:
        """Verify filter description."""
        filter_ = OTMProbabilityFilter(min_otm_probability=0.60, option_type="PUT")
        assert filter_.description() == "PUT OTM Probability >= 60%"


class TestOptionFilterPipeline:
    """Tests for filter pipeline composition."""

    def test_empty_pipeline_matches_all(self) -> None:
        """Empty pipeline passes all rows."""
        pipeline = OptionFilterPipeline()
        row = pd.Series({"iv_rank": 0.1, "volume": 10})
        assert pipeline.matches(row) is True

    def test_all_filters_must_pass(self) -> None:
        """All filters must pass for pipeline to match."""
        pipeline = OptionFilterPipeline([
            IVRankFilter(min_iv_rank=0.30),
            VolumeFilter(min_volume=500),
        ])
        row = pd.Series({"iv_rank": 0.45, "volume": 1000})
        assert pipeline.matches(row) is True

        row_fail = pd.Series({"iv_rank": 0.45, "volume": 200})
        assert pipeline.matches(row_fail) is False

    def test_apply_filters_dataframe(self) -> None:
        """Pipeline applies filters to DataFrame."""
        pipeline = OptionFilterPipeline([
            IVRankFilter(min_iv_rank=0.30),
        ])
        df = pd.DataFrame([
            {"iv_rank": 0.45, "symbol": "AAPL"},
            {"iv_rank": 0.20, "symbol": "GOOGL"},
            {"iv_rank": 0.50, "symbol": "MSFT"},
        ])
        filtered = pipeline.apply(df)
        assert len(filtered) == 2
        assert list(filtered["symbol"]) == ["AAPL", "MSFT"]

    def test_description_combines_all_filters(self) -> None:
        """Pipeline description combines all filter descriptions."""
        pipeline = OptionFilterPipeline([
            IVRankFilter(min_iv_rank=0.30),
            VolumeFilter(min_volume=500),
        ])
        assert pipeline.description() == "IV Rank >= 30% AND Volume >= 500"

    def test_method_chaining(self) -> None:
        """Supports method chaining for adding filters."""
        pipeline = OptionFilterPipeline()
        result = pipeline.add_filter(IVRankFilter(min_iv_rank=0.30))
        assert result is pipeline
        assert len(pipeline.filters) == 1
