"""Composable filter pipeline for options scanning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol

import pandas as pd


class OptionFilter(ABC):
    """Abstract base class for option filters."""

    @abstractmethod
    def matches(self, row: pd.Series) -> bool:
        """Check if an option row matches the filter criteria.

        Args:
            row: A row from an options DataFrame

        Returns:
            True if the option passes the filter
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the filter."""
        pass


class IVRankFilter(OptionFilter):
    """Filter options based on IV Rank threshold."""

    def __init__(self, min_iv_rank: float = 0.30) -> None:
        """Initialize the IV Rank filter.

        Args:
            min_iv_rank: Minimum IV Rank required (0.0 to 1.0)
        """
        self.min_iv_rank = min_iv_rank

    def matches(self, row: pd.Series) -> bool:
        iv_rank = row.get("iv_rank")
        if iv_rank is None:
            return False
        try:
            return float(iv_rank) >= self.min_iv_rank
        except (TypeError, ValueError):
            return False

    def description(self) -> str:
        return f"IV Rank >= {self.min_iv_rank:.0%}"


class VolumeFilter(OptionFilter):
    """Filter options based on volume threshold."""

    def __init__(self, min_volume: int = 500) -> None:
        """Initialize the volume filter.

        Args:
            min_volume: Minimum daily volume required
        """
        self.min_volume = min_volume

    def matches(self, row: pd.Series) -> bool:
        volume = row.get("volume")
        if volume is None:
            return False
        try:
            return int(volume) >= self.min_volume
        except (TypeError, ValueError):
            return False

    def description(self) -> str:
        return f"Volume >= {self.min_volume}"


class AnnualizedROIFilter(OptionFilter):
    """Filter options based on annualized ROI threshold."""

    def __init__(self, min_annualized_roi: float = 0.30) -> None:
        """Initialize the annualized ROI filter.

        Args:
            min_annualized_roi: Minimum annualized ROI (0.0 to 1.0)
        """
        self.min_annualized_roi = min_annualized_roi

    def matches(self, row: pd.Series) -> bool:
        roi = row.get("annualized_roi")
        if roi is None:
            return False
        try:
            return float(roi) >= self.min_annualized_roi
        except (TypeError, ValueError):
            return False

    def description(self) -> str:
        return f"Annualized ROI >= {self.min_annualized_roi:.0%}"


class OTMProbabilityFilter(OptionFilter):
    """Filter options based on probability of expiring OTM."""

    def __init__(
        self,
        min_otm_probability: float = 0.60,
        option_type: Literal["CALL", "PUT"] = "PUT",
    ) -> None:
        """Initialize the OTM probability filter.

        Args:
            min_otm_probability: Minimum probability of expiring OTM (0.0 to 1.0)
            option_type: Option type for probability calculation
        """
        self.min_otm_probability = min_otm_probability
        self.option_type = option_type

    def matches(self, row: pd.Series) -> bool:
        prob_field = f"otm_probability_{self.option_type.lower()}"
        probability = row.get(prob_field)
        if probability is None:
            return False
        try:
            return float(probability) >= self.min_otm_probability
        except (TypeError, ValueError):
            return False

    def description(self) -> str:
        return f"{self.option_type} OTM Probability >= {self.min_otm_probability:.0%}"


class OptionFilterPipeline:
    """Composable pipeline of option filters."""

    def __init__(self, filters: list[OptionFilter] | None = None) -> None:
        """Initialize the filter pipeline.

        Args:
            filters: Optional list of filters to add
        """
        self._filters: list[OptionFilter] = []
        if filters:
            self._filters = list(filters)

    def add_filter(self, filter_: OptionFilter) -> "OptionFilterPipeline":
        """Add a filter to the pipeline.

        Args:
            filter_: Filter to add

        Returns:
            Self for method chaining
        """
        self._filters.append(filter_)
        return self

    def matches(self, row: pd.Series) -> bool:
        """Check if a row passes all filters in the pipeline.

        Args:
            row: A row from an options DataFrame

        Returns:
            True if the option passes all filters
        """
        return all(f.matches(row) for f in self._filters)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all filters to a DataFrame.

        Args:
            df: Options DataFrame

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        mask = df.apply(self.matches, axis=1)
        return df[mask]

    def description(self) -> str:
        """Return a description of all filters in the pipeline."""
        return " AND ".join(f.description() for f in self._filters)

    @property
    def filters(self) -> list[OptionFilter]:
        """Return the list of filters."""
        return self._filters.copy()
