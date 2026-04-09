# WheelStrategy Sell Put Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Cash-Secured Put scanner with Black-Scholes probability calculation and quantitative filtering.

**Architecture:** Four modular components: (1) Black-Scholes model for probability calculation, (2) composable filter pipeline, (3) Put scanner that applies filters to option chains, (4) WheelStrategy that integrates scanner with the existing strategy framework.

**Tech Stack:** Python 3.x, pandas, NumPy for math, scipy.stats for normal distribution, existing optionscanner framework (NautilusTrader-based strategies, IBKR data fetcher).

---

## File Structure

**New Files to Create:**
- `src/optionscanner/analytics/__init__.py` - Package init
- `src/optionscanner/analytics/bs_model.py` - Black-Scholes model
- `src/optionscanner/filters/__init__.py` - Package init
- `src/optionscanner/filters/options_filters.py` - Filter pipeline and implementations
- `src/optionscanner/scanners/__init__.py` - Package init
- `src/optionscanner/scanners/put_scanner.py` - Put scanner
- `src/optionscanner/strategies/strategy_wheel.py` - WheelStrategy implementation
- `tests/test_bs_model.py` - BSModel unit tests
- `tests/test_options_filters.py` - Filter unit tests
- `tests/test_put_scanner.py` - Scanner integration tests
- `tests/test_wheel_strategy.py` - Strategy tests

**Existing Files to Modify:**
- `src/optionscanner/runner.py` - Register WheelStrategy discovery
- `config.yaml` - Add WheelStrategy configuration section

---

## Task 1: Black-Scholes Model

**Files:**
- Create: `src/optionscanner/analytics/__init__.py`
- Create: `src/optionscanner/analytics/bs_model.py`
- Test: `tests/test_bs_model.py`

- [ ] **Step 1: Create analytics package init**

Create `src/optionscanner/analytics/__init__.py`:

```python
"""Analytics module for option pricing and probability calculations."""

from optionscanner.analytics.bs_model import BSModel

__all__ = ["BSModel"]
```

- [ ] **Step 2: Commit package init**

```bash
git add src/optionscanner/analytics/__init__.py
git commit -m "feat: add analytics package for Black-Scholes model"
```

- [ ] **Step 3: Write failing test for BSModel**

Create `tests/test_bs_model.py`:

```python
"""Unit tests for Black-Scholes option pricing model."""

import pytest
from datetime import datetime, timezone

from optionscanner.analytics.bs_model import BSModel


class TestBSModelCalculations:
    """Test Black-Scholes calculation accuracy."""
    
    def test_d1_d2_calculation(self):
        """Verify d1 and d2 calculation against known values."""
        model = BSModel(risk_free_rate=0.05)
        
        # Test case: S=100, K=100, T=1.0, sigma=0.2
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.2
        d1, d2 = model.calculate_d1_d2(S, K, T, sigma)
        
        # Expected values (from standard BS calculators)
        expected_d1 = pytest.approx(0.35, abs=0.01)
        expected_d2 = pytest.approx(0.15, abs=0.01)
        
        assert d1 == expected_d1, f"d1={d1}, expected ~0.35"
        assert d2 == expected_d2, f"d2={d2}, expected ~0.15"
    
    def test_put_option_price(self):
        """Verify put option price calculation."""
        model = BSModel(risk_free_rate=0.05)
        
        # ATM put: S=100, K=100, T=1.0, sigma=0.2
        S, K, T, sigma = 100.0, 100.0, 1.0, 0.2
        price = model.calculate_option_price(S, K, T, sigma, "PUT")
        
        # Expected: ~4.6 (standard BS calculator reference)
        assert price == pytest.approx(4.6, abs=0.2), f"Put price={price}, expected ~4.6"
    
    def test_otm_probability_for_put(self):
        """Verify OTM probability calculation for put options."""
        model = BSModel(risk_free_rate=0.05)
        
        # OTM put: S=100, K=90 (strike below spot), T=0.1, sigma=0.3
        S, K, T, sigma = 100.0, 90.0, 0.1, 0.3
        prob = model.calculate_otm_probability(S, K, T, sigma, "PUT")
        
        # High probability of expiring OTM (stock stays above 90)
        assert prob > 0.90, f"OTM prob={prob}, expected >0.90 for deep OTM put"
    
    def test_otm_probability_itm_put(self):
        """Verify low OTM probability for ITM puts."""
        model = BSModel(risk_free_rate=0.05)
        
        # ITM put: S=100, K=110 (strike above spot), T=0.1, sigma=0.3
        S, K, T, sigma = 100.0, 110.0, 0.1, 0.3
        prob = model.calculate_otm_probability(S, K, T, sigma, "PUT")
        
        # Low probability of expiring OTM (stock stays above 110)
        assert prob < 0.30, f"OTM prob={prob}, expected <0.30 for ITM put"
```

- [ ] **Step 4: Run test to verify it fails**

```bash
source .venv/bin/activate
python -m pytest tests/test_bs_model.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'optionscanner.analytics'"

- [ ] **Step 5: Implement BSModel class**

Create `src/optionscanner/analytics/bs_model.py`:

```python
"""Black-Scholes option pricing and probability calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from scipy.stats import norm


@dataclass(slots=True)
class BSModel:
    """Black-Scholes option pricing and probability calculation.
    
    Implements the Black-Scholes-Merton formula for European options,
    including theoretical price and probability of expiring OTM.
    """
    
    risk_free_rate: float = 0.05  # Annual risk-free rate
    
    def calculate_d1_d2(
        self,
        S: float,  # Underlying price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        sigma: float,  # Implied volatility
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters for Black-Scholes formula.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility (annualized)
        
        Returns:
            Tuple of (d1, d2) parameters
        """
        if T <= 0:
            # Handle expired options
            return (float('inf'), float('-inf')) if S > K else (float('-inf'), float('inf'))
        
        if sigma <= 0:
            # Handle zero volatility edge case
            return (float('inf'), float('-inf')) if S > K else (float('-inf'), float('inf'))
        
        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (self.risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        
        return d1, d2
    
    def calculate_option_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,  # "CALL" or "PUT"
    ) -> float:
        """Calculate theoretical option price using Black-Scholes formula.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility (annualized)
            option_type: "CALL" or "PUT"
        
        Returns:
            Theoretical option price
        """
        d1, d2 = self.calculate_d1_d2(S, K, T, sigma)
        
        if option_type.upper() == "CALL":
            # Call price: S * N(d1) - K * e^(-rT) * N(d2)
            return S * norm.cdf(d1) - K * math.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:
            # Put price: K * e^(-rT) * N(-d2) - S * N(-d1)
            return K * math.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def calculate_otm_probability(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """Calculate probability of option expiring out-of-the-money.
        
        For Put options: P(stock > strike) = N(d2)
        For Call options: P(stock < strike) = N(-d2)
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility (annualized)
            option_type: "CALL" or "PUT"
        
        Returns:
            Probability of expiring OTM (0.0 to 1.0)
        """
        d1, d2 = self.calculate_d1_d2(S, K, T, sigma)
        
        if option_type.upper() == "PUT":
            # Put expires OTM if stock > strike: P = N(d2)
            return norm.cdf(d2)
        else:
            # Call expires OTM if stock < strike: P = N(-d2)
            return norm.cdf(-d2)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
source .venv/bin/activate
python -m pytest tests/test_bs_model.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/optionscanner/analytics/bs_model.py tests/test_bs_model.py
git commit -m "feat: implement Black-Scholes model with OTM probability calculation

- BSModel class with d1/d2, option price, and OTM probability methods
- Uses scipy.stats.norm for cumulative normal distribution
- Unit tests verify calculation accuracy against known reference values
"
```

---

## Task 2: Options Filter Pipeline

**Files:**
- Create: `src/optionscanner/filters/__init__.py`
- Create: `src/optionscanner/filters/options_filters.py`
- Test: `tests/test_options_filters.py`

- [ ] **Step 1: Create filters package init**

Create `src/optionscanner/filters/__init__.py`:

```python
"""Options filter pipeline for screening trade candidates."""

from optionscanner.filters.options_filters import (
    OptionFilter,
    FilterResult,
    OptionFilterPipeline,
    IVRankFilter,
    VolumeFilter,
    AnnualizedROIFilter,
    OTMProbabilityFilter,
)

__all__ = [
    "OptionFilter",
    "FilterResult",
    "OptionFilterPipeline",
    "IVRankFilter",
    "VolumeFilter",
    "AnnualizedROIFilter",
    "OTMProbabilityFilter",
]
```

- [ ] **Step 2: Commit package init**

```bash
git add src/optionscanner/filters/__init__.py
git commit -m "feat: add filters package for options screening"
```

- [ ] **Step 3: Write failing tests for filters**

Create `tests/test_options_filters.py`:

```python
"""Unit tests for options filter pipeline."""

import pytest

from optionscanner.filters.options_filters import (
    FilterResult,
    OptionFilterPipeline,
    IVRankFilter,
    VolumeFilter,
    AnnualizedROIFilter,
    OTMProbabilityFilter,
)
from optionscanner.analytics.bs_model import BSModel


class TestIVRankFilter:
    """Test IV Rank filtering."""
    
    def test_passes_when_iv_rank_above_threshold(self):
        filter = IVRankFilter(min_iv_rank=0.30)
        option_data = {"iv_rank": 0.45}
        context = {}
        
        result = filter.check(option_data, context)
        
        assert result.passed is True
        assert "0.45" in result.reason
    
    def test_fails_when_iv_rank_below_threshold(self):
        filter = IVRankFilter(min_iv_rank=0.30)
        option_data = {"iv_rank": 0.15}
        context = {}
        
        result = filter.check(option_data, context)
        
        assert result.passed is False
        assert "0.15" in result.reason


class TestVolumeFilter:
    """Test volume and open interest filtering."""
    
    def test_passes_when_volume_and_oi_above_threshold(self):
        filter = VolumeFilter(min_volume=500, min_open_interest=1000)
        option_data = {"volume": 1000, "open_interest": 2000}
        context = {}
        
        result = filter.check(option_data, context)
        
        assert result.passed is True
    
    def test_fails_when_volume_below_threshold(self):
        filter = VolumeFilter(min_volume=500, min_open_interest=1000)
        option_data = {"volume": 100, "open_interest": 2000}
        context = {}
        
        result = filter.check(option_data, context)
        
        assert result.passed is False
    
    def test_fails_when_oi_below_threshold(self):
        filter = VolumeFilter(min_volume=500, min_open_interest=1000)
        option_data = {"volume": 1000, "open_interest": 500}
        context = {}
        
        result = filter.check(option_data, context)
        
        assert result.passed is False


class TestAnnualizedROIFilter:
    """Test annualized ROI filtering."""
    
    def test_passes_when_roi_above_threshold(self):
        filter = AnnualizedROIFilter(min_annualized_roi=0.30)
        option_data = {"annualized_roi": 0.50}
        context = {}
        
        result = filter.check(option_data, context)
        
        assert result.passed is True
    
    def test_fails_when_roi_below_threshold(self):
        filter = AnnualizedROIFilter(min_annualized_roi=0.30)
        option_data = {"annualized_roi": 0.15}
        context = {}
        
        result = filter.check(option_data, context)
        
        assert result.passed is False


class TestOTMProbabilityFilter:
    """Test OTM probability filtering."""
    
    def test_passes_when_probability_above_threshold(self):
        filter = OTMProbabilityFilter(min_otm_probability=0.60)
        option_data = {
            "strike": 95.0,
            "days_to_expiry": 10,
            "implied_volatility": 0.30,
        }
        context = {"underlying_price": 100.0}
        
        result = filter.check(option_data, context)
        
        # OTM put with strike 95, stock at 100 should have high OTM probability
        assert result.passed is True
    
    def test_fails_when_probability_below_threshold(self):
        filter = OTMProbabilityFilter(min_otm_probability=0.60)
        option_data = {
            "strike": 105.0,  # ITM put
            "days_to_expiry": 10,
            "implied_volatility": 0.30,
        }
        context = {"underlying_price": 100.0}
        
        result = filter.check(option_data, context)
        
        # ITM put should have low OTM probability
        assert result.passed is False


class TestOptionFilterPipeline:
    """Test filter pipeline composition."""
    
    def test_all_filters_must_pass(self):
        pipeline = OptionFilterPipeline(filters=[
            IVRankFilter(min_iv_rank=0.30),
            VolumeFilter(min_volume=500, min_open_interest=1000),
        ])
        
        # All pass
        option_data = {"iv_rank": 0.45, "volume": 1000, "open_interest": 2000}
        passed, reasons = pipeline.check_all(option_data, {})
        assert passed is True
        assert len(reasons) == 2
        
        # One fails
        option_data = {"iv_rank": 0.15, "volume": 1000, "open_interest": 2000}
        passed, reasons = pipeline.check_all(option_data, {})
        assert passed is False
        assert "0.15" in reasons[0]
```

- [ ] **Step 4: Run test to verify it fails**

```bash
source .venv/bin/activate
python -m pytest tests/test_options_filters.py -v
```

Expected: FAIL with ModuleNotFoundError

- [ ] **Step 5: Implement filter classes**

Create `src/optionscanner/filters/options_filters.py`:

```python
"""Options filter pipeline for screening trade candidates."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from optionscanner.analytics.bs_model import BSModel


@dataclass(slots=True)
class FilterResult:
    """Result of a single filter check."""
    
    passed: bool
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class OptionFilter(abc.ABC):
    """Abstract base class for option filters."""
    
    @abc.abstractmethod
    def check(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FilterResult:
        """Check if option passes the filter.
        
        Args:
            option_data: Option data including strike, volume, IV, etc.
            context: Filter context including underlying price
        
        Returns:
            FilterResult indicating pass/fail and reason
        """
        pass


@dataclass(slots=True)
class IVRankFilter(OptionFilter):
    """Filter based on IV Rank (percentile of current IV vs historical range)."""
    
    min_iv_rank: float = 0.30  # Minimum IV Rank threshold
    
    def check(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FilterResult:
        iv_rank = float(option_data.get("iv_rank", 0.0))
        
        if iv_rank >= self.min_iv_rank:
            return FilterResult(
                passed=True,
                reason=f"IV Rank {iv_rank:.1%} >= {self.min_iv_rank:.0%}",
                metrics={"iv_rank": iv_rank},
            )
        return FilterResult(
            passed=False,
            reason=f"IV Rank {iv_rank:.1%} < {self.min_iv_rank:.0%}",
            metrics={"iv_rank": iv_rank},
        )


@dataclass(slots=True)
class VolumeFilter(OptionFilter):
    """Filter based on trading volume and open interest for liquidity."""
    
    min_volume: int = 500
    min_open_interest: int = 1000
    
    def check(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FilterResult:
        volume = int(option_data.get("volume", 0))
        oi = int(option_data.get("open_interest", 0))
        
        if volume >= self.min_volume and oi >= self.min_open_interest:
            return FilterResult(
                passed=True,
                reason=f"Volume {volume} >= {self.min_volume}, OI {oi} >= {self.min_open_interest}",
                metrics={"volume": volume, "open_interest": oi},
            )
        return FilterResult(
            passed=False,
            reason=f"Liquidity too low (vol={volume}, oi={oi})",
            metrics={"volume": volume, "open_interest": oi},
        )


@dataclass(slots=True)
class AnnualizedROIFilter(OptionFilter):
    """Filter based on annualized return on capital."""
    
    min_annualized_roi: float = 0.30  # 30% annualized minimum
    
    def check(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FilterResult:
        annualized_roi = float(option_data.get("annualized_roi", 0.0))
        
        if annualized_roi >= self.min_annualized_roi:
            return FilterResult(
                passed=True,
                reason=f"Annualized ROI {annualized_roi:.1%} >= {self.min_annualized_roi:.0%}",
                metrics={"annualized_roi": annualized_roi},
            )
        return FilterResult(
            passed=False,
            reason=f"Annualized ROI {annualized_roi:.1%} < {self.min_annualized_roi:.0%}",
            metrics={"annualized_roi": annualized_roi},
        )


@dataclass(slots=True)
class OTMProbabilityFilter(OptionFilter):
    """Filter based on probability of expiring OTM (Black-Scholes)."""
    
    min_otm_probability: float = 0.60  # 60% minimum probability
    bs_model: BSModel = field(default_factory=lambda: BSModel(risk_free_rate=0.05))
    
    def check(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FilterResult:
        S = float(context.get("underlying_price", 0))
        K = float(option_data.get("strike", 0))
        T = float(option_data.get("days_to_expiry", 1)) / 365.0
        sigma = float(option_data.get("implied_volatility", 0.3))
        
        prob = self.bs_model.calculate_otm_probability(S, K, T, sigma, "PUT")
        
        if prob >= self.min_otm_probability:
            return FilterResult(
                passed=True,
                reason=f"OTM Probability {prob:.1%} >= {self.min_otm_probability:.0%}",
                metrics={"otm_probability": prob},
            )
        return FilterResult(
            passed=False,
            reason=f"OTM Probability {prob:.1%} < {self.min_otm_probability:.0%}",
            metrics={"otm_probability": prob},
        )


@dataclass(slots=True)
class OptionFilterPipeline:
    """Composes multiple filters into a sequential pipeline."""
    
    filters: List[OptionFilter]
    
    def add_filter(self, filter: OptionFilter) -> None:
        """Add a filter to the pipeline."""
        self.filters.append(filter)
    
    def check_all(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Execute all filters sequentially.
        
        Args:
            option_data: Option data including strike, volume, IV, etc.
            context: Filter context including underlying price
        
        Returns:
            Tuple of (passed_all, list_of_reasons_from_each_filter)
        """
        reasons = []
        for f in self.filters:
            result = f.check(option_data, context)
            reasons.append(result.reason)
            if not result.passed:
                return False, reasons
        return True, reasons
```

- [ ] **Step 6: Run test to verify it passes**

```bash
source .venv/bin/activate
python -m pytest tests/test_options_filters.py -v
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/optionscanner/filters/options_filters.py tests/test_options_filters.py
git commit -m "feat: implement options filter pipeline with 4 screening filters

- IVRankFilter: minimum IV percentile threshold
- VolumeFilter: minimum volume and open interest for liquidity
- AnnualizedROIFilter: minimum annualized return on capital
- OTMProbabilityFilter: Black-Scholes probability of expiring OTM
- OptionFilterPipeline: sequential composition with short-circuit
"
```

---

## Task 3: Put Scanner

**Files:**
- Create: `src/optionscanner/scanners/__init__.py`
- Create: `src/optionscanner/scanners/put_scanner.py`
- Test: `tests/test_put_scanner.py`

[... continuing in next message due to length ...]
```

Let me continue writing the complete plan file:
---

## Task 3: Put Scanner

**Files:**
- Create: `src/optionscanner/scanners/__init__.py`
- Create: `src/optionscanner/scanners/put_scanner.py`
- Test: `tests/test_put_scanner.py`

- [ ] **Step 1: Create scanners package init**

Create `src/optionscanner/scanners/__init__.py`:

```python
"""Scanner modules for finding trade opportunities."""

from optionscanner.scanners.put_scanner import PutScanner, PutScanResult

__all__ = ["PutScanner", "PutScanResult"]
```

- [ ] **Step 2: Commit package init**

```bash
git add src/optionscanner/scanners/__init__.py
git commit -m "feat: add scanners package for opportunity discovery"
```

- [ ] **Step 3: Write failing test for PutScanner**

Create `tests/test_put_scanner.py`:

```python
"""Integration tests for Put Scanner."""

import pytest
from datetime import datetime, timezone, timedelta

from optionscanner.scanners.put_scanner import PutScanner, PutScanResult
from optionscanner.filters.options_filters import (
    OptionFilterPipeline,
    IVRankFilter,
    VolumeFilter,
    AnnualizedROIFilter,
    OTMProbabilityFilter,
)
from optionscanner.option_data import OptionChainSnapshot


class TestPutScanner:
    """Test Put Scanner filtering and sorting."""
    
    def test_scan_returns_empty_for_no_puts(self):
        """Scanner returns empty when no put options available."""
        pipeline = OptionFilterPipeline(filters=[])
        scanner = PutScanner(filter_pipeline=pipeline)
        
        # Create snapshot with only calls
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=datetime.now(timezone.utc),
            options=[
                {"option_type": "CALL", "strike": 145.0, "expiry": datetime.now(timezone.utc) + timedelta(days=10)},
            ],
        )
        
        results = scanner.scan(snapshot)
        
        assert len(results) == 0
    
    def test_scan_filters_by_expiry_range(self):
        """Scanner filters options outside expiry range."""
        pipeline = OptionFilterPipeline(filters=[])
        scanner = PutScanner(
            filter_pipeline=pipeline,
            min_days_to_expiry=5,
            max_days_to_expiry=15,
        )
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[
                # Too soon (2 days)
                {"option_type": "PUT", "strike": 145.0, "expiry": now + timedelta(days=2), "bid": 0.5},
                # In range (10 days)
                {"option_type": "PUT", "strike": 145.0, "expiry": now + timedelta(days=10), "bid": 1.0},
                # Too late (30 days)
                {"option_type": "PUT", "strike": 145.0, "expiry": now + timedelta(days=30), "bid": 2.0},
            ],
        )
        
        results = scanner.scan(snapshot)
        
        assert len(results) == 1
        assert results[0].days_to_expiry == 10
    
    def test_scan_filters_otm_puts(self):
        """Scanner only considers OTM puts (strike < underlying)."""
        pipeline = OptionFilterPipeline(filters=[])
        scanner = PutScanner(
            filter_pipeline=pipeline,
            min_strike_range_pct=0.02,
            max_strike_range_pct=0.20,
        )
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[
                # ITM put (strike > underlying) - should be filtered
                {"option_type": "PUT", "strike": 155.0, "expiry": now + timedelta(days=10), "bid": 6.0},
                # OTM put at 145 (3.3% OTM) - should pass
                {"option_type": "PUT", "strike": 145.0, "expiry": now + timedelta(days=10), "bid": 1.0},
                # Too far OTM at 120 (20% OTM) - might be filtered
                {"option_type": "PUT", "strike": 120.0, "expiry": now + timedelta(days=10), "bid": 0.1},
            ],
        )
        
        results = scanner.scan(snapshot)
        
        # Only the 145 strike should pass (assuming 120 is outside max_strike_range_pct)
        assert len(results) >= 1
        assert results[0].strike == 145.0
    
    def test_scan_sorts_by_annualized_roi(self):
        """Scanner returns results sorted by annualized ROI descending."""
        pipeline = OptionFilterPipeline(filters=[])
        scanner = PutScanner(filter_pipeline=pipeline, max_strike_range_pct=0.30)
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[
                {"option_type": "PUT", "strike": 145.0, "expiry": now + timedelta(days=10), "bid": 0.5},
                {"option_type": "PUT", "strike": 140.0, "expiry": now + timedelta(days=10), "bid": 1.5},
                {"option_type": "PUT", "strike": 135.0, "expiry": now + timedelta(days=10), "bid": 2.0},
            ],
        )
        
        results = scanner.scan(snapshot)
        
        # Verify sorted by annualized_roi descending
        for i in range(len(results) - 1):
            assert results[i].annualized_roi >= results[i+1].annualized_roi
    
    def test_scan_result_contains_all_fields(self):
        """Verify PutScanResult has all required fields populated."""
        pipeline = OptionFilterPipeline(filters=[])
        scanner = PutScanner(filter_pipeline=pipeline, max_strike_range_pct=0.30)
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[{
                "option_type": "PUT",
                "strike": 145.0,
                "expiry": now + timedelta(days=10),
                "bid": 1.0,
                "iv_rank": 0.45,
                "volume": 1000,
                "open_interest": 2000,
                "implied_volatility": 0.30,
                "delta": -0.25,
            }],
        )
        
        results = scanner.scan(snapshot)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.symbol == "AAPL"
        assert result.strike == 145.0
        assert result.underlying_price == 150.0
        assert result.option_bid == 1.0
        assert result.iv_rank == 0.45
        assert result.volume == 1000
        assert result.open_interest == 2000
        assert result.days_to_expiry == 10
        assert result.annualized_roi > 0
        assert result.otm_probability > 0
        assert result.delta == -0.25
```

- [ ] **Step 4: Run test to verify it fails**

```bash
source .venv/bin/activate
python -m pytest tests/test_put_scanner.py -v
```

Expected: FAIL with ModuleNotFoundError

- [ ] **Step 5: Implement PutScanner class**

Create `src/optionscanner/scanners/put_scanner.py`:

```python
"""Put Scanner for finding Cash-Secured Put opportunities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from optionscanner.filters.options_filters import OptionFilterPipeline
from optionscanner.analytics.bs_model import BSModel
from optionscanner.option_data import OptionChainSnapshot


@dataclass(slots=True)
class PutScanResult:
    """Result of a Sell Put scan for a single option."""
    
    symbol: str
    expiry: datetime
    strike: float
    option_type: str
    
    # Market data
    underlying_price: float
    option_bid: float
    iv_rank: float
    volume: int
    open_interest: int
    
    # Calculated metrics
    days_to_expiry: int
    annualized_roi: float
    otm_probability: float
    delta: float
    
    # Filter status
    passed_filters: bool
    filter_reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output or further processing."""
        return {
            "symbol": self.symbol,
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type,
            "underlying_price": self.underlying_price,
            "option_bid": self.option_bid,
            "iv_rank": self.iv_rank,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "days_to_expiry": self.days_to_expiry,
            "annualized_roi": self.annualized_roi,
            "otm_probability": self.otm_probability,
            "delta": self.delta,
            "passed_filters": self.passed_filters,
            "filter_reasons": self.filter_reasons,
        }


@dataclass(slots=True)
class PutScanner:
    """Scans option chains for Cash-Secured Put opportunities.
    
    Applies a filter pipeline to find OTM puts with attractive
    risk/reward characteristics.
    """
    
    filter_pipeline: OptionFilterPipeline
    bs_model: BSModel = field(default_factory=lambda: BSModel(risk_free_rate=0.05))
    
    # Configuration parameters
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 15
    min_strike_range_pct: float = 0.05  # Strike within 5% below underlying
    max_strike_range_pct: float = 0.20  # Strike within 20% below underlying
    
    def scan(
        self,
        snapshot: OptionChainSnapshot,
    ) -> List[PutScanResult]:
        """Scan for Sell Put opportunities in an option chain.
        
        Args:
            snapshot: Option chain snapshot for a single symbol
        
        Returns:
            List of PutScanResult objects sorted by annualized ROI descending
        """
        results: List[PutScanResult] = []
        underlying_price = snapshot.underlying_price
        
        if underlying_price <= 0:
            return results
        
        # Filter Put options
        puts = [opt for opt in snapshot.options if opt.get("option_type") == "PUT"]
        
        for option in puts:
            # Parse expiry
            expiry_raw = option.get("expiry")
            if expiry_raw is None:
                continue
            
            expiry = pd.to_datetime(expiry_raw, utc=True) if not isinstance(expiry_raw, datetime) else expiry_raw
            if getattr(expiry, "tzinfo", None) is None:
                expiry = expiry.tz_localize(timezone.utc)
            else:
                expiry = expiry.tz_convert(timezone.utc)
            
            # Calculate days to expiry
            days_to_expiry = (expiry - datetime.now(timezone.utc)).days
            
            # Filter by expiry range
            if days_to_expiry < self.min_days_to_expiry or days_to_expiry > self.max_days_to_expiry:
                continue
            
            # Get strike and validate OTM
            strike = float(option.get("strike", 0))
            if strike <= 0:
                continue
            
            # OTM Put: strike < underlying price
            otm_pct = (underlying_price - strike) / underlying_price
            if otm_pct < self.min_strike_range_pct or otm_pct > self.max_strike_range_pct:
                continue
            
            # Get premium (prefer bid, fallback to mark)
            premium = float(option.get("bid", option.get("mark", 0.0)))
            if premium <= 0:
                continue
            
            # Calculate annualized ROI
            collateral = strike  # Cash-secured requires strike amount
            roi = premium / collateral
            annualized_roi = roi * (365.0 / max(days_to_expiry, 1))
            
            # Prepare filter context
            context = {
                "underlying_price": underlying_price,
                "symbol": snapshot.symbol,
            }
            
            # Update option data for filters
            option["days_to_expiry"] = days_to_expiry
            option["annualized_roi"] = annualized_roi
            
            # Execute filter pipeline
            passed, reasons = self.filter_pipeline.check_all(option, context)
            
            if not passed:
                continue
            
            # Calculate OTM probability
            T = days_to_expiry / 365.0
            sigma = float(option.get("implied_volatility", 0.3))
            otm_prob = self.bs_model.calculate_otm_probability(
                underlying_price, strike, T, sigma, "PUT"
            )
            
            results.append(PutScanResult(
                symbol=snapshot.symbol,
                expiry=expiry,
                strike=strike,
                option_type="PUT",
                underlying_price=underlying_price,
                option_bid=premium,
                iv_rank=float(option.get("iv_rank", 0.0)),
                volume=int(option.get("volume", 0)),
                open_interest=int(option.get("open_interest", 0)),
                days_to_expiry=days_to_expiry,
                annualized_roi=annualized_roi,
                otm_probability=otm_prob,
                delta=float(option.get("delta", 0.0)),
                passed_filters=True,
                filter_reasons=reasons,
            ))
        
        # Sort by annualized ROI descending
        results.sort(key=lambda r: r.annualized_roi, reverse=True)
        return results
```

- [ ] **Step 6: Run test to verify it passes**

```bash
source .venv/bin/activate
python -m pytest tests/test_put_scanner.py -v
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/optionscanner/scanners/put_scanner.py tests/test_put_scanner.py
git commit -m "feat: implement PutScanner for Cash-Secured Put opportunities

- PutScanResult dataclass with all scan metrics
- PutScanner with expiry/strike range filtering
- Annualized ROI calculation
- Black-Scholes OTM probability integration
- Results sorted by annualized ROI descending
"
```

---

## Task 4: WheelStrategy Implementation

**Files:**
- Create: `src/optionscanner/strategies/strategy_wheel.py`
- Test: `tests/test_wheel_strategy.py`

- [ ] **Step 1: Write failing test for WheelStrategy**

Create `tests/test_wheel_strategy.py`:

```python
"""Tests for WheelStrategy Cash-Secured Put implementation."""

import pytest
from datetime import datetime, timezone, timedelta

from optionscanner.strategies.strategy_wheel import WheelStrategy
from optionscanner.option_data import OptionChainSnapshot


class TestWheelStrategy:
    """Test WheelStrategy signal generation."""
    
    def test_generates_signals_for_qualifying_puts(self):
        """Strategy generates signals for puts passing all filters."""
        strategy = WheelStrategy(
            min_days_to_expiry=0,
            max_days_to_expiry=15,
            min_iv_rank=0.20,  # Lower threshold for testing
            min_volume=100,
            min_open_interest=500,
            min_annualized_roi=0.10,  # Lower threshold for testing
            min_otm_probability=0.50,  # Lower threshold for testing
            max_signals_per_symbol=3,
        )
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[{
                "option_type": "PUT",
                "strike": 145.0,
                "expiry": now + timedelta(days=10),
                "bid": 1.0,
                "iv_rank": 0.45,
                "volume": 1000,
                "open_interest": 2000,
                "implied_volatility": 0.30,
                "delta": -0.25,
            }],
        )
        
        signals = strategy.on_data([snapshot])
        
        assert len(signals) >= 1
        signal = signals[0]
        
        assert signal.symbol == "AAPL"
        assert signal.strike == 145.0
        assert signal.option_type == "PUT"
        assert signal.direction == "SHORT_PUT"
    
    def test_limits_signals_per_symbol(self):
        """Strategy respects max_signals_per_symbol limit."""
        strategy = WheelStrategy(
            min_days_to_expiry=0,
            max_days_to_expiry=15,
            min_iv_rank=0.20,
            min_volume=100,
            min_open_interest=500,
            min_annualized_roi=0.10,
            min_otm_probability=0.50,
            max_signals_per_symbol=2,
        )
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[
                {"option_type": "PUT", "strike": 145.0, "expiry": now + timedelta(days=10), "bid": 1.0, "iv_rank": 0.45, "volume": 1000, "open_interest": 2000, "implied_volatility": 0.30},
                {"option_type": "PUT", "strike": 140.0, "expiry": now + timedelta(days=10), "bid": 0.8, "iv_rank": 0.40, "volume": 800, "open_interest": 1500, "implied_volatility": 0.28},
                {"option_type": "PUT", "strike": 135.0, "expiry": now + timedelta(days=10), "bid": 0.5, "iv_rank": 0.35, "volume": 500, "open_interest": 1000, "implied_volatility": 0.25},
            ],
        )
        
        signals = strategy.on_data([snapshot])
        
        assert len(signals) <= 2  # max_signals_per_symbol=2
    
    def test_signal_rationale_contains_key_metrics(self):
        """Signal rationale includes IV Rank, Volume, ROI, OTM Probability."""
        strategy = WheelStrategy(
            min_days_to_expiry=0,
            max_days_to_expiry=15,
            min_iv_rank=0.20,
            min_volume=100,
            min_open_interest=500,
            min_annualized_roi=0.10,
            min_otm_probability=0.50,
        )
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[{
                "option_type": "PUT",
                "strike": 145.0,
                "expiry": now + timedelta(days=10),
                "bid": 1.0,
                "iv_rank": 0.45,
                "volume": 1000,
                "open_interest": 2000,
                "implied_volatility": 0.30,
            }],
        )
        
        signals = strategy.on_data([snapshot])
        
        assert len(signals) >= 1
        assert "IV Rank" in signals[0].rationale
        assert "Volume" in signals[0].rationale
        assert "Annualized ROI" in signals[0].rationale or "ROI" in signals[0].rationale
        assert "OTM Probability" in signals[0].rationale or "Prob" in signals[0].rationale
    
    def test_signal_has_risk_reward_ratio(self):
        """Signal includes risk_reward_ratio for ranking."""
        strategy = WheelStrategy(
            min_days_to_expiry=0,
            max_days_to_expiry=15,
            min_iv_rank=0.20,
            min_volume=100,
            min_open_interest=500,
            min_annualized_roi=0.10,
            min_otm_probability=0.50,
        )
        
        now = datetime.now(timezone.utc)
        snapshot = OptionChainSnapshot(
            symbol="AAPL",
            underlying_price=150.0,
            timestamp=now,
            options=[{
                "option_type": "PUT",
                "strike": 145.0,
                "expiry": now + timedelta(days=10),
                "bid": 1.0,
                "iv_rank": 0.45,
                "volume": 1000,
                "open_interest": 2000,
                "implied_volatility": 0.30,
            }],
        )
        
        signals = strategy.on_data([snapshot])
        
        assert len(signals) >= 1
        assert signals[0].risk_reward_ratio is not None
        assert signals[0].risk_reward_ratio > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
python -m pytest tests/test_wheel_strategy.py -v
```

Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement WheelStrategy class**

Create `src/optionscanner/strategies/strategy_wheel.py`:

```python
"""WheelStrategy - Cash-Secured Put implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List

from loguru import logger

from optionscanner.strategies.base import BaseOptionStrategy, SignalLeg, TradeSignal
from optionscanner.scanners.put_scanner import PutScanner, PutScanResult
from optionscanner.filters.options_filters import (
    OptionFilterPipeline,
    IVRankFilter,
    VolumeFilter,
    AnnualizedROIFilter,
    OTMProbabilityFilter,
)


class WheelStrategy(BaseOptionStrategy):
    """Cash-Secured Put strategy.
    
    Sells OTM put options to collect premium, aiming to earn
    time value decay without being assigned. Uses quantitative
    filters to select high-probability opportunities.
    """
    
    def __init__(
        self,
        min_days_to_expiry: int = 0,
        max_days_to_expiry: int = 15,
        min_iv_rank: float = 0.30,
        min_volume: int = 500,
        min_open_interest: int = 1000,
        min_annualized_roi: float = 0.30,
        min_otm_probability: float = 0.60,
        min_strike_range_pct: float = 0.05,
        max_strike_range_pct: float = 0.20,
        max_signals_per_symbol: int = 3,
        published_win_rate: float = 0.65,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        
        self.max_signals_per_symbol = max_signals_per_symbol
        self.published_win_rate = published_win_rate
        
        # Build filter pipeline
        filters = [
            IVRankFilter(min_iv_rank=min_iv_rank),
            VolumeFilter(min_volume=min_volume, min_open_interest=min_open_interest),
            AnnualizedROIFilter(min_annualized_roi=min_annualized_roi),
            OTMProbabilityFilter(min_otm_probability=min_otm_probability),
        ]
        self.filter_pipeline = OptionFilterPipeline(filters=filters)
        
        # Initialize scanner
        self.scanner = PutScanner(
            filter_pipeline=self.filter_pipeline,
            min_days_to_expiry=min_days_to_expiry,
            max_days_to_expiry=max_days_to_expiry,
            min_strike_range_pct=min_strike_range_pct,
            max_strike_range_pct=max_strike_range_pct,
        )
        
        logger.info(
            "WheelStrategy initialized | DTE={min_dte}-{max_dte} min_iv={min_iv} min_roi={min_roi}",
            min_dte=min_days_to_expiry,
            max_dte=max_days_to_expiry,
            min_iv=min_iv_rank,
            min_roi=min_annualized_roi,
        )
    
    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """Process option chain data and generate Sell Put signals.
        
        Args:
            data: Iterable of OptionChainSnapshot objects
        
        Returns:
            List of TradeSignal objects for qualifying Sell Put opportunities
        """
        signals: List[TradeSignal] = []
        
        for snapshot in data:
            # Scan for qualifying puts
            results = self.scanner.scan(snapshot)
            
            if not results:
                continue
            
            # Limit signals per symbol
            for result in results[: self.max_signals_per_symbol]:
                signal = self._build_signal(result)
                if signal:
                    signals.append(signal)
        
        logger.info(
            "WheelStrategy generated {count} signals | symbols={symbols}",
            count=len(signals),
            symbols=",".join(set(s.symbol for s in signals)) if signals else "none",
        )
        
        return signals
    
    def _build_signal(self, result: PutScanResult) -> TradeSignal:
        """Build TradeSignal from scan result.
        
        Args:
            result: PutScanResult from scanner
        
        Returns:
            TradeSignal ready for execution or ranking
        """
        rationale = (
            f"Cash-Secured Put: sell {result.strike:.2f}P exp {result.expiry.strftime('%Y-%m-%d')} "
            f"premium ${result.option_bid:.2f} | "
            f"IV Rank {result.iv_rank:.1%} | Volume {result.volume} | OI {result.open_interest} | "
            f"Annualized ROI {result.annualized_roi:.1%} | "
            f"OTM Probability {result.otm_probability:.1%} | "
            f"DTE {result.days_to_expiry}d"
        )
        
        return TradeSignal(
            symbol=result.symbol,
            expiry=result.expiry,
            strike=result.strike,
            option_type="PUT",
            direction="SHORT_PUT",
            rationale=rationale,
            risk_reward_ratio=result.annualized_roi,  # Use annualized ROI as R/R proxy
            max_profit=result.option_bid,
            max_loss=result.strike - result.option_bid,  # Strike - premium
            legs=(
                SignalLeg(
                    action="SELL",
                    option_type="PUT",
                    strike=result.strike,
                    expiry=result.expiry,
                    quantity=1,
                ),
            ),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source .venv/bin/activate
python -m pytest tests/test_wheel_strategy.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/optionscanner/strategies/strategy_wheel.py tests/test_wheel_strategy.py
git commit -m "feat: implement WheelStrategy for Cash-Secured Puts

- Integrates PutScanner with strategy framework
- Quantitative filter pipeline (IV, Volume, ROI, OTM Probability)
- Signal rationale includes all key metrics
- Respects max_signals_per_symbol limit
- Published win rate for signal ranking
"
```

---

## Task 5: Integration and Configuration

**Files to Modify:**
- `src/optionscanner/runner.py` - Strategy discovery
- `config.yaml` - WheelStrategy configuration
- `src/optionscanner/notifications/slack.py` - Enhanced notifications

- [ ] **Step 1: Add WheelStrategy to config.yaml**

Add to `config.yaml` under `strategies:` section:

```yaml
strategies:
  # ... existing strategies ...
  
  WheelStrategy:
    params:
      min_days_to_expiry: 0
      max_days_to_expiry: 15
      min_iv_rank: 0.30
      min_volume: 500
      min_open_interest: 1000
      min_annualized_roi: 0.30
      min_otm_probability: 0.60
      min_strike_range_pct: 0.05
      max_strike_range_pct: 0.20
      max_signals_per_symbol: 3
    published_win_rate: 0.65
```

- [ ] **Step 2: Verify strategy auto-discovery**

The existing `discover_strategies()` function in `main.py` should automatically find `strategy_wheel.py`. Verify by running:

```bash
source .venv/bin/activate
python -c "from optionscanner.main import discover_strategies; strategies = discover_strategies(); print([s.name for s in strategies])"
```

Expected: `WheelStrategy` appears in the list

- [ ] **Step 3: Commit configuration**

```bash
git add config.yaml
git commit -m "config: add WheelStrategy configuration section

- 0-15 day expiry range for ultra-short term premium collection
- IV Rank >= 30% for elevated premium
- Volume >= 500, OI >= 1000 for liquidity
- Annualized ROI >= 30% threshold
- OTM Probability >= 60% for high win rate
- Published win rate 65% for signal ranking
"
```

---

## Task 6: End-to-End Testing

**Files:**
- Modify: `tests/test_runner.py` (optional: add integration test)

- [ ] **Step 1: Run full test suite**

```bash
source .venv/bin/activate
python -m pytest tests/test_bs_model.py tests/test_options_filters.py tests/test_put_scanner.py tests/test_wheel_strategy.py -v
```

Expected: All tests PASS

- [ ] **Step 2: Run strategy discovery test**

```bash
source .venv/bin/activate
python -c "
from optionscanner.main import discover_strategies
from optionscanner.strategies.strategy_wheel import WheelStrategy

strategies = discover_strategies()
wheel = [s for s in strategies if isinstance(s, WheelStrategy)]
if wheel:
    print(f'WheelStrategy found: {wheel[0].name}')
    print(f'  max_signals_per_symbol: {wheel[0].max_signals_per_symbol}')
    print(f'  published_win_rate: {wheel[0].published_win_rate}')
else:
    print('ERROR: WheelStrategy not found!')
"
```

Expected: WheelStrategy found with correct parameters

- [ ] **Step 3: Commit after verification**

```bash
git commit --allow-empty -m "test: verify WheelStrategy end-to-end integration

- All unit tests passing
- Strategy auto-discovery working
- Configuration validated
"
```

---

## Summary

Total: 6 Tasks, ~25 steps

**Completed deliverables:**
1. Black-Scholes model with OTM probability calculation
2. Composable filter pipeline (IV Rank, Volume, ROI, OTM Probability)
3. Put Scanner with expiry/strike filtering
4. WheelStrategy integrated with strategy framework
5. Configuration and auto-discovery

**Next steps after implementation:**
- Run scanner with `--run-mode local --market-data FROZEN` to test
- Monitor Slack notifications for Sell Put opportunities
- Adjust filter thresholds in config.yaml based on results
