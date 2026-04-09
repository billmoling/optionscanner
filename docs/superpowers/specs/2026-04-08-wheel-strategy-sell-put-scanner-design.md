# WheelStrategy - Sell Put Scanner Design Specification

**Date:** 2026-04-08
**Status:** Draft
**Author:** Claude Code (AI Assistant)

---

## Overview

This design specification describes the complete requirements for adding the **WheelStrategy** module to the optionscanner platform, specifically for scanning and trading **Cash-Secured Put (CSP)** opportunities. The strategy sells out-of-the-money put options to collect premiums, aiming to earn time value decay returns without being assigned.

### Design Goals

1. **Pure Sell Put Strategy** - Implement cash-secured put scanning without automatic stock assignment
2. **Ultra-Short Expiry** - Focus on 0-15 days to expiry options to maximize theta decay rate
3. **Quantitative Filtering** - Apply four-stage filtering: IV Rank, Volume, Annualized ROI, OTM Probability
4. **Probability Analysis** - Calculate probability of expiring OTM using Black-Scholes model
5. **Extensible Scanner Framework** - Modular design supporting future scanner strategies

### Non-Goals

- Full Wheel cycle (Sell Covered Call after stock assignment)
- Automatic assignment or stock purchase logic
- Modifying existing PutCreditSpreadStrategy
- Replacing IBKR data fetch layer

---

## Case Study: AI Options Scanning Best Practices

Based on the user's shared AI Sell Put scanning case, extract the following core features:

| Feature | Case Implementation | This Design |
|---------|--------------------|-------------|
| Watchlist Scanning | 12 stocks (NVDA, AAPL, SPY, TQQQ, SOXL, etc.) | Reuse config.yaml tickers |
| Expiry Coverage | 10 dates within 0-45 days | 0-15 days, all available expiries |
| IV Filter | IV ≥ 30% | IV Rank ≥ 30% |
| Volume Filter | Volume ≥ 500 contracts | Volume ≥ 500 AND OI ≥ 1000 |
| Annualized ROI | Target ≥ 30% | Annualized ROI ≥ 30% |
| Probability Calculation | Black-Scholes OTM Probability ≥ 60% | Fully adopted |
| Output Format | Excel 4 worksheets | CSV + Enhanced Slack notifications |

---

## Architecture

### Module Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    WheelStrategy Module                          │
├─────────────────────────────────────────────────────────────────┤
│  strategy_wheel.py          │  Core strategy implementation     │
│  scanners/put_scanner.py    │  Generic scanner framework        │
│  analytics/bs_model.py      │  Black-Scholes probability calc   │
│  filters/options_filters.py │  IV/Volume/ROI filters            │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │     Existing Framework        │
                │  option_data.py               │
                │  strategies/base.py           │
                │  signal_ranking.py            │
                └───────────────────────────────┘
```

### Dependency Order

```
1. analytics/bs_model.py      - Foundation math model, no internal deps
   ↓
2. filters/options_filters.py - Depends on bs_model for probability
   ↓
3. scanners/put_scanner.py    - Depends on filters for screening
   ↓
4. strategy_wheel.py          - Depends on scanner for signals
```

---

## Module 1: Black-Scholes Model (`analytics/bs_model.py`)

### Purpose

Implement the Black-Scholes option pricing model to calculate theoretical option prices and probability of expiring OTM.

### Components

#### BSModel Class

```python
@dataclass(slots=True)
class BSModel:
    """Black-Scholes option pricing and probability calculation."""
    
    risk_free_rate: float = 0.05  # Annual risk-free rate
    
    def calculate_d1_d2(
        self,
        S: float,  # Underlying price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        sigma: float,  # Implied volatility
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters."""
        
    def calculate_option_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,  # "CALL" or "PUT"
    ) -> float:
        """Calculate theoretical option price."""
        
    def calculate_otm_probability(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """
        Calculate probability of option expiring OTM.
        
        For Put options: P(stock > strike) = N(d2)
        For Call options: P(stock < strike) = N(-d2)
        """
```

### Configuration Parameters

```yaml
# config.yaml
black_scholes:
  risk_free_rate: 0.05  # 5% risk-free rate
  probability_threshold: 0.60  # OTM probability ≥ 60%
```

---

## Module 2: Options Filters (`filters/options_filters.py`)

### Purpose

Implement a composable option filter pipeline supporting flexible configuration of multiple screening criteria.

### Components

#### FilterResult Dataclass

```python
@dataclass(slots=True)
class FilterResult:
    """Filter result."""
    
    passed: bool
    reason: str
    metrics: Dict[str, Any]  # Contains IV, volume, ROI, probability, etc.
```

#### Filter Interface

```python
class OptionFilter(abc.ABC):
    """Base option filter class."""
    
    @abc.abstractmethod
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        """Check if option passes the filter."""
```

#### Concrete Filter Implementations

```python
@dataclass(slots=True)
class IVRankFilter(OptionFilter):
    """IV Rank filter."""
    min_iv_rank: float = 0.30  # ≥ 30%
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        iv_rank = option_data.get("iv_rank", 0.0)
        if iv_rank >= self.min_iv_rank:
            return FilterResult(passed=True, reason=f"IV Rank {iv_rank:.1%} >= {self.min_iv_rank:.0%}")
        return FilterResult(passed=False, reason=f"IV Rank {iv_rank:.1%} < {self.min_iv_rank:.0%}")


@dataclass(slots=True)
class VolumeFilter(OptionFilter):
    """Volume and open interest filter."""
    min_volume: int = 500
    min_open_interest: int = 1000
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        volume = option_data.get("volume", 0)
        oi = option_data.get("open_interest", 0)
        if volume >= self.min_volume and oi >= self.min_open_interest:
            return FilterResult(passed=True, reason=f"Volume {volume} >= {self.min_volume}, OI {oi} >= {self.min_open_interest}")
        return FilterResult(passed=False, reason=f"Liquidity too low (vol={volume}, oi={oi})")


@dataclass(slots=True)
class AnnualizedROIFilter(OptionFilter):
    """Annualized ROI filter."""
    min_annualized_roi: float = 0.30  # ≥ 30%
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        annualized_roi = option_data.get("annualized_roi", 0.0)
        if annualized_roi >= self.min_annualized_roi:
            return FilterResult(passed=True, reason=f"Annualized ROI {annualized_roi:.1%} >= {self.min_annualized_roi:.0%}")
        return FilterResult(passed=False, reason=f"Annualized ROI {annualized_roi:.1%} < {self.min_annualized_roi:.0%}")


@dataclass(slots=True)
class OTMProbabilityFilter(OptionFilter):
    """OTM probability filter (based on Black-Scholes)."""
    min_otm_probability: float = 0.60  # ≥ 60%
    bs_model: BSModel = field(default_factory=BSModel)
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        S = context["underlying_price"]
        K = option_data["strike"]
        T = option_data["days_to_expiry"] / 365.0
        sigma = option_data.get("implied_volatility", 0.3)
        
        prob = self.bs_model.calculate_otm_probability(S, K, T, sigma, "PUT")
        if prob >= self.min_otm_probability:
            return FilterResult(passed=True, reason=f"OTM Probability {prob:.1%} >= {self.min_otm_probability:.0%}")
        return FilterResult(passed=False, reason=f"OTM Probability {prob:.1%} < {self.min_otm_probability:.0%}")
```

#### Filter Pipeline

```python
@dataclass(slots=True)
class OptionFilterPipeline:
    """Filter pipeline that executes multiple filters sequentially."""
    
    filters: List[OptionFilter]
    
    def add_filter(self, filter: OptionFilter) -> None:
        self.filters.append(filter)
    
    def check_all(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Execute all filters.
        
        Returns:
            (passed, reasons): Whether all filters passed, and list of reasons from each filter
        """
        reasons = []
        for f in self.filters:
            result = f.check(option_data, context)
            reasons.append(result.reason)
            if not result.passed:
                return False, reasons
        return True, reasons
```

---

## Module 3: Put Scanner (`scanners/put_scanner.py`)

### Purpose

Scan all available option contracts, apply the filter pipeline, and return qualifying Sell Put opportunities.

### Components

#### PutScanResult Dataclass

```python
@dataclass(slots=True)
class PutScanResult:
    """Sell Put scan result."""
    
    symbol: str
    expiry: datetime
    strike: float
    option_type: str  # "PUT"
    
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
        """Convert to dict for output or further processing."""
```

#### PutScanner Class

```python
@dataclass(slots=True)
class PutScanner:
    """Sell Put opportunity scanner."""
    
    filter_pipeline: OptionFilterPipeline
    bs_model: BSModel = field(default_factory=BSModel)
    
    # Configuration parameters
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 15
    min_strike_range_pct: float = 0.05  # Strike must be within 5% below underlying
    max_strike_range_pct: float = 0.20  # Strike must be within 20% below underlying
    
    def scan(
        self,
        snapshot: OptionChainSnapshot,
    ) -> List[PutScanResult]:
        """
        Scan Sell Put opportunities for a single stock.
        
        Args:
            snapshot: Option chain snapshot
        
        Returns:
            List of qualifying Sell Put opportunities
        """
        results = []
        underlying_price = snapshot.underlying_price
        
        # Filter Put options
        puts = [opt for opt in snapshot.options if opt.get("option_type") == "PUT"]
        
        for option in puts:
            # Calculate days to expiry
            expiry = pd.to_datetime(option["expiry"], utc=True)
            days_to_expiry = (expiry - datetime.now(timezone.utc)).days
            
            # Skip options outside expiry range
            if days_to_expiry < self.min_days_to_expiry or days_to_expiry > self.max_days_to_expiry:
                continue
            
            # Calculate strike range (OTM Put: strike < underlying price)
            strike = float(option["strike"])
            otm_pct = (underlying_price - strike) / underlying_price
            if otm_pct < self.min_strike_range_pct or otm_pct > self.max_strike_range_pct:
                continue
            
            # Calculate annualized ROI
            premium = float(option.get("bid", option.get("mark", 0.0)))
            collateral = strike  # Cash-secured requires strike amount
            roi = premium / collateral if collateral > 0 else 0.0
            annualized_roi = roi * (365.0 / max(days_to_expiry, 1))
            
            # Prepare filter context
            context = {
                "underlying_price": underlying_price,
                "symbol": snapshot.symbol,
            }
            
            # Update option data
            option["days_to_expiry"] = days_to_expiry
            option["annualized_roi"] = annualized_roi
            
            # Execute filter pipeline
            passed, reasons = self.filter_pipeline.check_all(option, context)
            
            if passed:
                # Calculate OTM probability
                T = days_to_expiry / 365.0
                sigma = option.get("implied_volatility", 0.3)
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
                    iv_rank=option.get("iv_rank", 0.0),
                    volume=option.get("volume", 0),
                    open_interest=option.get("open_interest", 0),
                    days_to_expiry=days_to_expiry,
                    annualized_roi=annualized_roi,
                    otm_probability=otm_prob,
                    delta=option.get("delta", 0.0),
                    passed_filters=True,
                    filter_reasons=reasons,
                ))
        
        # Sort by annualized ROI descending
        results.sort(key=lambda r: r.annualized_roi, reverse=True)
        return results
```

---

## Module 4: WheelStrategy Implementation (`strategies/strategy_wheel.py`)

### Purpose

Integrate the Put Scanner into the strategy framework to generate executable TradeSignals.

### Configuration Parameters

```yaml
# config.yaml
strategies:
  WheelStrategy:
    params:
      # Expiry range
      min_days_to_expiry: 0
      max_days_to_expiry: 15
      
      # Screening criteria
      min_iv_rank: 0.30
      min_volume: 500
      min_open_interest: 1000
      min_annualized_roi: 0.30
      min_otm_probability: 0.60
      
      # Strike range
      min_strike_range_pct: 0.05
      max_strike_range_pct: 0.20
      
      # Other
      max_signals_per_symbol: 3  # Max 3 signals per symbol
      published_win_rate: 0.65
```

### Strategy Implementation

```python
from optionscanner.strategies.base import BaseOptionStrategy, TradeSignal, SignalLeg
from optionscanner.scanners.put_scanner import PutScanner, PutScanResult
from optionscanner.filters.options_filters import (
    OptionFilterPipeline,
    IVRankFilter,
    VolumeFilter,
    AnnualizedROIFilter,
    OTMProbabilityFilter,
)
from optionscanner.analytics.bs_model import BSModel


class WheelStrategy(BaseOptionStrategy):
    """Cash-Secured Put strategy.
    
    Sells OTM put options to collect premium, aiming to earn
    time value decay without being assigned.
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
        **kwargs,
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
    
    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """Process option chain data and generate Sell Put signals."""
        signals: List[TradeSignal] = []
        
        for snapshot in data:
            # Scan Sell Put opportunities
            results = self.scanner.scan(snapshot)
            
            # Limit signals per symbol
            for result in results[: self.max_signals_per_symbol]:
                signal = self._build_signal(result)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _build_signal(self, result: PutScanResult) -> TradeSignal:
        """Build TradeSignal."""
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

---

## Output and Notification Enhancements

### CSV Output Fields

New fields added to existing `signals_*.csv`:

| Field | Description |
|-------|-------------|
| `iv_rank` | IV percentile |
| `volume` | Volume |
| `open_interest` | Open interest |
| `annualized_roi` | Annualized return |
| `otm_probability` | OTM probability |
| `days_to_expiry` | Days to expiry |
| `option_bid` | Premium |

### Slack Notification Enhancement

Add `send_wheel_strategy_signals` method to `notifications/slack.py`:

```python
def send_wheel_strategy_signals(
    self,
    scan_results: List[PutScanResult],
    csv_path: Optional[Path] = None,
) -> None:
    """Send WheelStrategy Sell Put opportunities to Slack."""
    if not self.enabled or not scan_results:
        return
    
    lines = [f"*{self.title}* | Sell Put Opportunities | {timestamp}"]
    lines.append("")
    
    # Top 5 Recommendations
    lines.append("*Top 5 Recommendations:*")
    for idx, result in enumerate(scan_results[:5], start=1):
        lines.append(
            f"*{idx}. {result.symbol}* | Strike ${result.strike:.0f} | "
            f"DTE {result.days_to_expiry}d | "
            f"ROI {result.annualized_roi:.1%} | "
            f"Prob {result.otm_probability:.0%}"
        )
    
    if csv_path:
        lines.append("")
        lines.append(f"Full results: `{csv_path}`")
    
    message = "\n".join(lines)
    # Send Slack...
```

---

## Configuration Integration

### config.yaml New Configuration

```yaml
strategies:
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

# Black-Scholes configuration
black_scholes:
  risk_free_rate: 0.05
  probability_threshold: 0.60
```

---

## Testing Strategy

### Unit Tests

1. **BSModel Tests** (`tests/test_bs_model.py`)
   - Verify d1/d2 calculation
   - Verify option price calculation (compare against known reference values)
   - Verify OTM probability calculation

2. **Filter Tests** (`tests/test_options_filters.py`)
   - Boundary condition tests for each filter
   - Filter pipeline combination tests

3. **Scanner Tests** (`tests/test_put_scanner.py`)
   - Use mock option chain data
   - Verify filter result correctness

4. **Strategy Tests** (`tests/test_wheel_strategy.py`)
   - Integration test verifying complete flow

### Integration Tests

- Use IBKR paper account data
- Verify scan results under real market data

---

## Implementation Plan

### Phase 1: Foundation Model
1. Implement `analytics/bs_model.py`
2. Write unit tests

### Phase 2: Filters
1. Implement `filters/options_filters.py`
2. Write unit tests

### Phase 3: Scanner
1. Implement `scanners/put_scanner.py`
2. Write integration tests

### Phase 4: Strategy Integration
1. Implement `strategies/strategy_wheel.py`
2. Integrate into runner.py
3. End-to-end testing

### Phase 5: Output and Notifications
1. Enhance CSV output fields
2. Enhance Slack notifications
3. User acceptance testing

---

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| IV data unavailable | Cannot calculate IV Rank | Use historical volatility as proxy, or skip this filter |
| Black-Scholes calculation error | Inaccurate probability | Validate against known reference values, add error tolerance |
| Ultra-short term option liquidity | Cannot execute | Strict volume/OI filtering, set minimum liquidity threshold |
| Expired options (< 0 DTE) | Invalid signals | Strict filter days_to_expiry > 0 |

---

## Success Criteria

1. **Functional Completeness**
   - [ ] All 4 filters working correctly
   - [ ] Black-Scholes probability calculation error < 5%
   - [ ] Scan results sorted by annualized ROI

2. **Performance Metrics**
   - [ ] Single scan (12 stocks × 10 expiries) < 5 seconds
   - [ ] Memory usage < 100MB

3. **User Experience**
   - [ ] Slack notification shows Top 5 recommendations
   - [ ] CSV output contains all key fields
   - [ ] Configuration parameters adjustable in config.yaml

---

## Future Extensions

1. **Full Wheel Cycle** - Automatically switch to Covered Call mode after assignment
2. **Dynamic Adjustment** - Adjust filter thresholds based on market regime
3. **Backtesting Support** - Historical data backtesting framework
4. **Automated Trading** - Integrate with TradeExecutor for automatic order submission

---

## Appendix: Black-Scholes Formula Reference

### d1 and d2 Calculation

$$d1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d2 = d1 - \sigma\sqrt{T}$$

### Put Option Theoretical Price

$$P = K e^{-rT} N(-d2) - S N(-d1)$$

### Put Option OTM Probability

$$P(\text{OTM}) = P(S_T > K) = N(d2)$$

Where:
- $S$ = Underlying price
- $K$ = Strike price
- $r$ = Risk-free rate
- $\sigma$ = Implied volatility
- $T$ = Time to expiry (years)
- $N(x)$ = Standard normal cumulative distribution function
