# Signal Quality Enhancement System - Design Specification

**Date:** 2026-03-26
**Status:** Approved
**Author:** Claude Code (AI Assistant)

---

## Overview

This specification describes a modular enhancement system for improving signal quality and profitability in the optionscanner platform. The system is organized into four independent modules that can be developed, tested, and deployed separately.

### Goals

1. **Improve Entry Quality** - Filter signals through pattern recognition and confluence checks
2. **Optimize Exits** - Systematic profit-taking, time-based exits, and trailing stops
3. **Adapt to Market Regimes** - Dynamic parameter adjustment based on market conditions
4. **Enrich Data Foundation** - Add options flow, historical tracking, and fundamental data

### Non-Goals

- Replacing existing strategy implementations
- Changing the IBKR integration layer
- Modifying the NautilusTrader base strategy class
- Overhauling the Gemini AI selection system

---

## Architecture

### Module Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Signal Quality Modules                            │
├─────────────────┬──────────────────┬─────────────────┬─────────────────────┤
│   Entry Filter  │   Exit Engine    │  Regime Adapter │   Data Enhancer     │
│   (Module A)    │   (Module C)     │  (Module D)     │   (Module E)        │
├─────────────────┼──────────────────┼─────────────────┼─────────────────────┤
│ • PatternRecog  │ • DynamicTargets │ • RegimeConfig  │ • OptionsFlow       │
│ • ConfluenceCheck│ • TimeExits     │ • ParamOverride │ • HistoryStore      │
│ • StratSelector │ • TrailingStops  │ • StrategyGate  │ • FundamentalData   │
│                 │ • AIExitAdvisor  │ • TransitionDet │ • SimilarityMatcher │
└─────────────────┴──────────────────┴─────────────────┴─────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │     Core Framework            │
                    │  signal_ranking.py            │
                    │  strategies/base.py           │
                    │  market_state.py              │
                    │  portfolio/rules.py           │
                    └───────────────────────────────┘
```

### Dependency Order

```
Module E (Data Enhancer) - Foundation layer, no internal dependencies
    ↓
Module D (Regime Adapter) - Depends on Module E for historical context
    ↓
Module A (Entry Filter) - Depends on Module D for regime-aware gating
    ↓
Module C (Exit Engine) - Depends on Module E for historical calibration
```

---

## Module E: Data Enhancer

### Purpose

Enrich the signal generation pipeline with additional data sources for better decision-making.

### Components

#### E.1: OptionsFlowFetcher (`data/flow.py`)

```python
@dataclass(slots=True)
class FlowAlert:
    symbol: str
    timestamp: datetime
    volume: int
    open_interest: int
    volume_oi_ratio: float
    premium: float
    side: str  # "BUY" or "SELL"
    sweep_detected: bool

class OptionsFlowFetcher:
    def __init__(self, ibkr_client: IBKRDataFetcher)
    def fetch_unusual_activity(self, symbols: List[str]) -> Dict[str, List[FlowAlert]]
    def compute_flow_score(self, symbol: str, alerts: List[FlowAlert]) -> float
```

**Data Sources:**
- IBKR option chain volume and open interest
- Volume/OI ratio > 2.0 signals unusual activity
- Sweep detection via multi-leg, multi-exchange patterns

---

#### E.2: HistoryStore (`data/history.py`)

```python
@dataclass(slots=True)
class SignalOutcome:
    signal_id: str
    strategy: str
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    max_profit: Optional[float]
    max_loss: Optional[float]
    outcome: str  # "WIN", "LOSS", "OPEN"

class HistoryStore:
    def __init__(self, data_dir: Path)
    def record_signal(self, signal_id: str, signal: TradeSignal, entry_price: float)
    def record_exit(self, signal_id: str, exit_price: float, exit_date: datetime)
    def get_strategy_stats(self, strategy: str, window_days: int = 90) -> StrategyStats
    def get_similar_signals(self, signal: TradeSignal, top_k: int = 10) -> List[SignalOutcome]
```

**Storage Format:**
- JSONL format in `data/signal_history.jsonl`
- Indexed by strategy, symbol, and date

---

#### E.3: FundamentalFetcher (`data/fundamental.py`)

```python
@dataclass(slots=True)
class EarningsData:
    symbol: str
    report_date: datetime
    eps_actual: Optional[float]
    eps_estimate: Optional[float]
    eps_surprise_pct: Optional[float]
    revenue_actual: Optional[float]
    revenue_estimate: Optional[float]
    guidance_direction: Optional[str]  # "UP", "DOWN", "NEUTRAL"

class FundamentalFetcher:
    def __init__(self, api_key: str)  # FMP API key
    def get_earnings_calendar(self, symbols: List[str], days_ahead: int = 30) -> Dict[str, EarningsData]
    def get_analyst_revisions(self, symbol: str, days: int = 30) -> AnalystRevisions
```

**Integration:**
- Extends existing `market_context.py` earnings data
- Uses FMP API (same as current earnings monitoring)

---

#### E.4: SimilarityMatcher (`data/similarity.py`)

```python
@dataclass(slots=True)
class SignalFeatures:
    strategy: str
    symbol: str
    market_state: str
    vix_level: float
    dte: int
    delta: float
    underlying_ma_position: float  # % above/below MA50

class SimilarityMatcher:
    def __init__(self, history_store: HistoryStore)
    def extract_features(self, signal: TradeSignal, context: dict) -> SignalFeatures
    def find_similar(self, features: SignalFeatures, top_k: int = 20) -> List[SignalOutcome]
    def compute_historical_win_rate(self, similar: List[SignalOutcome]) -> float
```

**Algorithm:**
- Euclidean distance in normalized feature space
- Weight recent signals higher (exponential decay, half-life = 30 days)

---

### Integration Points

1. **`runner.py:run_once()`** - Inject flow scores and historical similarity into signal context
2. **`signal_ranking.py`** - Add historical similarity as a scoring factor
3. **Strategy `on_data()`** - Optional flow confirmation check before generating signals

---

## Module D: Regime Adapter

### Purpose

Dynamically adjust strategy behavior based on detected market regime.

### Regime Taxonomy

| Regime | VIX Range | SPY vs MA50 | Description |
|--------|-----------|-------------|-------------|
| `LOW_VOL_BULL` | < 15 | Above | complacent uptrend |
| `NORMAL_BULL` | 15-20 | Above | healthy uptrend |
| `NORMAL_BEAR` | 15-20 | Below | healthy downtrend |
| `HIGH_VOL_BEAR` | 20-35 | Below | stress/panic |
| `CRUSH` | > 35 | Any | extreme fear |
| `TRANSITION_UP` | Falling | Crossing up | bottoming |
| `TRANSITION_DOWN` | Rising | Crossing down | topping |

---

### Components

#### D.1: RegimeDetector (`regime/detector.py`)

```python
@dataclass(slots=True)
class RegimeResult:
    regime: RegimeType
    confidence: float
    as_of: datetime
    signals: Dict[str, Any]  # Supporting indicators

class RegimeDetector:
    def __init__(self, vix_threshold_low: float = 15.0, vix_threshold_high: float = 25.0)
    def detect(self, market_data: Dict[str, Any]) -> RegimeResult
    def detect_transition(self, history: pd.DataFrame) -> Optional[TransitionSignal]
```

**Inputs:**
- VIX level and trend (5-day MA)
- SPY/QQQ/IWM vs MA50 and MA200
- Market breadth (% of stocks above MA50)
- VIX term structure (VIX vs VIX3M)

---

#### D.2: RegimeConfig (`regime/config.py`)

```yaml
# regime_config.yaml
regimes:
  LOW_VOL_BULL:
    strategies:
      PutCreditSpreadStrategy:
        enabled: true
        params_override:
          min_days_to_expiry: 14
          max_days_to_expiry: 30
          short_target_delta: 0.20
      VerticalSpreadStrategy:
        enabled: true
        params_override:
          bullish_only: true
    exits:
      profit_target_pct: 0.50
      trailing_stop_delta: 0.15

  HIGH_VOL_BEAR:
    strategies:
      PutCreditSpreadStrategy:
        enabled: false
      VerticalSpreadStrategy:
        enabled: true
        params_override:
          bullish_only: false
          bearish_iv_rank_trigger: 0.40
    exits:
      profit_target_pct: 0.75
      stop_loss_pct: 0.50
```

---

#### D.3: StrategyGater (`regime/gater.py`)

```python
class StrategyGater:
    def __init__(self, regime_config: RegimeConfig)
    def is_enabled(self, strategy_name: str, regime: RegimeType) -> bool
    def get_overrides(self, strategy_name: str, regime: RegimeType) -> Dict[str, Any]
```

---

#### D.4: ParamOverrider (`regime/params.py`)

```python
class ParamOverrider:
    def __init__(self, strategy: BaseOptionStrategy, overrides: Dict[str, Any])
    def apply(self)
    def restore(self)
```

**Implementation:**
- Monkey-patch strategy attributes at runtime
- Restore original values after signal generation

---

### Integration Points

1. **`main.py:discover_strategies()`** - Inject `StrategyGater` and `ParamOverrider`
2. **`market_state.py`** - Extend `MarketStateProvider` to expose regime type
3. **`runner.py:run_once()`** - Compute regime once per run, pass to all strategies

---

## Module A: Entry Filter

### Purpose

Add pre-signal validation layers to improve entry quality.

### Components

#### A.1: PatternRecognizer (`entry/patterns.py`)

```python
@dataclass(slots=True)
class PatternSignal:
    pattern_type: str  # "BREAKOUT", "REVERSAL", "CONSOLIDATION"
    symbol: str
    timestamp: datetime
    confidence: float
    price_level: float

class PatternRecognizer:
    def detect_breakout(self, ohlcv: pd.DataFrame, lookback: int = 20) -> Optional[PatternSignal]
    def detect_reversal(self, ohlcv: pd.DataFrame) -> Optional[PatternSignal]
    def detect_consolidation(self, ohlcv: pd.DataFrame) -> Optional[PatternSignal]
```

**Patterns:**
- **Breakout:** Price > 20-day high with volume > 1.5x average
- **Reversal:** RSI divergence + hammer/shooting star candlestick
- **Consolidation:** ATR < 50% of 20-day average for 5+ days

---

#### A.2: ConfluenceChecker (`entry/confluence.py`)

```python
@dataclass(slots=True)
class ConfluenceResult:
    passed: bool
    signals_aligned: int
    total_signals: int
    alignment_score: float

class ConfluenceChecker:
    def __init__(self, min_aligned: int = 3)
    def check(self, snapshot: OptionChainSnapshot, context: dict) -> ConfluenceResult
```

**Confluence Signals (require N of M):**
1. Price above MA50
2. Price above MA200
3. RSI(14) > 50 for bullish, < 50 for bearish
4. Volume > 20-day average
5. MACD above signal line
6. Options flow confirms direction
7. Historical similarity win rate > 55%

---

#### A.3: StrategySelector (`entry/selector.py`)

```python
class StrategySelector:
    def __init__(self, history_store: HistoryStore)
    def get_preferred_strategies(self, market_state: str, symbol: str) -> List[str]
    def get_boost_factor(self, strategy: str, symbol: str) -> float
```

**Ranking Boost:**
- Strategies with >60% win rate in current regime: +20% score boost
- Strategies with <40% win rate in current regime: -20% score penalty

---

### Integration Points

1. **`runner.py:run_once()`** - Apply confluence filter before signal ranking
2. **`signal_ranking.py`** - Apply strategy boost factors
3. **Strategy `on_data()`** - Optional pattern check before signal generation

---

## Module C: Exit Engine

### Purpose

Systematic exit management with multiple exit strategies.

### Components

#### C.1: ExitConfig (`exit/config.py`)

```python
@dataclass(slots=True)
class ExitRule:
    strategy: str
    profit_target_pct: Optional[float] = 0.50
    stop_loss_pct: Optional[float] = None
    close_at_dte: int = 5
    trailing_stop_delta: Optional[float] = None
    use_ai_advisor: bool = False

class ExitEngine:
    def __init__(self, rules: List[ExitRule], history_store: HistoryStore)
    def evaluate(self, position: Position, snapshot: OptionChainSnapshot) -> Optional[ExitAction]
```

---

#### C.2: DynamicTargetCalculator (`exit/targets.py`)

```python
class DynamicTargetCalculator:
    def compute_profit_target(self, symbol: str, entry_price: float, atr: float) -> float
    def compute_stop_loss(self, symbol: str, entry_price: float, atr: float) -> float
```

**Formula:**
- Profit target = Entry + (1.5 × ATR) for debit spreads
- Profit target = Entry × (1 - 0.50) for credit spreads (50% max profit)
- Stop loss = Entry - (1.0 × ATR) for debit spreads
- Stop loss = Entry × (1 + 0.50) for credit spreads (50% max loss)

---

#### C.3: TimeExitEvaluator (`exit/time_based.py`)

```python
class TimeExitEvaluator:
    def should_close(self, position: Position, current_dte: int, rule: ExitRule) -> bool
    def get_close_reason(self, position: Position, current_dte: int) -> str
```

**Rules:**
- Close at DTE <= 5 if profit > 25%
- Close at DTE <= 3 regardless (gamma risk)
- Close immediately if underlying breached strike (for short options)

---

#### C.4: TrailingStopManager (`exit/trailing.py`)

```python
class TrailingStopManager:
    def update(self, position: Position, current_price: float)
    def check_stop(self, current_price: float) -> bool
```

**Trailing Logic:**
- For long options: Trail when delta drops below threshold (e.g., 0.30 from 0.50)
- For credit spreads: Trail when underlying moves favorably by 2× ATR

---

#### C.5: AIExitAdvisor (`exit/ai_advisor.py`)

```python
class AIExitAdvisor:
    def __init__(self, gemini_client: GeminiClient)
    def recommend(self, position: Position, market_context: dict) -> ExitRecommendation
```

**Prompt Template:**
```
You are an options exit advisor. Evaluate this position:

Position: {position_summary}
Current Market: {market_state}
Historical Context: {similar_outcomes}

Recommend: HOLD, CLOSE, or ROLL with brief reasoning.
```

---

### Integration Points

1. **`portfolio/rules.py`** - Add exit rule evaluation
2. **`portfolio/evaluation.py`** - Integrate exit engine into position evaluation
3. **`portfolio/manager.py`** - Generate exit actions from recommendations

---

## Implementation Sequence

### Phase 1: Module E (Data Enhancer)

1. Implement `HistoryStore` for signal tracking
2. Add `OptionsFlowFetcher` for unusual activity detection
3. Implement `FundamentalFetcher` for earnings/analyst data
4. Build `SimilarityMatcher` for historical comparison

**Files to Create:**
- `data/flow.py`
- `data/history.py`
- `data/fundamental.py`
- `data/similarity.py`

**Files to Modify:**
- `market_context.py` - Add flow and fundamental data
- `signal_ranking.py` - Add historical similarity factor

---

### Phase 2: Module D (Regime Adapter)

1. Extend `market_state.py` with regime detection
2. Create `RegimeConfig` YAML structure
3. Implement `StrategyGater` for enabling/disabling strategies
4. Build `ParamOverrider` for runtime parameter injection

**Files to Create:**
- `regime/detector.py`
- `regime/config.py`
- `regime/gater.py`
- `regime/params.py`
- `regime_config.yaml`

**Files to Modify:**
- `market_state.py` - Add regime classification
- `main.py` - Inject regime overrides into strategies
- `runner.py` - Compute regime once per run

---

### Phase 3: Module A (Entry Filter)

1. Implement `PatternRecognizer` for technical patterns
2. Build `ConfluenceChecker` for multi-indicator alignment
3. Create `StrategySelector` for regime-aware strategy ranking

**Files to Create:**
- `entry/patterns.py`
- `entry/confluence.py`
- `entry/selector.py`

**Files to Modify:**
- `runner.py` - Apply confluence filter before ranking
- `signal_ranking.py` - Apply strategy boost factors

---

### Phase 4: Module C (Exit Engine)

1. Define `ExitRule` configuration per strategy
2. Implement `DynamicTargetCalculator` for ATR-based targets
3. Build `TimeExitEvaluator` for DTE-based exits
4. Create `TrailingStopManager` for delta-based trailing
5. Integrate `AIExitAdvisor` for Gemini-driven exits

**Files to Create:**
- `exit/config.py`
- `exit/targets.py`
- `exit/time_based.py`
- `exit/trailing.py`
- `exit/ai_advisor.py`

**Files to Modify:**
- `portfolio/rules.py` - Add exit evaluation
- `portfolio/evaluation.py` - Integrate exit engine
- `portfolio/manager.py` - Generate exit actions

---

## Configuration Changes

### config.yaml Additions

```yaml
# Entry filter configuration
entry_filter:
  confluence:
    min_aligned: 3  # Require 3 of 7 signals aligned
  patterns:
    enabled: true
    min_confidence: 0.6
  strategy_selector:
    enabled: true
    boost_factor: 0.20  # 20% score boost for favored strategies

# Exit engine configuration
exit_engine:
  default_rules:
    profit_target_pct: 0.50
    stop_loss_pct: null
    close_at_dte: 5
    trailing_stop_delta: null
  strategy_rules:
    PutCreditSpreadStrategy:
      profit_target_pct: 0.50
      close_at_dte: 7
    VerticalSpreadStrategy:
      profit_target_pct: 0.50
      stop_loss_pct: 0.50
  ai_advisor:
    enabled: true
    min_confidence: 0.6

# Regime adapter configuration
regime_adapter:
  enabled: true
  config_file: regime_config.yaml
  detect_transitions: true

# Data enhancer configuration
data_enhancer:
  options_flow:
    enabled: true
    volume_oi_threshold: 2.0
  history:
    enabled: true
    storage_file: data/signal_history.jsonl
  fundamentals:
    enabled: true
    fmp_api_key: ${FMP_API_KEY}
```

---

## Testing Strategy

### Unit Tests

- `test_patterns.py` - Pattern detection accuracy
- `test_confluence.py` - Confluence calculation
- `test_regime_detector.py` - Regime classification
- `test_exit_engine.py` - Exit trigger logic
- `test_similarity.py` - Historical matching accuracy

### Integration Tests

- End-to-end signal generation with all modules enabled
- Regime-aware strategy filtering
- Exit recommendation generation

### Backtesting

- Compare signal win rates before/after entry filters
- Compare exit P&L with/without dynamic targets
- Measure regime-specific performance improvement

---

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Signal win rate | ~55% | >65% |
| Average R/R ratio | ~1.5 | >2.0 |
| Exit timing P&L improvement | - | +15% |
| Regime-specific win rate | - | +10% in HIGH_VOL_BEAR |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to historical data | High | Use out-of-sample testing, limit lookback window |
| Increased latency from data fetches | Medium | Cache data, async prefetching |
| False positive pattern detection | Medium | Require confluence, not single patterns |
| Regime whipsaw (rapid transitions) | Medium | Add hysteresis to regime detection |
| API rate limits (FMP, Gemini) | Low | Implement rate limiting, graceful fallbacks |

---

## Appendix: Interface Contracts

### TradeSignal Extensions

```python
@dataclass(slots=True)
class TradeSignal:
    # Existing fields...
    symbol: str
    expiry: datetime
    strike: float
    option_type: str
    direction: str
    rationale: str
    legs: Tuple[SignalLeg, ...]

    # New optional fields (Module A)
    pattern_confidence: Optional[float] = None
    confluence_score: Optional[float] = None
    historical_win_rate: Optional[float] = None
```

### Position Extensions

```python
@dataclass(slots=True)
class Position:
    # Existing fields...
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime

    # New optional fields (Module C)
    profit_target: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop_delta: Optional[float] = None
```

### MarketContext Extensions

```python
@dataclass(slots=True)
class MarketContext:
    # Existing fields...
    vix_level: float
    market_states: Dict[str, str]

    # New fields (Module D)
    regime: RegimeType
    regime_confidence: float

    # New fields (Module E)
    flow_scores: Dict[str, float]
    earnings_calendar: Dict[str, EarningsData]
```
