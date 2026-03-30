# Signal Quality Enhancement System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a modular signal quality enhancement system with 4 independent modules (Data Enhancer, Regime Adapter, Entry Filter, Exit Engine) to improve win rates from ~55% to >65%.

**Architecture:** Four independent modules that can be developed and merged separately: Module E (data foundation) → Module D (regime adaptation) → Module A (entry filtering) → Module C (exit optimization). Each module extends existing framework without replacing core components.

**Tech Stack:** Python 3.12, pandas, NautilusTrader, IBKR API, Google Gemini API, FMP API for fundamentals.

---

## Phase 1: Module E - Data Enhancer

### Task E.1: HistoryStore for Signal Tracking

**Files:**
- Create: `src/data/history.py`
- Create: `tests/test_history_store.py`
- Modify: `src/runner.py:116-118` (integrate HistoryStore)

- [ ] **Step 1: Write tests for HistoryStore**

```python
# tests/test_history_store.py
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone
from data.history import HistoryStore, SignalOutcome

class TestHistoryStore:
    def test_init_creates_directory(self, tmp_path):
        store = HistoryStore(tmp_path / "history.jsonl")
        assert (tmp_path / "history.jsonl").exists()

    def test_record_signal_writes_to_file(self, tmp_path):
        store = HistoryStore(tmp_path / "history.jsonl")
        signal = TradeSignal(
            symbol="NVDA",
            expiry=datetime.now(timezone.utc),
            strike=100.0,
            option_type="CALL",
            direction="BULL_CALL_DEBIT_SPREAD",
            rationale="Test signal"
        )
        store.record_signal("test_001", signal, 2.50)

        lines = (tmp_path / "history.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["signal_id"] == "test_001"
        assert data["entry_price"] == 2.50

    def test_record_exit_updates_record(self, tmp_path):
        store = HistoryStore(tmp_path / "history.jsonl")
        signal = TradeSignal(
            symbol="NVDA",
            expiry=datetime.now(timezone.utc),
            strike=100.0,
            option_type="CALL",
            direction="BULL_CALL_DEBIT_SPREAD",
            rationale="Test"
        )
        store.record_signal("test_001", signal, 2.50)
        store.record_exit("test_001", 4.80, datetime.now(timezone.utc))

        outcomes = store.get_all_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].exit_price == 4.80
        assert outcomes[0].outcome == "WIN"

    def test_get_strategy_stats_empty(self, tmp_path):
        store = HistoryStore(tmp_path / "history.jsonl")
        stats = store.get_strategy_stats("TestStrategy")
        assert stats.trade_count == 0
        assert stats.win_rate == 0.0

    def test_get_strategy_stats_with_data(self, tmp_path):
        store = HistoryStore(tmp_path / "history.jsonl")
        # Add 3 wins, 1 loss
        for i in range(3):
            signal = TradeSignal(symbol="NVDA", expiry=datetime.now(timezone.utc), strike=100.0, option_type="CALL", direction="BULL", rationale="Test")
            store.record_signal(f"win_{i}", signal, 2.0)
            store.record_exit(f"win_{i}", 4.0, datetime.now(timezone.utc))

        signal = TradeSignal(symbol="NVDA", expiry=datetime.now(timezone.utc), strike=100.0, option_type="CALL", direction="BULL", rationale="Test")
        store.record_signal("loss_1", signal, 2.0)
        store.record_exit("loss_1", 1.0, datetime.now(timezone.utc))

        stats = store.get_strategy_stats("TestStrategy", window_days=90)
        assert stats.trade_count == 4
        assert stats.win_rate == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Volumes/Data/Code/optionscanner
source .venv/bin/activate
pytest tests/test_history_store.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'data'"

- [ ] **Step 3: Create directory structure and implement HistoryStore**

```python
# src/data/__init__.py
# Empty init file for data package
```

```python
# src/data/history.py
"""Persistent signal history tracking for performance analysis."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from strategies.base import TradeSignal


@dataclass(slots=True)
class SignalOutcome:
    """Represents a tracked signal with its outcome."""

    signal_id: str
    strategy: str
    symbol: str
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    max_profit: Optional[float]
    max_loss: Optional[float]
    outcome: str  # "WIN", "LOSS", "OPEN", "BREAKEVEN"

    def is_closed(self) -> bool:
        return self.outcome in ("WIN", "LOSS", "BREAKEVEN")

    def is_win(self) -> bool:
        return self.outcome == "WIN"


@dataclass(slots=True)
class StrategyStats:
    """Aggregated statistics for a strategy."""

    strategy: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    open_trades: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class HistoryStore:
    """Persists and analyzes signal history for performance tracking.

    Signals are recorded when generated and updated when exited.
    Storage format: JSONL in data/signal_history.jsonl
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize history store.

        Args:
            data_dir: Directory for data storage (creates signal_history.jsonl)
        """
        self._storage_path = data_dir / "signal_history.jsonl"
        self._outcomes: List[SignalOutcome] = []
        self._stats_cache: Dict[str, StrategyStats] = {}
        self._cache_dirty = False

        # Ensure directory exists
        data_dir.mkdir(parents=True, exist_ok=True)
        if not self._storage_path.exists():
            self._storage_path.touch()

        self._load_history()

    def _load_history(self) -> None:
        """Load history from JSONL file."""
        if not self._storage_path.exists():
            return

        try:
            content = self._storage_path.read_text(encoding="utf-8")
            if not content.strip():
                return

            for line in content.strip().split("\n"):
                if line.strip():
                    data = json.loads(line)
                    self._outcomes.append(SignalOutcome(**data))

            logger.info(
                "Loaded signal history | count={count}",
                count=len(self._outcomes)
            )
        except Exception as exc:
            logger.warning("Failed to load signal history | path={path} reason={error}",
                          path=str(self._storage_path), error=exc)

    def _save_history(self) -> None:
        """Persist history to JSONL file."""
        try:
            lines = []
            for outcome in self._outcomes:
                lines.append(json.dumps(asdict(outcome)))

            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._cache_dirty = False

            logger.debug("Saved signal history | count={count}", count=len(self._outcomes))
        except Exception as exc:
            logger.warning("Failed to save signal history | reason={error}", error=exc)

    def record_signal(
        self,
        signal_id: str,
        signal: TradeSignal,
        entry_price: float,
        strategy: Optional[str] = None
    ) -> None:
        """Record a new signal entry.

        Args:
            signal_id: Unique identifier for this signal
            signal: TradeSignal object
            entry_price: Entry price paid/received
            strategy: Strategy name (defaults to signal.direction)
        """
        outcome = SignalOutcome(
            signal_id=signal_id,
            strategy=strategy or signal.direction,
            symbol=signal.symbol or "UNKNOWN",
            entry_date=datetime.now(timezone.utc).isoformat(),
            exit_date=None,
            entry_price=entry_price,
            exit_price=None,
            pnl=None,
            max_profit=None,
            max_loss=None,
            outcome="OPEN"
        )

        self._outcomes.append(outcome)
        self._cache_dirty = True

        logger.info(
            "Recorded signal | id={id} strategy={strategy} symbol={symbol} entry={entry}",
            id=signal_id,
            strategy=outcome.strategy,
            symbol=outcome.symbol,
            entry=entry_price
        )

    def record_exit(
        self,
        signal_id: str,
        exit_price: float,
        exit_date: Optional[datetime] = None,
        max_profit: Optional[float] = None,
        max_loss: Optional[float] = None
    ) -> None:
        """Record exit for a signal.

        Args:
            signal_id: Signal identifier to update
            exit_price: Exit price
            exit_date: Exit timestamp (defaults to now)
            max_profit: Max unrealized profit seen during trade
            max_loss: Max unrealized loss seen during trade
        """
        for outcome in self._outcomes:
            if outcome.signal_id == signal_id:
                pnl = exit_price - outcome.entry_price

                if pnl > 0.05:  # >5% gain = WIN
                    outcome_str = "WIN"
                elif pnl < -0.05:  # <5% loss = LOSS
                    outcome_str = "LOSS"
                else:
                    outcome_str = "BREAKEVEN"

                outcome.exit_date = (exit_date or datetime.now(timezone.utc)).isoformat()
                outcome.exit_price = exit_price
                outcome.pnl = pnl
                outcome.max_profit = max_profit
                outcome.max_loss = max_loss
                outcome.outcome = outcome_str

                self._cache_dirty = True

                logger.info(
                    "Recorded exit | id={id} exit={exit} pnl={pnl} outcome={outcome}",
                    id=signal_id,
                    exit=exit_price,
                    pnl=pnl,
                    outcome=outcome_str
                )
                return

        logger.warning("Signal not found for exit | id={id}", id=signal_id)

    def get_strategy_stats(
        self,
        strategy: str,
        window_days: int = 90
    ) -> StrategyStats:
        """Get aggregated statistics for a strategy.

        Args:
            strategy: Strategy name
            window_days: Lookback window in days (0 = all time)

        Returns:
            StrategyStats with aggregated metrics
        """
        cache_key = f"{strategy}_{window_days}"
        if cache_key in self._stats_cache and not self._cache_dirty:
            return self._stats_cache[cache_key]

        cutoff = datetime.now(timezone.utc).timestamp() - (window_days * 86400) if window_days > 0 else 0

        strategy_outcomes = [
            o for o in self._outcomes
            if o.strategy == strategy and o.is_closed()
        ]

        if window_days > 0:
            strategy_outcomes = [
                o for o in strategy_outcomes
                if o.exit_date and datetime.fromisoformat(o.exit_date).timestamp() > cutoff
            ]

        open_outcomes = [o for o in self._outcomes if o.strategy == strategy and not o.is_closed()]

        win_count = sum(1 for o in strategy_outcomes if o.is_win())
        loss_count = len(strategy_outcomes) - win_count

        win_rate = win_count / len(strategy_outcomes) if strategy_outcomes else 0.0

        pnl_values = [o.pnl for o in strategy_outcomes if o.pnl is not None]
        avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0
        total_pnl = sum(pnl_values) if pnl_values else 0.0

        stats = StrategyStats(
            strategy=strategy,
            trade_count=len(strategy_outcomes),
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            open_trades=len(open_outcomes)
        )

        self._stats_cache[cache_key] = stats
        return stats

    def get_all_outcomes(self) -> List[SignalOutcome]:
        """Return all signal outcomes."""
        return list(self._outcomes)

    def get_closed_outcomes(self) -> List[SignalOutcome]:
        """Return only closed outcomes."""
        return [o for o in self._outcomes if o.is_closed()]

    def get_open_outcomes(self) -> List[SignalOutcome]:
        """Return only open outcomes."""
        return [o for o in self._outcomes if not o.is_closed()]

    def flush(self) -> None:
        """Force save history to disk."""
        if self._cache_dirty:
            self._save_history()


__all__ = ["HistoryStore", "SignalOutcome", "StrategyStats"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Volumes/Data/Code/optionscanner
source .venv/bin/activate
pytest tests/test_history_store.py::TestHistoryStore::test_init_creates_directory -v
pytest tests/test_history_store.py::TestHistoryStore::test_record_signal_writes_to_file -v
pytest tests/test_history_store.py::TestHistoryStore::test_record_exit_updates_record -v
pytest tests/test_history_store.py::TestHistoryStore::test_get_strategy_stats_empty -v
pytest tests/test_history_store.py::TestHistoryStore::test_get_strategy_stats_with_data -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/__init__.py src/data/history.py tests/test_history_store.py
git commit -m "feat: add HistoryStore for signal performance tracking (Module E.1)"
```

---

### Task E.2: SimilarityMatcher for Historical Comparison

**Files:**
- Create: `src/data/similarity.py`
- Create: `tests/test_similarity.py`

- [ ] **Step 1: Write tests for SimilarityMatcher**

```python
# tests/test_similarity.py
import pytest
from datetime import datetime, timezone
from data.similarity import SimilarityMatcher, SignalFeatures
from data.history import HistoryStore, SignalOutcome
from strategies.base import TradeSignal

class TestSimilarityMatcher:
    @pytest.fixture
    def history_store(self, tmp_path):
        return HistoryStore(tmp_path / "history.jsonl")

    def test_extract_features(self, history_store):
        matcher = SimilarityMatcher(history_store)
        signal = TradeSignal(
            symbol="NVDA",
            expiry=datetime.now(timezone.utc),
            strike=100.0,
            option_type="CALL",
            direction="BULL_CALL_DEBIT_SPREAD",
            rationale="Test"
        )
        context = {
            "market_state": "bull",
            "vix_level": 18.0,
            "dte": 30,
            "delta": 0.35,
            "underlying_price": 105.0,
            "ma50": 100.0
        }

        features = matcher.extract_features(signal, context)

        assert features.strategy == "BULL_CALL_DEBIT_SPREAD"
        assert features.symbol == "NVDA"
        assert features.market_state == "bull"
        assert features.vix_level == 18.0
        assert features.dte == 30
        assert features.underlying_ma_position == pytest.approx(0.05, rel=0.01)

    def test_find_similar_empty_history(self, history_store):
        matcher = SimilarityMatcher(history_store)
        features = SignalFeatures(
            strategy="BULL_CALL", symbol="NVDA", market_state="bull",
            vix_level=18.0, dte=30, delta=0.35, underlying_ma_position=0.05
        )

        similar = matcher.find_similar(features, top_k=10)
        assert len(similar) == 0

    def test_compute_historical_win_rate(self, history_store):
        matcher = SimilarityMatcher(history_store)

        # Add 4 wins, 1 loss
        outcomes = []
        for i in range(4):
            outcomes.append(SignalOutcome(
                signal_id=f"win_{i}", strategy="Test", symbol="NVDA",
                entry_date="2026-01-01", exit_date="2026-01-05",
                entry_price=2.0, exit_price=4.0, pnl=2.0, outcome="WIN"
            ))
        outcomes.append(SignalOutcome(
            signal_id="loss_1", strategy="Test", symbol="NVDA",
            entry_date="2026-01-01", exit_date="2026-01-05",
            entry_price=2.0, exit_price=1.0, pnl=-1.0, outcome="LOSS"
        ))

        win_rate = matcher.compute_historical_win_rate(outcomes)
        assert win_rate == pytest.approx(0.80)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_similarity.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'data.similarity'"

- [ ] **Step 3: Implement SimilarityMatcher**

```python
# src/data/similarity.py
"""Historical signal similarity matching for performance estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from data.history import HistoryStore, SignalOutcome
from strategies.base import TradeSignal


@dataclass(slots=True)
class SignalFeatures:
    """Feature vector for signal similarity comparison."""

    strategy: str
    symbol: str
    market_state: str
    vix_level: float
    dte: int
    delta: float
    underlying_ma_position: float  # % above/below MA50

    def normalize(self) -> Dict[str, float]:
        """Return normalized feature vector for distance calculation."""
        # Market state encoding
        state_map = {"bull": 1.0, "uptrend": 0.7, "bear": 0.0}

        return {
            "vix": self.vix_level / 50.0,  # Normalize to 0-1 (VIX typically 10-50)
            "dte": min(self.dte / 90.0, 1.0),  # Cap at 90 days
            "delta": abs(self.delta),  # 0-1 range
            "ma_position": max(-1.0, min(self.underlying_ma_position, 1.0)),  # -100% to +100%
            "state": state_map.get(self.market_state.lower(), 0.5),
        }


class SimilarityMatcher:
    """Finds historical signals similar to current candidate.

    Uses Euclidean distance in normalized feature space.
    Weights recent signals higher with exponential decay (half-life = 30 days).
    """

    def __init__(
        self,
        history_store: HistoryStore,
        half_life_days: int = 30
    ) -> None:
        """Initialize matcher.

        Args:
            history_store: History store to query
            half_life_days: Decay half-life for recency weighting
        """
        self._history = history_store
        self._half_life = half_life_days

    def extract_features(
        self,
        signal: TradeSignal,
        context: Dict[str, object]
    ) -> SignalFeatures:
        """Extract feature vector from signal and market context.

        Args:
            signal: Current trade signal
            context: Market context dict with vix_level, market_state, etc.

        Returns:
            SignalFeatures for similarity comparison
        """
        underlying_price = context.get("underlying_price", signal.strike)
        ma50 = context.get("ma50", underlying_price)
        ma_position = (underlying_price - ma50) / ma50 if ma50 > 0 else 0.0

        return SignalFeatures(
            strategy=signal.direction,
            symbol=(signal.symbol or "").upper(),
            market_state=str(context.get("market_state", "unknown")),
            vix_level=float(context.get("vix_level", 20.0)),
            dte=context.get("dte", 30),
            delta=context.get("delta", 0.3),
            underlying_ma_position=ma_position
        )

    def find_similar(
        self,
        features: SignalFeatures,
        top_k: int = 20
    ) -> List[SignalOutcome]:
        """Find historically similar signals.

        Args:
            features: Current signal features
            top_k: Number of similar signals to return

        Returns:
            List of similar historical outcomes, sorted by similarity
        """
        all_outcomes = self._history.get_closed_outcomes()

        if not all_outcomes:
            return []

        # Calculate distance for each historical outcome
        scored: List[tuple[float, SignalOutcome]] = []
        now = datetime.now(timezone.utc)

        for outcome in all_outcomes:
            # Skip outcomes without required features
            if not outcome.exit_date:
                continue

            # Reconstruct features from stored outcome (simplified - uses defaults)
            # In production, would store full feature vector with each outcome
            hist_features = SignalFeatures(
                strategy=outcome.strategy,
                symbol=outcome.symbol,
                market_state="unknown",
                vix_level=20.0,
                dte=30,
                delta=0.3,
                underlying_ma_position=0.0
            )

            distance = self._euclidean_distance(features, hist_features)

            # Apply recency weight
            try:
                exit_date = datetime.fromisoformat(outcome.exit_date)
                days_ago = (now - exit_date).days
                recency_weight = math.pow(0.5, days_ago / self._half_life)
            except (ValueError, TypeError):
                recency_weight = 0.5

            weighted_distance = distance / recency_weight if recency_weight > 0 else distance
            scored.append((weighted_distance, outcome))

        # Sort by weighted distance (ascending)
        scored.sort(key=lambda x: x[0])

        # Return top K
        similar = [outcome for _, outcome in scored[:top_k]]

        logger.debug(
            "Found similar signals | count={count} top_distance={dist:.3f}",
            count=len(similar),
            dist=scored[0][0] if scored else 0.0
        )

        return similar

    def compute_historical_win_rate(
        self,
        similar: List[SignalOutcome]
    ) -> float:
        """Compute win rate from similar historical signals.

        Args:
            similar: List of similar historical outcomes

        Returns:
            Win rate (0-1), or 0.5 if no data
        """
        if not similar:
            return 0.5  # Neutral prior

        wins = sum(1 for o in similar if o.is_win())
        win_rate = wins / len(similar)

        logger.debug(
            "Historical win rate | wins={wins} total={total} rate={rate:.2%}",
            wins=wins,
            total=len(similar),
            rate=win_rate
        )

        return win_rate

    @staticmethod
    def _euclidean_distance(a: SignalFeatures, b: SignalFeatures) -> float:
        """Calculate Euclidean distance between two feature vectors."""
        a_norm = a.normalize()
        b_norm = b.normalize()

        sum_sq = 0.0
        for key in a_norm:
            diff = a_norm[key] - b_norm.get(key, 0.5)
            sum_sq += diff * diff

        return math.sqrt(sum_sq)


__all__ = ["SimilarityMatcher", "SignalFeatures"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_similarity.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data/similarity.py tests/test_similarity.py
git commit -m "feat: add SimilarityMatcher for historical signal comparison (Module E.2)"
```

---

### Task E.3: OptionsFlowFetcher for Unusual Activity Detection

**Files:**
- Create: `src/data/flow.py`
- Create: `tests/test_flow.py`

- [ ] **Step 1: Write tests for OptionsFlowFetcher**

```python
# tests/test_flow.py
import pytest
from datetime import datetime, timezone
from data.flow import OptionsFlowFetcher, FlowAlert

class TestOptionsFlowFetcher:
    def test_compute_flow_score_high_volume(self):
        fetcher = OptionsFlowFetcher()
        alerts = [
            FlowAlert(
                symbol="NVDA", timestamp=datetime.now(timezone.utc),
                volume=10000, open_interest=2000, volume_oi_ratio=5.0,
                premium=50000.0, side="BUY", sweep_detected=False
            )
        ]

        score = fetcher.compute_flow_score("NVDA", alerts)
        assert score > 0.7  # High volume/OI ratio should score high

    def test_compute_flow_score_sweep(self):
        fetcher = OptionsFlowFetcher()
        alerts = [
            FlowAlert(
                symbol="NVDA", timestamp=datetime.now(timezone.utc),
                volume=5000, open_interest=2000, volume_oi_ratio=2.5,
                premium=25000.0, side="BUY", sweep_detected=True
            )
        ]

        score = fetcher.compute_flow_score("NVDA", alerts)
        assert score > 0.5  # Sweep detection should boost score

    def test_compute_flow_score_low_activity(self):
        fetcher = OptionsFlowFetcher()
        alerts = [
            FlowAlert(
                symbol="NVDA", timestamp=datetime.now(timezone.utc),
                volume=100, open_interest=2000, volume_oi_ratio=0.05,
                premium=500.0, side="BUY", sweep_detected=False
            )
        ]

        score = fetcher.compute_flow_score("NVDA", alerts)
        assert score < 0.3  # Low activity should score low
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_flow.py -v
```

- [ ] **Step 3: Implement OptionsFlowFetcher**

```python
# src/data/flow.py
"""Options flow detection for unusual activity monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger


@dataclass(slots=True)
class FlowAlert:
    """Represents unusual options activity alert."""

    symbol: str
    timestamp: datetime
    volume: int
    open_interest: int
    volume_oi_ratio: float
    premium: float
    side: str  # "BUY" or "SELL"
    sweep_detected: bool


class OptionsFlowFetcher:
    """Detects and scores unusual options flow activity.

    Monitors volume/OI ratio and sweep detection to identify
    potentially informed trading activity.
    """

    def __init__(
        self,
        volume_oi_threshold: float = 2.0
    ) -> None:
        """Initialize flow fetcher.

        Args:
            volume_oi_threshold: Volume/OI ratio threshold for alerts
        """
        self._threshold = volume_oi_threshold

    def compute_flow_score(
        self,
        symbol: str,
        alerts: List[FlowAlert]
    ) -> float:
        """Compute flow score (0-1) from alerts.

        Args:
            symbol: Underlying symbol
            alerts: List of flow alerts for this symbol

        Returns:
            Flow score (0-1, higher = more unusual activity)
        """
        if not alerts:
            return 0.5  # Neutral if no data

        scores: List[float] = []

        for alert in alerts:
            # Base score from volume/OI ratio
            ratio_score = min(alert.volume_oi_ratio / self._threshold, 2.0) / 2.0

            # Sweep bonus
            sweep_bonus = 0.2 if alert.sweep_detected else 0.0

            # Premium size bonus (scale by premium, cap at 0.2)
            premium_bonus = min(alert.premium / 100000.0, 0.2)

            # Combine
            alert_score = min(ratio_score + sweep_bonus + premium_bonus, 1.0)
            scores.append(alert_score)

        # Average score across all alerts
        avg_score = sum(scores) / len(scores)

        logger.debug(
            "Computed flow score | symbol={symbol} score={score:.2f} alerts={count}",
            symbol=symbol,
            score=avg_score,
            count=len(alerts)
        )

        return avg_score

    def fetch_unusual_activity(
        self,
        symbols: List[str]
    ) -> Dict[str, List[FlowAlert]]:
        """Fetch unusual activity for symbols.

        Note: This is a placeholder for future IBKR integration.
        Currently returns empty dicts - actual implementation requires
        IBKR option chain volume/OI data.

        Args:
            symbols: Symbols to check

        Returns:
            Dict mapping symbol to list of alerts
        """
        # Placeholder - returns empty for all symbols
        # Future implementation will query IBKR for:
        # - Current option chain volume
        # - Historical average volume
        # - Open interest by strike
        # - Multi-leg detection for sweeps

        logger.debug(
            "Unusual activity fetch (placeholder) | symbols={count}",
            count=len(symbols)
        )

        return {symbol: [] for symbol in symbols}


__all__ = ["OptionsFlowFetcher", "FlowAlert"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_flow.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data/flow.py tests/test_flow.py
git commit -m "feat: add OptionsFlowFetcher for unusual activity detection (Module E.3)"
```

---

### Task E.4: Integrate Module E with Signal Ranking

**Files:**
- Modify: `src/signal_ranking.py:116-166` (add historical similarity factor)
- Modify: `src/runner.py:116-120` (inject HistoryStore)

- [ ] **Step 1: Update runner.py to create HistoryStore**

Read current content first, then modify:

```python
# In runner.py, after line 116 (trade_history = TradeHistory(...))
# Add:
from data.history import HistoryStore

# After:
# trade_history = TradeHistory(data_dir / "trade_history.json")
# Add:
signal_history = HistoryStore(data_dir)
logger.info("Initialized signal history store")
```

- [ ] **Step 2: Update signal_ranking.py to use historical similarity**

Read current file, then add similarity-based scoring:

```python
# In signal_ranking.py, add import:
from data.history import HistoryStore
from data.similarity import SimilarityMatcher

# In SignalRanker.__init__, add:
def __init__(
    self,
    trade_history: TradeHistory,
    strategy_configs: Dict[str, StrategyConfig],
    top_k: int = 5,
    market_context: Optional[MarketContextProvider] = None,
    signal_history: Optional[HistoryStore] = None,  # NEW
) -> None:
    # ... existing init ...
    self._signal_history = signal_history
    self._similarity_matcher = SimilarityMatcher(signal_history) if signal_history else None

# In _score_signal method, add new component:
def _score_signal(self, strategy_name: str, signal: TradeSignal) -> Optional[SignalScore]:
    # ... existing code ...

    # NEW: Historical similarity score
    similarity_score = self._compute_similarity_score(signal)

    # Modify composite to include similarity (weight 0.1)
    composite = (
        weights["win_rate"] * win_rate_score
        + weights["rr"] * rr_score
        + weights["perf"] * perf_score
        + 0.1 * similarity_score  # NEW
        - context_penalty
    )

# Add new method:
def _compute_similarity_score(self, signal: TradeSignal) -> float:
    """Compute score based on historical similarity."""
    if not self._similarity_matcher:
        return 0.5  # Neutral if no history

    # Extract features from signal
    features = self._similarity_matcher.extract_features(signal, {})

    # Find similar historical signals
    similar = self._similarity_matcher.find_similar(features, top_k=10)

    # Compute historical win rate
    if not similar:
        return 0.5

    win_rate = self._similarity_matcher.compute_historical_win_rate(similar)

    logger.debug(
        "Similarity score | symbol={symbol} similar={count} hist_wr={wr:.2%}",
        symbol=signal.symbol,
        count=len(similar),
        wr=win_rate
    )

    return win_rate
```

- [ ] **Step 3: Run existing tests to ensure no regressions**

```bash
pytest tests/ -v -k "not integration"
```

- [ ] **Step 4: Commit**

```bash
git add src/runner.py src/signal_ranking.py
git commit -m "feat: integrate historical similarity into signal ranking (Module E.4)"
```

---

## Phase 2: Module D - Regime Adapter

### Task D.1: RegimeDetector for Market Classification

**Files:**
- Create: `src/regime/__init__.py`
- Create: `src/regime/detector.py`
- Create: `tests/test_regime_detector.py`

- [ ] **Step 1: Write tests for RegimeDetector**

```python
# tests/test_regime_detector.py
import pytest
from datetime import datetime, timezone
from regime.detector import RegimeDetector, RegimeType, RegimeResult

class TestRegimeDetector:
    def test_low_vol_bull(self):
        detector = RegimeDetector()
        market_data = {
            "vix_level": 12.0,
            "spy_vs_ma50": 0.05,  # 5% above MA50
            "qqq_vs_ma50": 0.03,
            "iwm_vs_ma50": 0.02,
        }

        result = detector.detect(market_data)

        assert result.regime == RegimeType.LOW_VOL_BULL
        assert result.confidence > 0.7

    def test_high_vol_bear(self):
        detector = RegimeDetector()
        market_data = {
            "vix_level": 28.0,
            "spy_vs_ma50": -0.08,  # 8% below MA50
            "qqq_vs_ma50": -0.10,
            "iwm_vs_ma50": -0.12,
        }

        result = detector.detect(market_data)

        assert result.regime == RegimeType.HIGH_VOL_BEAR
        assert result.confidence > 0.7

    def test_crush_regime(self):
        detector = RegimeDetector()
        market_data = {
            "vix_level": 40.0,
            "spy_vs_ma50": -0.15,
            "qqq_vs_ma50": -0.18,
            "iwm_vs_ma50": -0.20,
        }

        result = detector.detect(market_data)

        assert result.regime == RegimeType.CRUSH
        assert result.confidence > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_regime_detector.py -v
```

- [ ] **Step 3: Implement RegimeDetector**

```python
# src/regime/__init__.py
"""Regime detection and adaptation module."""
from .detector import RegimeDetector, RegimeType, RegimeResult

__all__ = ["RegimeDetector", "RegimeType", "RegimeResult"]
```

```python
# src/regime/detector.py
"""Market regime detection based on VIX and market state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from loguru import logger


class RegimeType(str, Enum):
    """Market regime classifications."""

    LOW_VOL_BULL = "LOW_VOL_BULL"
    NORMAL_BULL = "NORMAL_BULL"
    NORMAL_BEAR = "NORMAL_BEAR"
    HIGH_VOL_BEAR = "HIGH_VOL_BEAR"
    CRUSH = "CRUSH"
    TRANSITION_UP = "TRANSITION_UP"
    TRANSITION_DOWN = "TRANSITION_DOWN"


@dataclass(slots=True)
class RegimeResult:
    """Result of regime detection."""

    regime: RegimeType
    confidence: float
    as_of: datetime
    signals: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "as_of": self.as_of.isoformat(),
            **self.signals
        }


class RegimeDetector:
    """Detects market regime based on VIX and market state.

    Regime taxonomy:
    - LOW_VOL_BULL: VIX < 15, SPY > MA50
    - NORMAL_BULL: VIX 15-20, SPY > MA50
    - NORMAL_BEAR: VIX 15-20, SPY < MA50
    - HIGH_VOL_BEAR: VIX 20-35, SPY < MA50
    - CRUSH: VIX > 35 (any market direction)
    - TRANSITION_*: Detected via trend changes
    """

    def __init__(
        self,
        vix_low: float = 15.0,
        vix_high: float = 25.0,
        vix_extreme: float = 35.0
    ) -> None:
        """Initialize detector.

        Args:
            vix_low: Threshold for low volatility
            vix_high: Threshold for high volatility
            vix_extreme: Threshold for extreme/panic
        """
        self._vix_low = vix_low
        self._vix_high = vix_high
        self._vix_extreme = vix_extreme

    def detect(self, market_data: Dict[str, Any]) -> RegimeResult:
        """Detect current market regime.

        Args:
            market_data: Dict with keys:
                - vix_level: Current VIX
                - spy_vs_ma50: SPY % above/below MA50
                - qqq_vs_ma50: QQQ % above/below MA50
                - iwm_vs_ma50: IWM % above/below MA50

        Returns:
            RegimeResult with classification
        """
        vix = market_data.get("vix_level", 20.0)
        spy_pct = market_data.get("spy_vs_ma50", 0.0)
        qqq_pct = market_data.get("qqq_vs_ma50", 0.0)
        iwm_pct = market_data.get("iwm_vs_ma50", 0.0)

        # Average market breadth
        breadth = (spy_pct + qqq_pct + iwm_pct) / 3.0

        # Determine regime
        regime: RegimeType
        confidence: float
        signals = {
            "vix_level": vix,
            "breadth": breadth,
            "spy_pct": spy_pct,
        }

        # CRUSH regime (VIX > 35 overrides everything)
        if vix > self._vix_extreme:
            regime = RegimeType.CRUSH
            confidence = min(0.9 + (vix - self._vix_extreme) / 20.0, 1.0)

        # HIGH_VOL_BEAR
        elif vix > self._vix_high and breadth < 0:
            regime = RegimeType.HIGH_VOL_BEAR
            confidence = 0.7 + min(abs(breadth) * 2, 0.3)

        # LOW_VOL_BULL
        elif vix < self._vix_low and breadth > 0:
            regime = RegimeType.LOW_VOL_BULL
            confidence = 0.7 + min((self._vix_low - vix) / 10.0, 0.3)

        # NORMAL_BULL
        elif vix < self._vix_high and breadth > 0:
            regime = RegimeType.NORMAL_BULL
            confidence = 0.6

        # NORMAL_BEAR
        elif breadth < 0:
            regime = RegimeType.NORMAL_BEAR
            confidence = 0.6

        # Default
        else:
            regime = RegimeType.NORMAL_BEAR
            confidence = 0.5

        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            as_of=datetime.now(timezone.utc),
            signals=signals
        )

        logger.info(
            "Regime detected | regime={regime} confidence={conf:.2f} vix={vix} breadth={breadth:.2%}",
            regime=regime.value,
            conf=confidence,
            vix=vix,
            breadth=breadth
        )

        return result


__all__ = ["RegimeDetector", "RegimeType", "RegimeResult"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_regime_detector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/regime/__init__.py src/regime/detector.py tests/test_regime_detector.py
git commit -m "feat: add RegimeDetector for market classification (Module D.1)"
```

---

## Phase 3: Module A - Entry Filter

### Task A.1: PatternRecognizer for Technical Patterns

**Files:**
- Create: `src/entry/__init__.py`
- Create: `src/entry/patterns.py`
- Create: `tests/test_patterns.py`

- [ ] **Step 1: Write tests for PatternRecognizer**

```python
# tests/test_patterns.py
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from entry.patterns import PatternRecognizer, PatternSignal

class TestPatternRecognizer:
    def test_detect_breakout_above_high(self):
        recognizer = PatternRecognizer()

        # Create OHLCV data with 20-day breakout
        dates = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(30)]
        prices = [100.0] * 20 + [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
        volumes = [1000] * 20 + [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]

        df = pd.DataFrame({
            "close": prices,
            "volume": volumes,
        }, index=dates)
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
        df["open"] = df["close"]

        result = recognizer.detect_breakout(df, lookback=20)

        assert result is not None
        assert result.pattern_type == "BREAKOUT"
        assert result.confidence > 0.5

    def test_detect_consolidation_low_atr(self):
        recognizer = PatternRecognizer()

        # Create OHLCV data with low volatility (consolidation)
        dates = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(10)]
        prices = [100.0, 100.1, 99.9, 100.0, 100.2, 99.8, 100.1, 99.9, 100.0, 100.05]

        df = pd.DataFrame({"close": prices}, index=dates)
        df["high"] = df["close"] + 0.5
        df["low"] = df["close"] - 0.5
        df["open"] = df["close"]
        df["volume"] = 1000

        result = recognizer.detect_consolidation(df)

        assert result is not None
        assert result.pattern_type == "CONSOLIDATION"
```

- [ ] **Step 2: Implement PatternRecognizer**

```python
# src/entry/__init__.py
"""Entry filter module for signal quality enhancement."""
from .patterns import PatternRecognizer, PatternSignal

__all__ = ["PatternRecognizer", "PatternSignal"]
```

```python
# src/entry/patterns.py
"""Technical pattern recognition for entry timing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass(slots=True)
class PatternSignal:
    """Detected technical pattern."""

    pattern_type: str  # "BREAKOUT", "REVERSAL", "CONSOLIDATION"
    symbol: str
    timestamp: datetime
    confidence: float
    price_level: float


class PatternRecognizer:
    """Detects technical patterns in price data.

    Patterns:
    - BREAKOUT: Price above N-day high with volume confirmation
    - CONSOLIDATION: Low ATR relative to recent history
    - REVERSAL: RSI divergence + candlestick pattern
    """

    def detect_breakout(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20
    ) -> Optional[PatternSignal]:
        """Detect breakout above lookback high.

        Args:
            ohlcv: OHLCV DataFrame with columns: high, low, close, volume
            lookback: Number of days for high comparison

        Returns:
            PatternSignal if breakout detected, None otherwise
        """
        if len(ohlcv) < lookback + 5:
            return None

        # Calculate 20-day high
        rolling_high = ohlcv["high"].rolling(window=lookback).max()

        # Check if current price broke above high
        current_price = ohlcv["close"].iloc[-1]
        prev_high = rolling_high.iloc[-2] if len(rolling_high) > 1 else ohlcv["high"].iloc[-lookback:-1].max()

        if current_price <= prev_high:
            return None

        # Volume confirmation: current volume > 1.5x average
        avg_volume = ohlcv["volume"].iloc[-10:-1].mean()
        current_volume = ohlcv["volume"].iloc[-1]

        if current_volume < avg_volume * 1.5:
            return None

        # Calculate confidence
        breakout_pct = (current_price - prev_high) / prev_high
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        confidence = min(0.5 + breakout_pct * 10 + (volume_ratio - 1.5) * 0.2, 1.0)

        logger.debug(
            "Breakout detected | price={price} prev_high={high} breakout_pct={pct:.2%} volume_ratio={vol:.2f}",
            price=current_price,
            high=prev_high,
            pct=breakout_pct,
            vol=volume_ratio
        )

        return PatternSignal(
            pattern_type="BREAKOUT",
            symbol="",
            timestamp=datetime.now(),
            confidence=confidence,
            price_level=current_price
        )

    def detect_consolidation(
        self,
        ohlcv: pd.DataFrame,
        min_days: int = 5
    ) -> Optional[PatternSignal]:
        """Detect consolidation (low volatility) pattern.

        Args:
            ohlcv: OHLCV DataFrame
            min_days: Minimum days of consolidation

        Returns:
            PatternSignal if consolidation detected
        """
        if len(ohlcv) < min_days + 10:
            return None

        # Calculate ATR (simplified: high-low range)
        ohlcv = ohlcv.copy()
        ohlcv["range"] = ohlcv["high"] - ohlcv["low"]

        # Recent ATR vs 20-day ATR
        recent_atr = ohlcv["range"].iloc[-min_days:].mean()
        base_atr = ohlcv["range"].iloc[-30:-10].mean() if len(ohlcv) >= 30 else ohlcv["range"].iloc[:-min_days].mean()

        if base_atr <= 0:
            return None

        atr_ratio = recent_atr / base_atr

        # Consolidation: ATR < 50% of baseline
        if atr_ratio >= 0.5:
            return None

        confidence = 0.5 + (0.5 - atr_ratio)

        current_price = ohlcv["close"].iloc[-1]

        logger.debug(
            "Consolidation detected | atr_ratio={ratio:.2f} price={price}",
            ratio=atr_ratio,
            price=current_price
        )

        return PatternSignal(
            pattern_type="CONSOLIDATION",
            symbol="",
            timestamp=datetime.now(),
            confidence=min(confidence, 1.0),
            price_level=current_price
        )

    def detect_reversal(
        self,
        ohlcv: pd.DataFrame
    ) -> Optional[PatternSignal]:
        """Detect reversal pattern (RSI divergence + candlestick).

        Simplified implementation - checks for hammer/shooting star.

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            PatternSignal if reversal detected
        """
        if len(ohlcv) < 10:
            return None

        current = ohlcv.iloc[-1]
        prev = ohlcv.iloc[-2]

        # Hammer: small body, long lower shadow
        body = abs(current["close"] - current["open"])
        total_range = current["high"] - current["low"]
        lower_shadow = min(current["open"], current["close"]) - current["low"]
        upper_shadow = current["high"] - max(current["open"], current["close"])

        is_hammer = (
            total_range > 0 and
            body < total_range * 0.3 and
            lower_shadow > body * 2 and
            upper_shadow < body * 0.5
        )

        if is_hammer:
            # Bullish reversal after downtrend
            downtrend = current["close"] < prev["close"] < ohlcv.iloc[-3]["close"]
            if downtrend:
                return PatternSignal(
                    pattern_type="REVERSAL_BULLISH",
                    symbol="",
                    timestamp=datetime.now(),
                    confidence=0.6,
                    price_level=current["close"]
                )

        return None


__all__ = ["PatternRecognizer", "PatternSignal"]
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_patterns.py -v
git add src/entry/__init__.py src/entry/patterns.py tests/test_patterns.py
git commit -m "feat: add PatternRecognizer for technical patterns (Module A.1)"
```

---

## Phase 4: Module C - Exit Engine

### Task C.1: ExitConfig and ExitEngine Foundation

**Files:**
- Create: `src/exit/__init__.py`
- Create: `src/exit/config.py`
- Create: `tests/test_exit_config.py`

- [ ] **Step 1: Implement ExitConfig and ExitEngine**

```python
# src/exit/__init__.py
"""Exit engine module for systematic position management."""
from .config import ExitRule, ExitEngine

__all__ = ["ExitRule", "ExitEngine"]
```

```python
# src/exit/config.py
"""Exit configuration and engine foundation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


@dataclass(slots=True)
class ExitRule:
    """Exit rule configuration for a strategy."""

    strategy: str
    profit_target_pct: Optional[float] = 0.50
    stop_loss_pct: Optional[float] = None
    close_at_dte: int = 5
    trailing_stop_delta: Optional[float] = None
    use_ai_advisor: bool = False


class ExitEngine:
    """Systematic exit management engine.

    Evaluates positions against configured exit rules and
    generates exit recommendations.
    """

    def __init__(
        self,
        rules: List[ExitRule],
        default_profit_target: float = 0.50,
        default_close_dte: int = 5
    ) -> None:
        """Initialize exit engine.

        Args:
            rules: Per-strategy exit rules
            default_profit_target: Default profit target %
            default_close_dte: Default DTE for closing
        """
        self._rules: Dict[str, ExitRule] = {r.strategy: r for r in rules}
        self._default_profit_target = default_profit_target
        self._default_close_dte = default_close_dte

        logger.info(
            "ExitEngine initialized | rules={count}",
            count=len(rules)
        )

    def get_rule(self, strategy: str) -> ExitRule:
        """Get exit rule for strategy.

        Args:
            strategy: Strategy name

        Returns:
            ExitRule (defaults if not configured)
        """
        if strategy in self._rules:
            return self._rules[strategy]

        # Return default rule
        return ExitRule(
            strategy=strategy,
            profit_target_pct=self._default_profit_target,
            close_at_dte=self._default_close_dte
        )

    def evaluate(
        self,
        position: object,  # PositionGroup from portfolio/evaluation.py
        snapshot: Optional[object] = None
    ) -> Optional[str]:
        """Evaluate position for exit.

        Args:
            position: Position object to evaluate
            snapshot: Current market snapshot

        Returns:
            Exit recommendation string or None
        """
        # Placeholder - full implementation in C.2-C.5
        # Will integrate:
        # - DynamicTargetCalculator
        # - TimeExitEvaluator
        # - TrailingStopManager
        # - AIExitAdvisor

        return None


__all__ = ["ExitRule", "ExitEngine"]
```

- [ ] **Step 2: Commit**

```bash
git add src/exit/__init__.py src/exit/config.py tests/test_exit_config.py
git commit -m "feat: add ExitEngine foundation (Module C.1)"
```

---

## Final Integration Tasks

### Task: Add Configuration to config.yaml

**Files:**
- Modify: `config.yaml` (add module configurations)

- [ ] **Step 1: Add configuration sections**

Read config.yaml, then add:

```yaml
# Add after existing sections:

# Signal Quality Enhancement Modules
data_enhancer:
  history:
    enabled: true
    storage_dir: "./data"
  options_flow:
    enabled: true
    volume_oi_threshold: 2.0
  similarity:
    enabled: true
    top_k: 10
    half_life_days: 30

regime_adapter:
  enabled: true
  vix_low: 15.0
  vix_high: 25.0
  vix_extreme: 35.0

entry_filter:
  enabled: true
  confluence:
    min_aligned: 3
  patterns:
    enabled: true
    min_confidence: 0.5

exit_engine:
  enabled: true
  default_profit_target: 0.50
  default_close_dte: 5
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "config: add signal quality enhancement module settings"
```

---

## Testing Checklist

Before considering implementation complete:

- [ ] All unit tests pass: `pytest tests/test_*.py -v`
- [ ] Integration tests pass (if configured)
- [ ] No mypy errors: `uv run mypy src/`
- [ ] No lint errors: `uv run ruff check src/`

---

**Plan complete.** Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch subagent per task for fast parallel execution
2. **Inline Execution** - Execute tasks in this session with checkpoints

Which approach?
