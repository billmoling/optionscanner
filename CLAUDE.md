# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Optionscanner is a Python-based options trading toolkit that monitors option chains and portfolio exposure using the Interactive Brokers (IBKR) API. Built on NautilusTrader and ib_async, it supports automated signal generation, AI-driven trade selection via Google Gemini, and optional automated execution.

## Python Virtual Environment

**Always check and use the Python virtual environment before running Python commands.**

This project uses a `.venv` virtual environment managed by `uv`. Before running any Python commands:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Or use uv run to execute commands in the venv
uv run <command>
```

### Examples

```bash
# Running tests
uv run pytest

# Running the application
uv run python src/main.py

# Running mypy
uv run mypy src/

# Installing dependencies
uv pip install <package>
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run scanner (single run with frozen data)
python main.py --run-mode local --market-data FROZEN

# Run on schedule
python main.py --run-mode schedule
```

## Architecture

### Core Modules

- **`main.py`**: Entry point with CLI for run modes (`local`, `schedule`) and `--portfolio-only` flag
- **`runner.py`**: Execution orchestration (`run_once`, `run_scheduler`)
- **`option_data.py`**: IBKR option chain fetching via `IBKRDataFetcher`; `OptionChainSnapshot` container
- **`execution.py`**: Trade/portfolio execution adapters (`TradeExecutor`, `PortfolioActionExecutor`)
- **`portfolio/manager.py`**: Coordinates position loading, Greeks computation, risk evaluation, and Slack notifications

### Strategy Framework

- **`strategies/base.py`**: `BaseOptionStrategy` (extends NautilusTrader `Strategy`), `TradeSignal`, `SignalLeg`
- **`strategies/strategy_*.py`**: Concrete strategies (PutCreditSpread, VerticalSpread, IronCondor, CoveredCall, PMCC, VixFearFade, TqqqQqqRotation)
- Strategies are auto-discovered at runtime; can be enabled/disabled/tuned via `config.yaml:strategies` block

### Data Flow

```
IBKR Gateway → IBKRDataFetcher.fetch_all() → OptionChainSnapshot[]
              ↓
        Strategies.on_data() → TradeSignal[]
              ↓
        SignalBatchSelector (Gemini) → Finalists
              ↓
        TradeExecutor.execute_finalists() → IBKR Orders
```

### Market State System

- **`market_state.py`**: Classifies underlying stocks as BULL/BEAR/UPTREND/DOWNTREND using MA crossover logic
- **`technical_indicators.py`**: `TechnicalIndicatorProcessor` with registry pattern for SMA/RSI
- Market state is computed from stock history and passed to strategies via `MarketStateProvider`

### Position Cache & Exit Monitoring

- **`position_cache.py`**: Persists signals to `results/position_cache.json` for tracking entries
- Evaluates exit rules (DTE thresholds, strike breaches) and writes recommendations to `results/exits_*.csv`

## Configuration

### `config.yaml` Key Sections

```yaml
tickers: [...]                          # Symbols to scan
ibkr:                                   # IBKR connection (host, port, client_id)
strategies:                             # Per-strategy overrides
  PutCreditSpreadStrategy:
    params:
      min_days_to_expiry: 18
      max_days_to_expiry: 38
automation:                             # AI-driven execution toggle
  enabled: true
  trade_execution:
    enabled: true
    max_orders: 5
schedule:
  times: ["06:30", "10:00", ...]
  timezone: "America/Los_Angeles"
```

### Environment Variables (via `.env`)

- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`: Connection details
- `TWS_USERID`, `TWS_PASSWORD`, `TRADING_MODE`: For Docker IBKR gateway
- `GOOGLE_API_KEY` / `GEMINI_API_KEY`: For Gemini AI calls
- `SLACK_WEBHOOK_URL`: For notifications

## IBKR Gateway via Docker

```bash
# Start gateway
docker compose up -d ib-gateway

# Gateway ports: 4002 (paper), 4001 (live), 5900 (VNC)
```

The scanner connects to the gateway over the network; configure `ibkr.host`/`ibkr.port` in `config.yaml` or via environment.

## Testing

```bash
# Full test suite
pytest

# Specific modules
pytest tests/test_strategies.py -k "VerticalSpread"
pytest tests/test_ai_agents.py

# Integration tests (requires .env setup)
python -m unittest tests.test_ibkr_integration
python -m unittest tests.test_slack_notifier_integration
```

Unit tests use mocks; integration tests require live services (IBKR gateway, Slack webhook, Gemini API key).

## Common Patterns

### Strategy Implementation

Strategies extend `BaseOptionStrategy` and implement `on_data()`:
- Input: `Iterable[Any]` of `OptionChainSnapshot`
- Output: `List[TradeSignal]` with legs for multi-leg orders
- Use `self.emit_signal()` for logging

### Adding Technical Indicators

```python
indicator_processor.register("ma50", TechnicalIndicatorProcessor.simple_moving_average(50))
```

### Executing Trades

`TradeExecutor.execute_finalists()` accepts `(strategy_name, TradeSignal, score, reason)` tuples and submits IBKR orders with configurable limit padding and spread checks.

## Directories

- `strategies/`: Strategy implementations
- `portfolio/`: Risk management (greeks, rules, playbooks)
- `tests/`: Unit and integration tests
- `results/`: Generated signals, exit recommendations, position cache
- `data/`: Parquet snapshots from live runs
- `historydata/`: CSV history for backtesting/local runs
- `logs/`: Application logs
