# Signal Ranking System

## Overview

The signal ranking system replaces the AI-based selection with a deterministic composite scoring algorithm that ranks trade signals based on:

1. **Win Rate** - Published win rate transitioning to live performance
2. **Risk/Reward** - Expected return relative to risk
3. **Live Performance** - Historical P&L and consistency

Top 5 ranked signals are sent to Slack daily.

---

## Composite Score Formula

```
composite_score = (win_rate_w * win_rate_score) +
                  (rr_w * rr_score) +
                  (perf_w * perf_score)
```

### Weight Phases

| Phase | Trade Count | Win Rate Weight | R/R Weight | Performance Weight |
|-------|-------------|-----------------|------------|-------------------|
| Initial | < 30 trades | 0.5 | 0.3 | 0.2 |
| Mature | >= 30 trades | 0.2 | 0.3 | 0.5 |

---

## Configuration

### Published Win Rates (`config.yaml`)

Add `published_win_rate` to each strategy:

```yaml
strategies:
  VerticalSpreadStrategy:
    params:
      # ... existing params
    published_win_rate: 0.60  # 60% published win rate
  PutCreditSpreadStrategy:
    params:
      # ... existing params
    published_win_rate: 0.70
```

**Recommended starting values:**
- Credit spreads: 0.65-0.75 (higher win rate, lower R/R)
- Debit spreads: 0.55-0.65 (lower win rate, higher R/R)
- Directional strategies: 0.50-0.60
- Neutral strategies (Iron Condor): 0.60-0.70

---

## Trade History Tracking

### File Location

Trade history is persisted to `data/trade_history.json`.

### Submitting Trade Results

Drop JSON files into `data/trade_results/` directory. Files are automatically processed on next run.

**Single trade:**
```json
{
  "strategy": "VerticalSpreadStrategy",
  "symbol": "NVDA",
  "direction": "BULL_CALL_DEBIT_SPREAD",
  "entry_date": "2026-03-21T10:00:00Z",
  "entry_price": 2.50,
  "exit_date": "2026-03-25T15:30:00Z",
  "exit_price": 4.80,
  "pnl": 2.30,
  "status": "CLOSED_WIN",
  "quantity": 1,
  "notes": "Reached max profit target"
}
```

**Multiple trades (array):**
```json
[
  { ... trade 1 ... },
  { ... trade 2 ... }
]
```

### Status Values

| Status | Description |
|--------|-------------|
| `OPEN` | Position still open |
| `CLOSED_WIN` | Closed for profit |
| `CLOSED_LOSS` | Closed for loss |
| `CLOSED_EVEN` | Closed at breakeven |

### File Naming Convention

```
trade_YYYYMMDD_symbol_strategy.json
Example: trade_20260321_NVDA_VerticalSpread.json
```

---

## Slack Output Format

Ranked signals are sent as a formatted table:

```
*Daily Option Scanner Signals* | 2026-03-21 10:00 UTC

| # | Symbol | Direction | Strategy | Score | Reason |
|---|--------|-----------|----------|-------|--------|
| 1 | NVDA | Bull | Vertical Spread | 0.78 | High Live win rate 72%; favorable R/R |
| 2 | AAPL | Bear | Put Credit Spread | 0.71 | Moderate Blend win rate; acceptable R/R |
| 3 | MSFT | Bull | Poor Mans Covered Call | 0.65 | Stable live perf (67% win) |
```

---

## Component Details

### `trade_history.py`

- `TradeHistory` - Persistent trade storage and statistics
- `TradeResult` - Single trade record
- `StrategyStats` - Aggregated strategy metrics

### `signal_ranking.py`

- `SignalRanker` - Composite scoring engine
- `SignalScore` - Ranking result with component scores
- `StrategyConfig` - Strategy configuration (published win rate)

### Integration Points

1. **runner.py** - Initializes ranker, ranks signals before Slack
2. **notifications/slack.py** - `send_ranked_signals()` formats table output
3. **strategies/base.py** - `TradeSignal` includes `risk_reward_ratio` field

---

## Migration from Gemini Selection

The ranking system replaces `SignalBatchSelector` for signal selection. Gemini can still be used for:
- Portfolio management decisions
- Trade validation commentary
- Market analysis

To fully disable Gemini for selection:
```yaml
enable_gemini: false  # Uses pure ranking only
```

---

## Monitoring

Check ranking effectiveness:
1. Review `data/trade_history.json` for accumulated results
2. Monitor strategy win rates transitioning from published to live
3. Compare ranked signal performance vs random selection

---

## Troubleshooting

**Issue:** No signals sent to Slack
- Check `slack.enabled: true` in config
- Verify trade history directory exists: `data/trade_results/`
- Check logs for ranking errors

**Issue:** All scores look the same
- Ensure strategies have `published_win_rate` configured
- Check signal `risk_reward_ratio` is being set by strategies

**Issue:** Trade results not loading
- Verify JSON format matches template
- Check file is in `data/trade_results/` directory
- Ensure file extension is `.json`
