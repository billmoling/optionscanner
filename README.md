# optionscanner

Optionscanner is a lightweight toolkit for monitoring option chains and portfolio exposure using the Interactive Brokers (IBKR) API.

## IBKR gateway via Docker

The `docker-compose.yml` file now focuses solely on the IBKR Gateway container. Run the gateway in Docker, keep the scanner on the host (or Raspberry Pi), and connect over the published ports. This mirrors the production deployment while keeping local development simple.

1. **Populate `.env`.**  
   Docker Compose reads your credentials and runtime preferences from `.env`. Set at least:

   ```dotenv
   TWS_USERID=your-ibkr-username
   TWS_PASSWORD=your-ibkr-password
   IAPI_CLIENT_ID=1
   TRADING_MODE=paper   # use "live" only when you intend to trade live
   ```

   Every variable in `.env` is forwarded to `ghcr.io/gnzsnz/ib-gateway:stable`, so you can opt into advanced behaviors (auto restarts, read-only API, etc.) without editing the compose file.

2. **Start or rebuild the gateway container.**

```bash
docker compose up -d ib-gateway
```

   The service publishes `127.0.0.1:4002` → container `4004` (paper trading API), `127.0.0.1:4001` → `4003` (live trading API), and `127.0.0.1:5900` for VNC/2FA.

3. **Complete the interactive login.**  
   Attach a VNC client to `localhost:5900` and follow the prompts (password is documented in the upstream image). After IBKR finishes loading, you can close VNC; the container keeps running and reconnects on failure according to the environment settings.

4. **Point the scanner at the container.**  
   Set the connection details via `.env` so each host (developer laptop, Raspberry Pi, etc.) can override them without editing `config.yaml`:

   ```dotenv
   IBKR_HOST=127.0.0.1
   IBKR_PORT=4002    # 4001 when TRADING_MODE=live
   IBKR_CLIENT_ID=1
   ```

   The `ibkr` block in `config.yaml` now defers to those environment variables by default, but you can still hard-code overrides in the YAML if needed.

5. **Stop the gateway when finished.**

   ```bash
   docker compose down
   ```

   (Use `docker compose logs -f ib-gateway` to monitor gateway output while it runs.)

## Running the scanner

The scanner supports three execution modes that control how market data is sourced and whether Docker is required. All modes share the same configuration file (`config.yaml`) and logging/output directories.

Install dependencies locally with `pip install -r requirements.txt` before running any of the commands below.

### 1. Immediate local run (no Docker)

Use this when you have captured option chain snapshots on disk and want to test strategy and Gemini integrations without starting the IBKR Gateway.

```bash
python main.py --run-mode local --market-data FROZEN --config config.yaml
```

Place snapshot files (e.g., `NVDA_20240101_120000.parquet`) under the `data_dir` configured in `config.yaml`. The run finishes after processing the locally stored data once.

### 2. Scheduled runs (loop on configured times)

Use this when you want the scanner (or just the portfolio workflow) to run automatically at the times listed under `schedule.times` in `config.yaml` (respects `schedule.timezone`).

```bash
python main.py --run-mode schedule --config config.yaml
```

The process stays alive; ensure your Docker container remains running to honor the schedule. Combine with `--portfolio-only` to skip signal generation and run only the portfolio manager on the same schedule.

### CLI reference

| Flag | Description |
| --- | --- |
| `--run-mode {local,schedule}` | Select the execution mode (single run or scheduled loop). |
| `--config PATH` | Path to the YAML configuration file (defaults to `config.yaml`). |
| `--market-data TYPE` | IBKR market data type (`LIVE` or `FROZEN`). |
| `--portfolio-only` | Skip signal generation and run only the portfolio manager workflow. |

### Portfolio-only mode

Use `--portfolio-only` when you only need the portfolio risk workflow (positions, Greeks, risk checks, Slack summary) without fetching new option signals. The flag honors the selected `--run-mode`, so Docker helpers still start when requested, and it overrides the `DISABLE_PORTFOLIO_MANAGER` environment variable.

```bash
# Single-run portfolio workflow
python main.py --portfolio-only --run-mode local --config config.yaml

# Scheduled portfolio workflow (runs at times in config.yaml:schedule.times)
python main.py --portfolio-only --run-mode schedule --config config.yaml
```

Combine the flag with your preferred workflow to ensure the IBKR gateway container is running before the portfolio manager connects.

## Signal output and Gemini usage

Each run writes a timestamped CSV under `results/` (for example, `signals_20251110_184922.csv`). The default column order is `symbol, expiry, strike, option_type, strategy, direction, rationale` and the optional `explanation`/`validation` fields are appended only when Gemini output is enabled so files stay compact when AI summaries are disabled.

Gemini calls for explanations/validation can be disabled globally via `enable_gemini` in `config.yaml`:

```yaml
enable_gemini: false
```

When the flag is `false`, the scanner skips Google Gemini requests altogether—no explanation or validation text is generated, and those columns are omitted from the CSV/Slack output—while continuing to export strategy results and Slack notifications.

## Strategy overrides in `config.yaml`

All classes in `strategies/strategy_*.py` are auto-discovered, but you can now tune or disable them directly from `config.yaml` via the `strategies` block. Each key can be the class name (`PutCreditSpreadStrategy`) or the camel-cased variant (`put_credit_spread_strategy`). Set `enabled: false` to skip a strategy entirely or provide a `params` mapping to override constructor arguments. For example, to target a credit-heavy, large-cap universe in mild uptrends:

```yaml
strategies:
  PutCreditSpreadStrategy:
    params:
      min_days_to_expiry: 18
      max_days_to_expiry: 38
      spread_width: 5.0
      min_credit: 0.6
      short_target_delta: 0.23
      short_otm_pct: 0.04
  VerticalSpreadStrategy:
    params:
      max_iv_rank: 0.30
      min_risk_reward_ratio: 1.1
      target_days_to_expiry: 35
  IronCondorStrategy:
    params:
      allowed_symbols: ["SPY", "QQQ", "IWM", "NVDA"]
      spread_width: 10.0
  CoveredCallStrategy:
    params:
      enabled: true        # strategy-level switch inside the class
      min_annualized_yield: 0.15
  PoorMansCoveredCallStrategy:
    params:
      max_trade_ideas: 3
```

Set `enabled: false` at the top level (next to `params`) to skip instantiating a strategy entirely, and use the nested `params.enabled` knob for classes such as `CoveredCallStrategy` that internally gate their execution. During `main.py` start-up the overrides are applied before `run_once` executes, so the scheduled runner and the portfolio workflow see the updated configuration automatically.

## Exit monitoring and portfolio mapping

Each strategy signal now populates a lightweight `position_cache.json` file under `results/`. The cache stores the originating strategy, contract details, rationale, and last-seen timestamp so live IBKR positions can be matched back to their strategy. On every scanner run the cache evaluates simple exit rules (for example, short strike breaches or DTE thresholds) using the latest snapshots and writes any recommendations to `results/exits_YYYYMMDD_HHMMSS.csv`. Feed those CSVs into your portfolio workflow—or reconcile them with actual IBKR positions—to decide when to harvest profits, roll, or cut risk without rewriting the portfolio manager.

Need bespoke criteria? Register your own exit evaluator by calling `PositionCache.register_evaluator("MY_DIRECTION", func)` before the run; each evaluator receives the cached entry, the latest snapshot, and `datetime.utcnow()` so you can encode custom roll triggers per strategy.

## AI-driven execution and portfolio automation

The manual Slack-only flow (`generate signal → notify Slack → run portfolio analysis → notify Slack`) can be upgraded to a fully automated, AI-supervised workflow while still broadcasting every decision to Slack for visibility. Enable the automation block in `config.yaml` to switch the orchestrator into hands-off mode:

```yaml
automation:
  enabled: true            # master kill switch; disable to fall back to Slack-only
  trade_execution:
    enabled: true          # turn AI finalists into IBKR orders
    default_quantity: 1
    limit_padding_pct: 0.05
    max_spread_pct: 0.6    # skip fills when the quoted spread is wider than this
  portfolio_execution:
    enabled: true          # act on Gemini portfolio responses
    max_positions: 3       # cap auto-closes per run
    use_market_orders: false
```

1. **Signal finalist → execution.** After strategies emit candidate signals and the AI agent selects a finalist, the AI output is handed to a **Trade Executor** service that translates the suggestion into IBKR order objects (contract lookup, side, quantity, limit/stop params, time-in-force, account). The executor calls the IBKR API immediately, returning execution status/IDs to the orchestrator and posting a Slack summary with links to order IDs.
2. **Portfolio suggestion → execution.** The portfolio monitor streams positions/Greeks to the AI agent. When the AI returns rebalance/hedge instructions (rolls, closes, delta hedges, cash targets), a **Portfolio Executor** converts each instruction into the required IBKR calls (modify/cancel existing orders, adjust allocations, open hedges) and runs them in sequence or batch. Slack receives a pre-trade summary and a post-trade status update.

### Component and sequence changes

* **Orchestrator enhancements:** Add a light orchestration layer (async task queue or event bus) that chains `Signal → AI finalist → Trade Executor` and `Portfolio snapshot → AI recommendation → Portfolio Executor` without human intervention. Persist correlation IDs so Slack, AI responses, and IBKR order tickets are traceable.
* **Execution adapters:** Implement dedicated adapters for IBKR (order placement, order modification, portfolio info) decoupled from AI payload formats. Include schema validation and unit normalization so AI outputs cannot emit malformed IBKR orders.
* **State + idempotency:** Store execution state (pending, submitted, filled, failed) and AI decision metadata in a durable store to avoid duplicate orders on retries/restarts. Gate replays by checking recent fills against the correlation ID and contract hash.
* **Guardrails:** Enforce pre-trade risk checks (max notional/Greeks, concentration, trading window, paper vs. live mode) before any IBKR call. Reject or downsize trades that violate policy even if the AI requested them.
* **Error handling and fallbacks:** Surface IBKR rejections/timeouts back to the orchestrator for retries or AI re-planning. Keep Slack alerts for each failure with actionable context (contract, reason, retry schedule).
* **Observability:** Emit structured logs/metrics (latency per step, IBKR error codes, AI latency) and capture full audit trails (AI prompt/response, final order payload, IBKR orderId) for compliance.

### Integration challenges to plan for

* **IBKR session management:** Ensure the gateway is healthy and authenticated before firing orders; auto-reconnect and block new executions when the session is degraded.
* **Latency and sequencing:** AI calls can be slower than order routing; use timeouts and freshness checks so signals/portfolio snapshots are still valid when the IBKR call occurs.
* **Position parity:** Continuously reconcile live IBKR positions with the local `position_cache` so the AI and executors act on consistent data, especially after partial fills or manual interventions.
* **Permissioning and safety switches:** Keep a global "circuit breaker" and per-strategy kill switches to revert to Slack-only mode if anomalies are detected. Maintain paper/live separation with explicit configuration and environment overrides.

## Running tests

The project ships with focused unit tests for AI agents, strategy logic, and Slack notifications. Execute the full suite with:

```bash
pytest
```

You can target specific modules when iterating on a component:

```bash
pytest tests/test_ai_agents.py
pytest tests/test_strategies.py -k "VerticalSpread"
pytest tests/test_slack_notifier.py::SlackNotifierTests::test_each_signal_sends_individual_message
```

Unit tests rely on built-in fixtures and mock data, so no live Gemini, Slack, or IBKR services are required. The optional integration tests below exercise the real services.

### Slack integration test

If you want to verify end-to-end Slack delivery, an optional integration test is available. The test loads `.env` automatically and posts a single message through your real webhook when `SLACK_WEBHOOK_URL` is defined. To run it:

```bash
# Ensure .env contains SLACK_WEBHOOK_URL=<your-webhook> (and optionally SLACK_TEST_CHANNEL)
python -m unittest tests.test_slack_notifier_integration
```

Set `SLACK_TEST_CHANNEL` in `.env` if you need to override the default channel for integration runs. The test is skipped automatically when no webhook is configured.

### Gemini integration test

To confirm Gemini prompts work end-to-end, provide `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in `.env` and run:

```bash
python -m unittest tests.test_ai_agents_integration
```

The test uses your live Gemini access to generate an explanation for a sample trade signal and prints the response to the console. It is skipped automatically when the API key or `google-generativeai` package is missing.

### Gemini smoke test (genai SDK)

If you want a minimal sanity check of the newer `google-genai` SDK, install `google-genai` and run with stdout uncaptured so the inline prints are visible:

```bash
python -m pytest tests/test_gemini_call.py -q --capture=no
```

The test loads `.env`, uses `GOOGLE_API_KEY` or `GEMINI_API_KEY`, and makes a single live request to `gemini-2.5-flash`, echoing the raw response text to the console.

### IBKR integration test

Start the IBKR Gateway via Docker Compose (for example, `docker compose up -d ib-gateway`) or connect to an existing gateway, ensure `.env` contains `TWS_USERID`, `TWS_PASSWORD`, `IAPI_CLIENT_ID`, and `TRADING_MODE`, then run:

```bash
python -m unittest tests.test_ibkr_integration
```

The test connects to the gateway, requests the NVDA stock price over live or frozen market data, and prints the captured quote. Override the default connection settings with `IBKR_HOST`, `IBKR_PORT`, or `IBKR_MARKET_DATA_TYPE` if needed. The port automatically follows `TRADING_MODE` (`4002` for paper, `4001` for live) unless `IBKR_PORT` is provided.

If you also want to confirm option-chain access end-to-end, run:

```bash
python -m unittest tests.test_ibkr_options_integration
```

The option-chain test connects to the same gateway, resolves the next NVDA expiration, and fetches a single near-the-money call quote. It uses the identical `.env` configuration and skips automatically when the gateway credentials are absent.

### Local snapshot integration test

To sanity-check every strategy against your archived snapshots (without IBKR/Gemini), convert the CSV history under `historydata/` into temporary Parquet files and run all strategies via the local data fetcher.

Requirements:

- Install `pyarrow` (`pip install pyarrow`) so LocalDataFetcher can emit Parquet snapshots.
- Ensure `historydata/*.csv` exists (generated by previous live runs) and `results/` is writable.

Then execute:

```bash
pytest tests/test_local_data_integration.py -k test_historydata_run_all_strategies
```

The test will:

1. Convert each `historydata/*.csv` file into Parquet snapshots inside a pytest-provided temp directory.
2. Run `discover_strategies()` against those snapshots using `run_once()` (Gemini disabled).
3. Write a fresh `results/signals_YYYYMMDD_HHMMSS.csv` and copy the newest run to `results/signals_localtest.csv`.

The test skips automatically when `pyarrow` is missing or no history CSVs are available. After it passes, inspect `results/signals_localtest.csv` to review the combined output for tuning.
