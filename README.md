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

### 2. Immediate Docker-backed run

Start the IBKR Gateway via Docker Compose, fetch live market data once from the host-based scanner, and exit. This mode simply guarantees the `ib-gateway` container is running before the local process connects, which makes host and Raspberry Pi workflows identical.

```bash
python main.py --run-mode docker-immediate --config config.yaml \
  --compose-file docker-compose.yml --docker-service ib-gateway \
  --market-data FROZEN
```

The command automatically launches (or reuses) the `ib-gateway` service from the compose file before running the scanner on your host. Adjust `--market-data` if your IBKR account permits real-time feeds.

### 3. Scheduled Docker run

Run the scanner continuously on the configured schedule while keeping the `ib-gateway` container alive in the background.

```bash
python main.py --run-mode docker-scheduled --config config.yaml
```

Scheduled run times are controlled via the `schedule` section of `config.yaml`. The process stays alive on the host, restarts the gateway container if it fails, and sleeps between runs based on the configured window.

### CLI reference

| Flag | Description |
| --- | --- |
| `--run-mode {local,docker-immediate,docker-scheduled}` | Select the execution mode. |
| `--config PATH` | Path to the YAML configuration file (defaults to `config.yaml`). |
| `--compose-file PATH` | Docker Compose file used to start services in Docker modes. |
| `--docker-service NAME` | Service name for the IBKR Gateway inside the compose file. |
| `--market-data TYPE` | IBKR market data type (`LIVE` or `FROZEN`). |
| `--portfolio-only` | Skip signal generation and run only the portfolio manager workflow. |

### Portfolio-only mode

Use `--portfolio-only` when you only need the portfolio risk workflow (positions, Greeks, risk checks, Slack summary) without fetching new option signals. The flag honors the selected `--run-mode`, so Docker helpers still start when requested, and it overrides the `DISABLE_PORTFOLIO_MANAGER` environment variable.

```bash
python main.py --portfolio-only --run-mode local --config config.yaml
```

You can combine the flag with Docker-backed modes (e.g., `--run-mode docker-immediate`) to ensure the IBKR gateway container is running before the portfolio manager connects.

## Signal output and Gemini usage

Each run writes a timestamped CSV under `results/` (for example, `signals_20251110_184922.csv`). The default column order is `symbol, expiry, strike, option_type, strategy, direction, rationale` and the optional `explanation`/`validation` fields are appended only when Gemini output is enabled so files stay compact when AI summaries are disabled.

Gemini calls for explanations/validation can be disabled globally via `enable_gemini` in `config.yaml`:

```yaml
enable_gemini: false
```

When the flag is `false`, the scanner skips Google Gemini requests altogether—no explanation or validation text is generated, and those columns are omitted from the CSV/Slack output—while continuing to export strategy results and Slack notifications.

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
