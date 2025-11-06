# optionscanner

Optionscanner is a lightweight toolkit for monitoring option chains and portfolio exposure using the Interactive Brokers (IBKR) API.

## Docker usage

The repository ships with a `docker-compose.yml` file that starts both the IBKR Gateway and the scanner in a shared network. Follow the steps below to bring the stack online:

1. **Create credentials file (optional but recommended).**
   Create a `.env` file in the project root so Docker Compose can read your credentials and gateway preferences. At minimum, populate the following values:

   ```dotenv
   TWS_USERID=your-ibkr-username
   TWS_PASSWORD=your-ibkr-password
   IAPI_CLIENT_ID=1
   TRADING_MODE=paper
   ```

   The variables are forwarded directly to the `ghcr.io/ibkrcampus/ibkr-gateway` image. Refer to the image documentation for the full list of supported options (for example, 2FA timeout or region-specific settings).

2. **Prepare persistent storage for the gateway.**
   Create directories that will be mounted into the gateway container so that session settings and logs survive container restarts:

   ```bash
   mkdir -p gateway/config gateway/logs
   ```

   The compose file mounts these folders at `/home/ibkr/.ibgateway` and `/home/ibkr/logs`, respectively. They are excluded from version control via `gateway/.gitignore` so they can safely hold runtime data.

3. **Review application configuration.**
   The default `config.yaml` now targets the gateway container hostname:

   ```yaml
   ibkr:
     host: "ibkr-gateway"
   ```

   If you run the scanner outside of Docker or connect to a remote gateway, update `ibkr.host` to the appropriate hostname or IP address.

4. **Start the stack.**
   Build and launch both services in the foreground:

   ```bash
   docker compose up --build
   ```

   Docker Compose creates a user-defined bridge network named `ibkr-net` so the scanner can reach the gateway at `ibkr-gateway:4002`. Gateway ports `4001`/`4002` are published to the host for API access, and `5900` exposes the VNC session required to complete IBKR logins and 2FA from a VNC client. Logs from both containers appear in the same terminal. Use `Ctrl+C` to stop the stack.

5. **Complete the IBKR login.**
   Connect a VNC client to `localhost:5900` (password is provided by the gateway image documentation) to approve the interactive login or supply 2FA codes when prompted. The session only needs to remain open long enough for the gateway to finish initialization.

6. **Run in the background (optional).**
   To run detached, use `docker compose up --build -d`. Tail the combined logs with `docker compose logs -f`.

7. **Clean up.**
   Stop the services and remove the containers with `docker compose down`. Persistent gateway data remains inside the `gateway/` directory.

## Running the scanner

The scanner supports three execution modes that control how market data is sourced and whether Docker is required. All modes share the same configuration file (`config.yaml`) and logging/output directories.

Install dependencies locally with `pip install -r requirements.txt` before running any of the commands below.

### 1. Immediate local run (no Docker)

Use this when you have captured option chain snapshots on disk and want to test strategy and Gemini integrations without starting the IBKR Gateway.

```bash
python main.py --run-mode local --config config.yaml
```

Place snapshot files (e.g., `NVDA_20240101_120000.parquet`) under the `data_dir` configured in `config.yaml`. The run finishes after processing the locally stored data once.

### 2. Immediate Docker-backed run

Start the IBKR Gateway via Docker Compose, fetch live market data once, and exit. This mode is useful for quick validation against the live gateway without scheduling repeated scans.

```bash
python main.py --run-mode docker-immediate --config config.yaml \
  --compose-file docker-compose.yml --docker-service ibkr-gateway \
  --market-data DELAYED_FROZEN
```

The command automatically launches the gateway container defined in the compose file before running the scanner. Adjust `--market-data` if your IBKR account permits real-time feeds.

### 3. Scheduled Docker run

Run the scanner continuously on the configured schedule while managing the IBKR Gateway container lifecycle for you.

```bash
python main.py --run-mode docker-scheduled --config config.yaml
```

Scheduled run times are controlled via the `schedule` section of `config.yaml`. The process stays alive and sleeps between runs based on the configured time window.

### CLI reference

| Flag | Description |
| --- | --- |
| `--run-mode {local,docker-immediate,docker-scheduled}` | Select the execution mode. |
| `--config PATH` | Path to the YAML configuration file (defaults to `config.yaml`). |
| `--compose-file PATH` | Docker Compose file used to start services in Docker modes. |
| `--docker-service NAME` | Service name for the IBKR Gateway inside the compose file. |
| `--market-data TYPE` | IBKR market data type (`LIVE`, `FROZEN`, `DELAYED`, or `DELAYED_FROZEN`). |

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

Each test file uses built-in fixtures and mock data, so no live Gemini, Slack, or IBKR services are required.

### Slack integration test

If you want to verify end-to-end Slack delivery, an optional integration test is available. The test loads `.env` automatically and posts a single message through your real webhook when `SLACK_WEBHOOK_URL` is defined. To run it:

```bash
# Ensure .env contains SLACK_WEBHOOK_URL=<your-webhook> (and optionally SLACK_TEST_CHANNEL)
python -m unittest tests.test_slack_notifier_integration
```

Set `SLACK_TEST_CHANNEL` in `.env` if you need to override the default channel for integration runs. The test is skipped automatically when no webhook is configured.
