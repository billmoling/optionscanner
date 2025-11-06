# Option Scanner

This project scans option chains using IBKR market data, applies configured strategies, and outputs trade signals.

## Configuration

Update `config.yaml` to match your environment:

- `ibkr`: Connection details for the IBKR Gateway or TWS.
- `tickers`: Symbols to monitor.
- `schedule`: Daily trigger times. Provide one or more `HH:MM` entries in `times` (e.g., `06:30`, `10:00`, `12:15`) and set the `timezone` (defaults to `America/Los_Angeles`).
- `slack`: Incoming webhook configuration for delivering formatted results. Set `enabled` to `true` once a valid webhook URL (and optional username/channel overrides) are configured.

Signals are written to the `./results` directory with timestamps. When Slack notifications are enabled, the run summary (with up to the first 10 signals by default) is posted to the configured channel and references the CSV saved locally inside the container.

## Running Locally

Install dependencies and run the scheduler directly:

```bash
pip install -r requirements.txt
python main.py --mode testing
```

The `--mode` flag selects between delayed-frozen data (`testing`) and live market data (`live`).

## Docker

A Dockerfile is provided for containerized execution:

```bash
docker build -t optionscanner .
```

Supply configuration and secrets at runtime. For example, mount a customized config file and run in testing mode (default):

```bash
docker run \
  --mount type=bind,source="$(pwd)/config.yaml",target=/app/config.yaml,readonly \
  optionscanner
```

To use live market data, override the default command arguments:

```bash
docker run \
  --mount type=bind,source="$(pwd)/config.yaml",target=/app/config.yaml,readonly \
  optionscanner --mode live
```
