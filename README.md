# Option Scanner

This project scans option chains using IBKR market data, applies configured strategies, and outputs trade signals.

## Configuration

Update `config.yaml` to match your environment:

- `ibkr`: Connection details for the IBKR Gateway or TWS.
- `tickers`: Symbols to monitor.
- `schedule`: Daily trigger time. The default runs every day at 7:00 AM Pacific (`America/Los_Angeles`).
- `email`: SMTP configuration for sending the formatted results. Set `enabled` to `true` once valid credentials and recipient addresses are provided.

Signals are written to the `./results` directory with timestamps. When email notifications are enabled a CSV copy of each run is attached to the message alongside an HTML summary table.

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
