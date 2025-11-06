"""Main entry point for the option scanner using NautilusTrader and IBKR."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import pkgutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from enum import Enum

from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from ib_insync import IB, Option, Stock
from loguru import logger

from logging_utils import configure_logging
from strategies.base import BaseOptionStrategy, TradeSignal
from ai_agents import SignalExplainAgent, SignalValidationAgent
from notifications import SlackNotifier
from scheduling import compute_next_run, parse_schedule_times


MARKET_DATA_TYPE_CODES = {
    "LIVE": 1,
    "FROZEN": 2,
    "DELAYED": 3,
    "DELAYED_FROZEN": 4,
}


class RunMode(str, Enum):
    """Supported execution modes for the scanner."""

    LOCAL_IMMEDIATE = "local"
    DOCKER_IMMEDIATE = "docker-immediate"
    DOCKER_SCHEDULED = "docker-scheduled"


@dataclass(slots=True)
class OptionChainSnapshot:
    """Container for option chain data for a given symbol."""

    symbol: str
    underlying_price: float
    timestamp: datetime
    options: List[Dict[str, Any]]

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(self.options)
        if df.empty:
            return df
        df["symbol"] = self.symbol
        df["underlying_price"] = self.underlying_price
        df["timestamp"] = self.timestamp
        df["expiry"] = pd.to_datetime(df["expiry"])
        return df


class BaseDataFetcher:
    """Protocol-like base class for data fetchers."""

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:  # pragma: no cover - interface
        raise NotImplementedError


class IBKRDataFetcher(BaseDataFetcher):
    """Handles IBKR data retrieval using ib_insync with Nautilus compatibility."""

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        data_dir: Path,
        market_data_type: str,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.data_dir = data_dir
        self.market_data_type = market_data_type.upper()
        if self.market_data_type not in MARKET_DATA_TYPE_CODES:
            raise ValueError(
                f"Unsupported market data type '{market_data_type}'. "
                f"Choose one of {sorted(MARKET_DATA_TYPE_CODES)}."
            )
        self._market_data_type_code = MARKET_DATA_TYPE_CODES[self.market_data_type]
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ib = IB()
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            if self._ib.isConnected():
                return
            logger.info(
                "Connecting to IBKR Gateway host={host} port={port} client_id={client_id}",
                host=self.host,
                port=self.port,
                client_id=self.client_id,
            )
            await self._ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=5)
            if not self._ib.isConnected():
                raise ConnectionError("Failed to connect to IBKR Gateway")
            self._ib.reqMarketDataType(self._market_data_type_code)
            logger.info(
                "Set IBKR market data type to {market_data_type}",
                market_data_type=self.market_data_type,
            )
            logger.info("Successfully connected to IBKR Gateway")

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    async def fetch_option_chain(self, symbol: str) -> OptionChainSnapshot:
        await self.connect()
        contract = Stock(symbol, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        ticker = await self._ib.reqMktDataAsync(contract, "", False, False)
        underlying_price = float(ticker.last or ticker.close or ticker.marketPrice() or 0.0)

        params = await self._ib.reqSecDefOptParamsAsync(
            contract.symbol,
            "",
            contract.secType,
            contract.conId,
        )
        if not params:
            raise RuntimeError(f"No option parameters returned for {symbol}")
        # Select SMART exchange chain with most strikes
        chain = max(params, key=lambda p: len(p.strikes))
        target_expiries = sorted(chain.expirations)[:4]
        strikes = sorted(chain.strikes, key=lambda x: abs(x - underlying_price))[:40]
        options: List[Contract] = []
        for expiry in target_expiries:
            for strike in strikes:
                options.append(Option(symbol, expiry, strike, "C", "SMART", currency="USD"))
                options.append(Option(symbol, expiry, strike, "P", "SMART", currency="USD"))
        qualified = await self._ib.qualifyContractsAsync(*options)
        if not qualified:
            raise RuntimeError(f"Unable to qualify options for {symbol}")

        tickers = await self._ib.reqTickersAsync(*qualified)
        rows: List[Dict[str, Any]] = []
        now = datetime.utcnow()
        for ticker in tickers:
            contract = ticker.contract
            if not isinstance(contract, Option):
                continue
            expiry_dt = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")
            rows.append(
                {
                    "symbol": symbol,
                    "expiry": expiry_dt,
                    "strike": float(contract.strike),
                    "option_type": "CALL" if contract.right == "C" else "PUT",
                    "bid": float(ticker.bid or 0.0),
                    "ask": float(ticker.ask or 0.0),
                    "mark": float(ticker.midpoint() or 0.0),
                    "delta": getattr(ticker.modelGreeks, "delta", 0.0) if ticker.modelGreeks else 0.0,
                    "theta": getattr(ticker.modelGreeks, "theta", 0.0) if ticker.modelGreeks else 0.0,
                    "implied_volatility": getattr(ticker.modelGreeks, "impliedVol", 0.0)
                    if ticker.modelGreeks
                    else 0.0,
                }
            )
        snapshot = OptionChainSnapshot(
            symbol=symbol,
            underlying_price=underlying_price,
            timestamp=now,
            options=rows,
        )
        self._persist_snapshot(snapshot)
        return snapshot

    def _persist_snapshot(self, snapshot: OptionChainSnapshot) -> None:
        df = snapshot.to_pandas()
        if df.empty:
            return
        timestamp_str = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
        file_path = self.data_dir / f"{snapshot.symbol}_{timestamp_str}.parquet"
        df.to_parquet(file_path, index=False)
        logger.info("Saved option snapshot to {path}", path=str(file_path))

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:
        tasks = [self.fetch_option_chain(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: List[OptionChainSnapshot] = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.exception("Failed to fetch data for {symbol}", symbol=symbol)
                continue
            snapshots.append(result)
        return snapshots


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_strategies() -> List[BaseOptionStrategy]:
    strategy_dir = Path(__file__).parent / "strategies"
    strategies: List[BaseOptionStrategy] = []
    for module_info in pkgutil.iter_modules([str(strategy_dir)]):
        if not module_info.name.startswith("strategy_"):
            continue
        module = importlib.import_module(f"strategies.{module_info.name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseOptionStrategy)
                and obj is not BaseOptionStrategy
            ):
                try:
                    strategies.append(obj())
                except TypeError as exc:
                    logger.error(
                        "Failed to instantiate strategy {name}: {error}",
                        name=obj.__name__,
                        error=exc,
                    )
    logger.info("Loaded {count} strategies", count=len(strategies))
    return strategies


async def run_once(
    fetcher: "BaseDataFetcher",
    strategies: List[BaseOptionStrategy],
    symbols: Iterable[str],
    results_dir: Path,
    *,
    explain_agent: Optional[SignalExplainAgent] = None,
    validation_agent: Optional[SignalValidationAgent] = None,
    slack_notifier: Optional[SlackNotifier] = None,
) -> None:
    snapshots = await fetcher.fetch_all(symbols)
    aggregated_signals: List[tuple[str, TradeSignal]] = []
    for strategy in strategies:
        try:
            signals = strategy.on_data(snapshots)
            for signal in signals:
                aggregated_signals.append((strategy.name, signal))
        except Exception as exc:
            logger.exception("Strategy {name} failed", name=strategy.name)
    if not aggregated_signals:
        logger.info("No trade signals generated in this iteration")
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    explain_agent = explain_agent or SignalExplainAgent()
    validation_agent = validation_agent or SignalValidationAgent()
    snapshot_by_symbol = {snapshot.symbol: snapshot for snapshot in snapshots}
    rows = []
    signals_only = [signal for _strategy_name, signal in aggregated_signals]
    for strategy_name, signal in aggregated_signals:
        snapshot = snapshot_by_symbol.get(signal.symbol)
        explanation = explain_agent.explain(signal, snapshot)
        validation = validation_agent.review(signal, snapshot, signals_only)
        row = signal.__dict__.copy()
        row.update(
            {
                "explanation": explanation,
                "validation": validation,
                "strategy": strategy_name,
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = results_dir / f"signals_{timestamp}.csv"
    df.to_csv(file_path, index=False)
    logger.info("Saved {count} signals to {path}", count=len(df), path=str(file_path))

    if slack_notifier:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, slack_notifier.send_signals, df, file_path)


async def run_scheduler(
    config: Dict[str, Any],
    fetcher: "BaseDataFetcher",
    strategies: List[BaseOptionStrategy],
    symbols: Sequence[str],
    results_dir: Path,
    slack_notifier: Optional[SlackNotifier],
) -> None:
    schedule_config = config.get("schedule", {})
    scheduled_times = parse_schedule_times(schedule_config)
    timezone_name = schedule_config.get("timezone", "America/Los_Angeles")

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        logger.warning("Invalid timezone '{timezone}', defaulting to UTC", timezone=timezone_name)
        tz = ZoneInfo("UTC")

    logger.info(
        "Scheduling runs at {times} ({timezone})",
        times=", ".join(scheduled_time.strftime("%H:%M") for scheduled_time in scheduled_times),
        timezone=getattr(tz, "key", str(tz)),
    )

    explain_agent = SignalExplainAgent()
    validation_agent = SignalValidationAgent()

    while True:
        now = datetime.now(tz)
        next_run = compute_next_run(now, scheduled_times)
        sleep_seconds = max((next_run - now).total_seconds(), 0.0)
        logger.info(
            "Next run scheduled at {next_time} (sleeping {seconds:.2f}s)",
            next_time=next_run.isoformat(),
            seconds=sleep_seconds,
        )
        await asyncio.sleep(sleep_seconds)
        start = datetime.now(tz)
        logger.info("Starting scheduled run at {start}", start=start.isoformat())
        try:
            await run_once(
                fetcher,
                strategies,
                symbols,
                results_dir,
                explain_agent=explain_agent,
                validation_agent=validation_agent,
                slack_notifier=slack_notifier,
            )
        except Exception as exc:
            logger.exception("Run failed: {error}", error=exc)
        duration = (datetime.now(tz) - start).total_seconds()
        logger.info("Run completed in {duration:.2f}s", duration=duration)


class DockerControllerError(RuntimeError):
    """Raised when Docker orchestration fails."""


class DockerController:
    """Utility for managing docker-compose services required by the scanner."""

    def __init__(self, compose_file: Path) -> None:
        self.compose_file = compose_file

    def start_service(self, service: str) -> None:
        command = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
            "up",
            "-d",
            service,
        ]
        logger.info("Starting docker service {service} via docker compose", service=service)
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise DockerControllerError("docker command not found. Install Docker to use this mode.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else ""
            raise DockerControllerError(
                f"Failed to start docker service '{service}': {stderr.strip()}"
            ) from exc


class LocalDataFetcher(BaseDataFetcher):
    """Loads pre-recorded option data from disk for offline testing."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, self._load_snapshot, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: List[OptionChainSnapshot] = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.exception("Failed to load local data for {symbol}", symbol=symbol)
                continue
            snapshots.append(result)
        return snapshots

    def _load_snapshot(self, symbol: str) -> OptionChainSnapshot:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Local data directory '{self.data_dir}' does not exist")
        pattern = f"{symbol}_*.parquet"
        matches = sorted(self.data_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No local snapshot found for {symbol}. Expected files matching {pattern} in {self.data_dir}"
            )
        latest = matches[-1]
        df = pd.read_parquet(latest)
        if df.empty:
            raise ValueError(f"Local snapshot {latest} is empty")
        timestamp = pd.to_datetime(df["timestamp"].iloc[0])
        underlying_price = float(df["underlying_price"].iloc[0])
        options = df.drop(columns=[col for col in ("symbol", "underlying_price", "timestamp") if col in df.columns])
        return OptionChainSnapshot(
            symbol=symbol,
            underlying_price=underlying_price,
            timestamp=timestamp.to_pydatetime(),
            options=options.to_dict(orient="records"),
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nautilus option scanner.")
    parser.add_argument(
        "--run-mode",
        choices=[mode.value for mode in RunMode],
        default=RunMode.LOCAL_IMMEDIATE.value,
        help="Select how the scanner executes: local (no Docker), docker-immediate, or docker-scheduled.",
    )
    parser.add_argument(
        "--market-data",
        choices=sorted(MARKET_DATA_TYPE_CODES.keys()),
        default="DELAYED_FROZEN",
        help="Market data type requested from IBKR when using live data fetchers.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path("docker-compose.yml"),
        help="Path to the docker-compose file used when starting Docker services.",
    )
    parser.add_argument(
        "--docker-service",
        default="ibkr-gateway",
        help="Name of the docker-compose service that hosts the IBKR gateway.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file for the scanner.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    log_dir = Path(config.get("log_dir", "./logs"))
    configure_logging(log_dir, "strategy_signals")

    results_dir = Path(config.get("results_dir", "./results"))
    slack_notifier = SlackNotifier(config.get("slack"))
    strategies = discover_strategies()
    symbols: Sequence[str] = config.get("tickers", [])
    run_mode = RunMode(args.run_mode)

    explain_agent = SignalExplainAgent()
    validation_agent = SignalValidationAgent()

    if run_mode is RunMode.LOCAL_IMMEDIATE:
        fetcher = LocalDataFetcher(Path(config.get("data_dir", "./data")))
        try:
            asyncio.run(
                run_once(
                    fetcher,
                    strategies,
                    symbols,
                    results_dir,
                    explain_agent=explain_agent,
                    validation_agent=validation_agent,
                    slack_notifier=slack_notifier,
                )
            )
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        return

    docker_controller = DockerController(Path(args.compose_file))
    try:
        docker_controller.start_service(args.docker_service)
    except DockerControllerError as exc:
        logger.error("Unable to start Docker service: {error}", error=exc)
        raise SystemExit(1) from exc

    fetcher = IBKRDataFetcher(
        host=config["ibkr"]["host"],
        port=config["ibkr"]["port"],
        client_id=config["ibkr"].get("client_id", 1),
        data_dir=Path(config.get("data_dir", "./data")),
        market_data_type=args.market_data,
    )

    try:
        if run_mode is RunMode.DOCKER_IMMEDIATE:
            asyncio.run(
                run_once(
                    fetcher,
                    strategies,
                    symbols,
                    results_dir,
                    explain_agent=explain_agent,
                    validation_agent=validation_agent,
                    slack_notifier=slack_notifier,
                )
            )
        else:
            asyncio.run(
                run_scheduler(
                    config,
                    fetcher,
                    strategies,
                    symbols,
                    results_dir,
                    slack_notifier,
                )
            )
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")


if __name__ == "__main__":
    main()
