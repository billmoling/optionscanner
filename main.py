"""Main entry point for the option scanner using NautilusTrader and IBKR."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import pkgutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from ib_insync import IB, Option, Stock
from loguru import logger

from logging_utils import configure_logging
from strategies.base import BaseOptionStrategy, TradeSignal
from ai_agents import SignalExplainAgent, SignalValidationAgent
from notifications import SlackNotifier


MARKET_DATA_TYPE_CODES = {
    "LIVE": 1,
    "FROZEN": 2,
    "DELAYED": 3,
    "DELAYED_FROZEN": 4,
}


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


class IBKRDataFetcher:
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
    fetcher: IBKRDataFetcher,
    strategies: List[BaseOptionStrategy],
    symbols: Iterable[str],
    results_dir: Path,
    slack_notifier: Optional[SlackNotifier] = None,
) -> None:
    snapshots = await fetcher.fetch_all(symbols)
    aggregated_signals: List[TradeSignal] = []
    for strategy in strategies:
        try:
            signals = strategy.on_data(snapshots)
            aggregated_signals.extend(signals)
        except Exception as exc:
            logger.exception("Strategy {name} failed", name=strategy.name)
    if not aggregated_signals:
        logger.info("No trade signals generated in this iteration")
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    explain_agent = SignalExplainAgent()
    validation_agent = SignalValidationAgent()
    snapshot_by_symbol = {snapshot.symbol: snapshot for snapshot in snapshots}
    rows = []
    for signal in aggregated_signals:
        snapshot = snapshot_by_symbol.get(signal.symbol)
        explanation = explain_agent.explain(signal, snapshot)
        validation = validation_agent.review(signal, snapshot, aggregated_signals)
        row = signal.__dict__.copy()
        row.update(
            {
                "explanation": explanation,
                "validation": validation,
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


async def run_scheduler(config: Dict[str, Any], market_data_type: str) -> None:
    log_dir = Path(config.get("log_dir", "./logs"))
    configure_logging(log_dir, "strategy_signals")
    data_dir = Path(config.get("data_dir", "./data"))
    fetcher = IBKRDataFetcher(
        host=config["ibkr"]["host"],
        port=config["ibkr"]["port"],
        client_id=config["ibkr"].get("client_id", 1),
        data_dir=data_dir,
        market_data_type=market_data_type,
    )
    strategies = discover_strategies()
    symbols = config.get("tickers", [])
    results_dir = Path("./results")
    slack_notifier = SlackNotifier(config.get("slack"))

    schedule_config = config.get("schedule", {})
    schedule_time_str = str(schedule_config.get("time", "07:00"))
    timezone_name = schedule_config.get("timezone", "America/Los_Angeles")

    try:
        scheduled_time = datetime.strptime(schedule_time_str, "%H:%M").time()
    except ValueError:
        logger.warning("Invalid schedule time '{time}', defaulting to 07:00", time=schedule_time_str)
        scheduled_time = datetime.strptime("07:00", "%H:%M").time()

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        logger.warning("Invalid timezone '{timezone}', defaulting to UTC", timezone=timezone_name)
        tz = ZoneInfo("UTC")

    def compute_next_run(now: datetime) -> datetime:
        target = now.replace(
            hour=scheduled_time.hour,
            minute=scheduled_time.minute,
            second=0,
            microsecond=0,
        )
        if now >= target:
            target += timedelta(days=1)
        return target

    next_run: Optional[datetime] = None
    while True:
        now = datetime.now(tz)
        if next_run is None or now >= next_run:
            next_run = compute_next_run(now)
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
                slack_notifier=slack_notifier,
            )
        except Exception as exc:
            logger.exception("Run failed: {error}", error=exc)
        duration = (datetime.now(tz) - start).total_seconds()
        logger.info("Run completed in {duration:.2f}s", duration=duration)
        next_run = None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nautilus option scanner.")
    parser.add_argument(
        "--mode",
        choices=["testing", "live"],
        default="testing",
        help="Choose 'testing' for delayed frozen data (default) or 'live' for real-time market data.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    market_data_type = "DELAYED_FROZEN" if args.mode == "testing" else "LIVE"
    config = load_config(Path("config.yaml"))
    try:
        asyncio.run(run_scheduler(config, market_data_type))
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")


if __name__ == "__main__":
    main()
