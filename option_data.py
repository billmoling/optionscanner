"""Data fetching utilities for the option scanner."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from ib_async import Contract, IB, Option, Stock
from loguru import logger
from zoneinfo import ZoneInfo


MARKET_DATA_TYPE_CODES = {
    "LIVE": 1,
    "FROZEN": 2,
}
MARKET_DATA_CODE_TO_NAME = {code: name for name, code in MARKET_DATA_TYPE_CODES.items()}

OPTION_EXCHANGE_OVERRIDES: Dict[str, str] = {
    # Hard-coded exchange preferences for frequently scanned symbols.
    "NVDA": "SMART",
    "AAPL": "CBOE",
    "TSLA": "CBOE",
    "META": "CBOE",
    "AMZN": "NASDAQOM",
    "MSFT": "CBOE",
    "GOOG": "CBOE",
    "NFLX": "CBOE",
    "AMD": "CBOE",
    "JPM": "BOX",
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


class BaseDataFetcher:
    """Protocol-like base class for data fetchers."""

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:  # pragma: no cover - interface
        raise NotImplementedError


class IBKRDataFetcher(BaseDataFetcher):
    """Handles IBKR data retrieval using ib_async with Nautilus compatibility."""

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
        self.history_dir = Path("historydata")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._history_timezone = ZoneInfo("America/Los_Angeles")
        self._ib = IB()
        self._lock = asyncio.Lock()
        self._max_expiries = 4
        self._max_strikes_per_side = 12
        self._current_market_data_type_code: Optional[int] = None

    def _set_market_data_type(self, code: int, *, reason: Optional[str] = None) -> None:
        if self._current_market_data_type_code == code:
            return
        self._ib.reqMarketDataType(code)
        self._current_market_data_type_code = code
        name = MARKET_DATA_CODE_TO_NAME.get(code, str(code))
        if reason:
            logger.warning(
                "Set IBKR market data type to {market_data_type} ({reason})",
                market_data_type=name,
                reason=reason,
            )
        else:
            logger.info(
                "Set IBKR market data type to {market_data_type}",
                market_data_type=name,
            )

    async def _request_tickers(self, *contracts: Contract) -> List[Any]:
        try:
            self._set_market_data_type(self._market_data_type_code)
            if hasattr(self._ib, "reqTickersAsync"):
                tickers = await self._ib.reqTickersAsync(*contracts)
            else:
                tickers = self._ib.reqTickers(*contracts)
            return list(tickers)
        except Exception as exc:  # ib_async raises IBError; fallback to generic for compatibility
            error_code = getattr(exc, "errorCode", None)
            if error_code == 354:
                raise RuntimeError(
                    "IBKR rejected the market data request (error 354). "
                    "Ensure your account is entitled to the requested LIVE or FROZEN feed."
                ) from exc
            raise

    async def _request_contract_details(self, contract: Contract) -> List[Any]:
        if hasattr(self._ib, "reqContractDetailsAsync"):
            return await self._ib.reqContractDetailsAsync(contract)
        return list(self._ib.reqContractDetails(contract))

    async def _request_underlying_ticker(self, contract: Contract) -> Any:
        if hasattr(self._ib, "reqMktDataAsync"):
            try:
                self._set_market_data_type(self._market_data_type_code)
                return await self._ib.reqMktDataAsync(contract, "", True, False)
            except Exception as exc:
                error_code = getattr(exc, "errorCode", None)
                if error_code == 354:
                    raise RuntimeError(
                        f"IBKR rejected market data for {contract.symbol} (error 354). "
                        "Verify that LIVE or FROZEN data permissions are enabled for this symbol."
                    ) from exc
                raise
        tickers = await self._request_tickers(contract)
        if not tickers:
            raise RuntimeError(f"No market data returned for contract {contract}")
        return tickers[0]

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
            self._set_market_data_type(self._market_data_type_code)
            logger.info("Successfully connected to IBKR Gateway")

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    @property
    def ib(self) -> IB:
        """Return the underlying ib_async client instance."""
        return self._ib

    async def fetch_option_chain(self, symbol: str) -> OptionChainSnapshot:
        await self.connect()
        contract = Stock(symbol, "SMART", "USD")
        await self._ib.qualifyContractsAsync(contract)
        ticker = await self._request_underlying_ticker(contract)
        underlying_price = float(ticker.last or ticker.close or ticker.marketPrice() or 0.0)

        params = await self._ib.reqSecDefOptParamsAsync(
            contract.symbol,
            "",
            contract.secType,
            contract.conId,
        )
        if not params:
            raise RuntimeError(f"No option parameters returned for {symbol}")
        chain = max(params, key=lambda p: len(p.strikes))
        override_exchange = OPTION_EXCHANGE_OVERRIDES.get(symbol)
        chain_exchange = chain.exchange or ""
        target_exchange = override_exchange or chain_exchange or "CBOE"
        trading_class = getattr(chain, "tradingClass", "") or ""
        candidate_exchanges = [
            override_exchange,
            chain_exchange,
            target_exchange,
            "SMART",
            "CBOE",
            "BOX",
            "NASDAQOM",
            "ARCA",
            "BATSOP",
        ]
        preferred_exchanges: List[str] = []
        for exchange in candidate_exchanges:
            if exchange and exchange not in preferred_exchanges:
                preferred_exchanges.append(exchange)
        target_expiries = sorted(chain.expirations)[: self._max_expiries]
        selected_contracts: Dict[int, Option] = {}

        for expiry in target_expiries:
            details: Optional[List[Any]] = None
            for exchange in preferred_exchanges:
                base_contract = Option(symbol, expiry, 0.0, "", exchange, currency="USD")
                if trading_class and exchange not in ("SMART", ""):
                    base_contract.tradingClass = trading_class
                try:
                    details = await self._request_contract_details(base_contract)
                except Exception as exc:
                    logger.opt(exception=exc).warning(
                        "Failed to load contract details for {symbol} expiry {expiry} on {exchange}",
                        symbol=symbol,
                        expiry=expiry,
                        exchange=exchange,
                    )
                    continue
                if details:
                    break
            if not details:
                logger.warning(
                    "No option contract details returned for {symbol} expiry {expiry} on exchanges {exchanges}",
                    symbol=symbol,
                    expiry=expiry,
                    exchanges=", ".join(preferred_exchanges),
                )
                continue

            candidates_by_side: Dict[str, Dict[int, Option]] = {"C": {}, "P": {}}
            for detail in details:
                option_contract = getattr(detail, "contract", None)
                if not isinstance(option_contract, Option):
                    continue
                if trading_class and option_contract.tradingClass and option_contract.tradingClass != trading_class:
                    continue
                if option_contract.right not in candidates_by_side:
                    continue
                candidates_by_side[option_contract.right][option_contract.conId] = option_contract

            for right, candidates in candidates_by_side.items():
                if not candidates:
                    continue
                sorted_candidates = sorted(
                    candidates.values(),
                    key=lambda c: (abs(float(c.strike) - underlying_price), c.strike),
                )[: self._max_strikes_per_side]
                for option_contract in sorted_candidates:
                    selected_contracts[option_contract.conId] = option_contract

        contracts = list(selected_contracts.values())
        if not contracts:
            raise RuntimeError(f"No option contracts selected for {symbol}")

        qualified = await self._ib.qualifyContractsAsync(*contracts)
        if not qualified:
            raise RuntimeError(f"Unable to qualify options for {symbol}")

        tickers = await self._request_tickers(*qualified)
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
                    "gamma": getattr(ticker.modelGreeks, "gamma", 0.0) if ticker.modelGreeks else 0.0,
                    "vega": getattr(ticker.modelGreeks, "vega", 0.0) if ticker.modelGreeks else 0.0,
                    "theta": getattr(ticker.modelGreeks, "theta", 0.0) if ticker.modelGreeks else 0.0,
                    "rho": getattr(ticker.modelGreeks, "rho", 0.0) if ticker.modelGreeks else 0.0,
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
        df = df.copy()
        df["price"] = df.get("mark", 0.0)
        timestamp_utc = snapshot.timestamp.replace(tzinfo=timezone.utc)
        timestamp_str = timestamp_utc.strftime("%Y%m%d_%H%M%S")
        file_path = self.data_dir / f"{snapshot.symbol}_{timestamp_str}.parquet"
        df.to_parquet(file_path, index=False)
        logger.info("Saved option snapshot to {path}", path=str(file_path))

        timestamp_local = timestamp_utc.astimezone(self._history_timezone)
        df["timestamp"] = timestamp_local.isoformat()
        history_path = self.history_dir / f"{timestamp_local.strftime('%Y%m%d')}.csv"
        header = not history_path.exists()
        df.to_csv(history_path, mode="a", header=header, index=False)
        logger.info("Appended option snapshot to history file {path}", path=str(history_path))

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:
        tasks = [self.fetch_option_chain(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: List[OptionChainSnapshot] = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.opt(exception=result).error(
                    "Failed to fetch data for {symbol}: {error}",
                    symbol=symbol,
                    error=result,
                )
                continue
            snapshots.append(result)
        return snapshots


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
                logger.opt(exception=result).error(
                    "Failed to load local data for {symbol}: {error}",
                    symbol=symbol,
                    error=result,
                )
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


__all__ = [
    "BaseDataFetcher",
    "IBKRDataFetcher",
    "LocalDataFetcher",
    "OptionChainSnapshot",
    "MARKET_DATA_TYPE_CODES",
    "MARKET_DATA_CODE_TO_NAME",
]
