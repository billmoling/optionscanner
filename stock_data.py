"""Utilities for downloading underlying stock data via IBKR."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from ib_async import IB, Stock
from loguru import logger


MARKET_DATA_TYPES = {"LIVE": 1, "FROZEN": 2}


@dataclass
class HistoricalDataRequest:
    """Configuration for an IBKR historical data request."""

    duration: str = "90 D"
    bar_size: str = "1 day"
    what_to_show: str = "ADJUSTED_LAST"
    use_rth: bool = True
    end_time: Optional[datetime] = None


class StockDataFetcher:
    """Downloads historical stock data directly from IBKR."""

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        *,
        market_data_type: str = "FROZEN",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.exchange = exchange
        self.currency = currency
        market_data_type = market_data_type.upper()
        if market_data_type not in MARKET_DATA_TYPES:
            raise ValueError(f"Unsupported market data type: {market_data_type}")
        self.market_data_type = market_data_type
        self._ib = IB()
        self._lock = asyncio.Lock()
        self._market_data_type_code = MARKET_DATA_TYPES[market_data_type]
        self._connected = False

    async def connect(self) -> None:
        """Establish a connection to IBKR if not already connected."""

        async with self._lock:
            if self._connected:
                return
            logger.info(
                "Connecting StockDataFetcher to IBKR | host={host} port={port} client_id={client_id}",
                host=self.host,
                port=self.port,
                client_id=self.client_id,
            )
            await self._ib.connectAsync(self.host, self.port, clientId=self.client_id, timeout=5)
            if not self._ib.isConnected():
                raise ConnectionError("Failed to connect to IBKR Gateway")
            self._ib.reqMarketDataType(self._market_data_type_code)
            self._connected = True
            logger.info("StockDataFetcher connected to IBKR")

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False

    async def fetch_history(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Download OHLCV bars for ``symbol`` and return a DataFrame."""

        await self.connect()
        request = self._build_request(kwargs)
        contract = Stock(symbol, self.exchange, self.currency)
        end_time = request.end_time.strftime("%Y%m%d %H:%M:%S") if request.end_time else ""
        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_time,
            durationStr=request.duration,
            barSizeSetting=request.bar_size,
            whatToShow=request.what_to_show,
            useRTH=1 if request.use_rth else 0,
            formatDate=1,
        )
        frame = self._bars_to_frame(symbol, bars)
        if frame.empty:
            logger.warning("No historical data returned | symbol={symbol}", symbol=symbol)
        return frame

    async def fetch_history_many(self, symbols: Iterable[str], **kwargs: Any) -> Dict[str, pd.DataFrame]:
        """Download history for multiple symbols sequentially."""

        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                results[symbol] = await self.fetch_history(symbol, **kwargs)
            except Exception as exc:  # pragma: no cover - network errors
                logger.exception("Failed to download stock data | symbol={symbol}", symbol=symbol)
                results[symbol] = pd.DataFrame()
        return results

    def _build_request(self, overrides: Dict[str, str]) -> HistoricalDataRequest:
        request = HistoricalDataRequest()
        for key, value in overrides.items():
            if not hasattr(request, key):
                raise AttributeError(f"HistoricalDataRequest has no field '{key}'")
            setattr(request, key, value)
        return request

    @staticmethod
    def _bars_to_frame(symbol: str, bars: List[Any]) -> pd.DataFrame:
        rows = []
        for bar in bars or []:
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": pd.to_datetime(getattr(bar, "date", None), utc=True),
                    "open": float(getattr(bar, "open", float("nan"))),
                    "high": float(getattr(bar, "high", float("nan"))),
                    "low": float(getattr(bar, "low", float("nan"))),
                    "close": float(getattr(bar, "close", float("nan"))),
                    "volume": float(getattr(bar, "volume", float("nan"))),
                }
            )
        frame = pd.DataFrame(rows)
        return frame


__all__ = ["StockDataFetcher", "HistoricalDataRequest"]
