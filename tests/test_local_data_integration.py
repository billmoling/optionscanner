import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import pytest

from main import discover_strategies
from option_data import LocalDataFetcher
from runner import run_once
from technical_indicators import TechnicalIndicatorProcessor


def _convert_history_to_parquet(source: Path, destination: Path) -> List[str]:
    destination.mkdir(parents=True, exist_ok=True)
    symbols = []
    for csv_path in sorted(source.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty or "symbol" not in df.columns:
            continue
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        for symbol, subset in df.groupby("symbol"):
            target = destination / f"{symbol}_localtest.parquet"
            subset.to_parquet(target, index=False)
            symbols.append(symbol)
    return sorted(set(symbols))


def _build_price_map(parquet_dir: Path, symbols: Sequence[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for symbol in symbols:
        parquet_path = parquet_dir / f"{symbol}_localtest.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue
        price_column = "underlying_price" if "underlying_price" in df.columns else "close"
        if price_column not in df.columns:
            continue
        value = df[price_column].dropna().iloc[0]
        try:
            prices[symbol] = float(value)
        except (TypeError, ValueError):
            continue
    return prices


class _SyntheticStockFetcher:
    """Fabricated stock fetcher used for integration tests."""

    def __init__(self, price_map: Dict[str, float]) -> None:
        self._price_map = price_map

    async def fetch_history_many(self, symbols, **kwargs):
        histories: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            histories[symbol] = self._build_history(symbol)
        return histories

    def _build_history(self, symbol: str) -> pd.DataFrame:
        base_price = self._price_map.get(symbol)
        if base_price is None:
            return pd.DataFrame()
        periods = 60
        dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=periods, freq="D")
        close = pd.Series(
            [base_price * (1 + 0.01 * idx / periods) for idx in range(periods)],
            index=dates,
        )
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": dates,
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            }
        )
        return frame


def test_historydata_run_all_strategies(tmp_path):
    pytest.importorskip("pyarrow")
    history_dir = Path("historydata")
    if not history_dir.exists():
        pytest.skip("historydata directory is not available")

    parquet_dir = tmp_path / "local_snapshots"
    symbols = _convert_history_to_parquet(history_dir, parquet_dir)
    if not symbols:
        pytest.skip("No convertible history snapshots found")

    fetcher = LocalDataFetcher(parquet_dir)
    price_map = _build_price_map(parquet_dir, symbols)
    stock_fetcher = _SyntheticStockFetcher(price_map)
    indicator_processor = TechnicalIndicatorProcessor()
    strategies = discover_strategies()
    if not strategies:
        pytest.skip("No strategies were discovered")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    existing = set(results_dir.glob("signals_*.csv"))

    asyncio.run(
        run_once(
            fetcher,
            strategies,
            symbols,
            results_dir,
            enable_gemini=False,
            stock_fetcher=stock_fetcher,
            indicator_processor=indicator_processor,
        )
    )

    generated = sorted(set(results_dir.glob("signals_*.csv")) - existing, key=lambda p: p.stat().st_mtime)
    assert generated, "Local integration run did not produce any signal files"

    latest = generated[-1]
    target = results_dir / "signals_localtest.csv"
    if target.exists():
        target.unlink()
    shutil.copy2(latest, target)

    df = pd.read_csv(target)
    assert not df.empty, "signals_localtest.csv should contain at least one signal"
