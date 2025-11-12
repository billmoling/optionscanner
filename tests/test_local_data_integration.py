import asyncio
import shutil
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from main import discover_strategies
from option_data import LocalDataFetcher
from runner import run_once


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
