import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest

from option_data import OptionChainSnapshot
from runner import run_once
from strategies.base import TradeSignal


class DummyFetcher:
    async def fetch_all(self, symbols):
        return [
            OptionChainSnapshot(
                symbol=symbol,
                underlying_price=100.0,
                timestamp=datetime.now(timezone.utc),
                options=[],
            )
            for symbol in symbols
        ]


class DummyStrategy:
    def __init__(self) -> None:
        self.name = "DummyStrategy"

    def on_data(self, snapshots):
        signals = []
        for snapshot in snapshots:
            signals.append(
                TradeSignal(
                    symbol=snapshot.symbol,
                    expiry=datetime.now(timezone.utc),
                    strike=100.0,
                    option_type="CALL",
                    direction="LONG_CALL",
                    rationale="Test rationale",
                )
            )
        return signals


def test_run_once_omits_ai_fields_when_disabled(tmp_path):
    fetcher = DummyFetcher()
    strategy = DummyStrategy()
    asyncio.run(
        run_once(
            fetcher,
            [strategy],
            ["NVDA"],
            tmp_path,
            enable_gemini=False,
        )
    )
    files = list(tmp_path.glob("signals_*.csv"))
    assert files, "Expected a signals CSV to be written"
    df = pd.read_csv(files[0])
    assert "explanation" not in df.columns
    assert "validation" not in df.columns
