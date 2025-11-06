"""Offline backtesting harness for option strategies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from loguru import logger

from option_data import OptionChainSnapshot
from strategies.base import BaseOptionStrategy


@dataclass(slots=True)
class BacktestMetrics:
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float


class BacktestRunner:
    """Evaluates strategies using stored option data."""

    def __init__(
        self,
        strategies: Iterable[BaseOptionStrategy],
        data_dir: Path,
        results_dir: Path,
    ) -> None:
        self.strategies = list(strategies)
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_snapshots(self) -> List[OptionChainSnapshot]:
        snapshots: List[OptionChainSnapshot] = []
        for file_path in sorted(self.data_dir.glob("*.parquet")):
            df = pd.read_parquet(file_path)
            if df.empty:
                continue
            symbol = df["symbol"].iloc[0]
            timestamp = pd.to_datetime(df["timestamp"].iloc[0])
            options = df.drop(columns=["symbol", "timestamp"]).to_dict("records")
            snapshot = OptionChainSnapshot(
                symbol=symbol,
                underlying_price=float(df["underlying_price"].iloc[0]),
                timestamp=timestamp.to_pydatetime(),
                options=options,
            )
            snapshots.append(snapshot)
        return snapshots

    def run(self) -> None:
        snapshots = self.load_snapshots()
        if not snapshots:
            logger.warning("No data snapshots found for backtesting")
            return
        metrics_rows: List[Dict[str, float]] = []
        for strategy in self.strategies:
            logger.info("Backtesting strategy {name}", name=strategy.name)
            pnl_series = self._simulate_strategy(strategy, snapshots)
            metrics = self._calculate_metrics(pnl_series)
            metrics_rows.append(
                {
                    "strategy": strategy.name,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                }
            )
        df = pd.DataFrame(metrics_rows)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"backtest_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        logger.info("Backtest metrics saved to {path}", path=str(output_path))

    def _simulate_strategy(
        self, strategy: BaseOptionStrategy, snapshots: List[OptionChainSnapshot]
    ) -> pd.Series:
        pnl_records: List[float] = []
        for i in range(len(snapshots) - 1):
            current = snapshots[i]
            future = snapshots[i + 1]
            signals = strategy.on_data([current])
            if not signals:
                continue
            current_df = current.to_pandas().set_index(["expiry", "strike", "option_type"])
            future_df = future.to_pandas().set_index(["expiry", "strike", "option_type"])
            for signal in signals:
                key = (signal.expiry, signal.strike, signal.option_type)
                if key not in current_df.index or key not in future_df.index:
                    continue
                entry = current_df.loc[key]["mark"]
                exit_price = future_df.loc[key]["mark"]
                direction = 1.0
                if "SHORT" in signal.direction:
                    direction = -1.0
                pnl = (exit_price - entry) * direction
                pnl_records.append(pnl)
        if not pnl_records:
            return pd.Series([0.0])
        return pd.Series(pnl_records)

    def _calculate_metrics(self, pnl: pd.Series) -> BacktestMetrics:
        pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
        if pnl.empty:
            return BacktestMetrics(0.0, 0.0, 0.0, 0.0)
        wins = (pnl > 0).sum()
        losses = (pnl < 0).sum()
        win_rate = wins / len(pnl)
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = -pnl[pnl < 0].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf
        sharpe_ratio = pnl.mean() / pnl.std() * math.sqrt(252) if pnl.std() > 0 else 0.0
        cumulative = pnl.cumsum()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max).min()
        return BacktestMetrics(
            win_rate=win_rate,
            profit_factor=float(profit_factor),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(drawdown),
        )


def load_strategies() -> List[BaseOptionStrategy]:
    from main import discover_strategies

    return discover_strategies()


def main() -> None:
    strategies = load_strategies()
    runner = BacktestRunner(strategies, Path("./data"), Path("./results"))
    runner.run()


if __name__ == "__main__":
    main()
