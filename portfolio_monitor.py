"""Daily portfolio monitoring using NautilusTrader + IBKR."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from ib_insync import IB, Contract, Option
from loguru import logger

from logging_utils import configure_logging


class PortfolioMonitor:
    """Polls IBKR for open positions and evaluates risk metrics."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.log_dir = Path(config.get("log_dir", "./logs")) / "portfolio"
        configure_logging(self.log_dir, "portfolio_monitor")
        self.results_dir = Path("./results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._ib = IB()

    async def connect(self) -> None:
        if self._ib.isConnected():
            return
        await self._ib.connectAsync(
            self.config["ibkr"]["host"],
            self.config["ibkr"]["port"],
            clientId=self.config["ibkr"].get("client_id", 1) + 1,
            timeout=5,
        )
        if not self._ib.isConnected():
            raise ConnectionError("Portfolio monitor could not connect to IBKR")
        logger.info("Connected to IBKR for portfolio monitoring")

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        await self.connect()
        positions = await self._ib.reqPositionsAsync()
        summaries: List[Dict[str, Any]] = []
        tickers = []
        for account, contract, position, avg_cost in positions:
            market_price = await self._ensure_market_price(contract)
            pnl = (market_price - avg_cost) * position * contract.multiplier if isinstance(contract, Option) else (market_price - avg_cost) * position
            delta = theta = 0.0
            days_to_expiry = None
            if isinstance(contract, Option):
                ticker = await self._ib.reqMktDataAsync(contract, "", False, False)
                if ticker.modelGreeks:
                    delta = ticker.modelGreeks.delta or 0.0
                    theta = ticker.modelGreeks.theta or 0.0
                expiry_dt = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")
                days_to_expiry = (expiry_dt - datetime.utcnow()).days
            summaries.append(
                {
                    "account": account,
                    "symbol": contract.symbol,
                    "position": position,
                    "avg_cost": avg_cost,
                    "market_price": market_price,
                    "pnl": pnl,
                    "delta": delta,
                    "theta": theta,
                    "days_to_expiry": days_to_expiry,
                    "expiry": contract.lastTradeDateOrContractMonth if isinstance(contract, Option) else None,
                    "contract": contract,
                }
            )
        return summaries

    async def _ensure_market_price(self, contract: Contract) -> float:
        if isinstance(contract, Option):
            await self._ib.qualifyContractsAsync(contract)
        ticker = await self._ib.reqMktDataAsync(contract, "", False, False)
        price = ticker.last or ticker.close or ticker.marketPrice()
        return float(price or 0.0)

    def evaluate_actions(self, position: Dict[str, Any]) -> str:
        pnl = position["pnl"]
        days = position.get("days_to_expiry")
        delta = position.get("delta", 0.0)
        theta = position.get("theta", 0.0)
        if days is None:
            return "HOLD"
        if days <= 3 or abs(pnl) > 500:
            return "ROLL" if pnl > 0 else "CLOSE"
        if abs(delta) > 0.5:
            return "ADJUST_DELTA"
        if theta < -5:
            return "REDUCE_THETA"
        return "HOLD"

    async def run(self) -> None:
        positions = await self.fetch_positions()
        if not positions:
            logger.info("No open positions found")
            return
        for pos in positions:
            suggestion = self.evaluate_actions(pos)
            logger.info(
                "Position | symbol={symbol} qty={qty} pnl={pnl:.2f} delta={delta:.2f} theta={theta:.2f} suggestion={suggestion}",
                symbol=pos["symbol"],
                qty=pos["position"],
                pnl=pos["pnl"],
                delta=pos.get("delta", 0.0),
                theta=pos.get("theta", 0.0),
                suggestion=suggestion,
            )
            pos["suggestion"] = suggestion
        df = pd.DataFrame(positions)
        df["timestamp"] = datetime.utcnow()
        csv_path = self.results_dir / f"portfolio_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Portfolio summary saved to {path}", path=str(csv_path))


async def main() -> None:
    config = load_config(Path("config.yaml"))
    monitor = PortfolioMonitor(config)
    try:
        await monitor.run()
    finally:
        await monitor.disconnect()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    asyncio.run(main())
