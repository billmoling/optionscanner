import logging
import math
import os
import sys
import time
import unittest
from datetime import datetime

from dotenv import load_dotenv
from ib_async import IB, Option, Stock

from execution import TradeExecutionConfig, TradeExecutor
from option_data import MARKET_DATA_TYPE_CODES, OptionChainSnapshot
from strategies.base import TradeSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

load_dotenv()


def _is_ibkr_gateway_configured() -> bool:
    required_vars = ("TWS_USERID", "TWS_PASSWORD", "TRADING_MODE")
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        log.warning(
            "Skipping IBKR execution test: set TWS_USERID, TWS_PASSWORD, and TRADING_MODE in your .env file."
        )
        return False
    return True


def _resolve_gateway_endpoint() -> tuple[str, int]:
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    explicit_port = os.getenv("IBKR_PORT")
    if explicit_port:
        return host, int(explicit_port)

    trading_mode = os.getenv("TRADING_MODE", "paper").lower()
    if trading_mode not in {"paper", "live"}:
        raise ValueError("TRADING_MODE must be 'paper' or 'live'")
    default_port = 4001 if trading_mode == "live" else 4002
    return host, default_port


@unittest.skipUnless(
    _is_ibkr_gateway_configured(), "IBKR gateway settings not provided; skipping execution integration test."
)
class IBKRExecutionIntegrationTest(unittest.TestCase):
    def test_places_qqq_call_order_via_trade_executor(self) -> None:
        host, port = _resolve_gateway_endpoint()
        client_id = int(os.getenv("IAPI_CLIENT_ID", "1"))
        market_data_type = os.getenv("IBKR_MARKET_DATA_TYPE", "FROZEN").upper()
        if market_data_type not in MARKET_DATA_TYPE_CODES:
            raise ValueError("IBKR_MARKET_DATA_TYPE must be 'LIVE' or 'FROZEN'")

        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=5)
            ib.reqMarketDataType(MARKET_DATA_TYPE_CODES[market_data_type])

            stock = Stock("QQQ", "SMART", "USD")
            qualified_stock = ib.qualifyContracts(stock)
            if not qualified_stock:
                raise AssertionError("Unable to qualify QQQ stock contract")
            stock = qualified_stock[0]
            stock_ticker = ib.reqMktData(stock, "", True, False)

            underlying_price = float("nan")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                candidate = float(stock_ticker.last or stock_ticker.close or stock_ticker.marketPrice() or 0.0)
                if math.isfinite(candidate) and candidate > 0.0:
                    underlying_price = candidate
                    break
                ib.sleep(0.25)
            if not math.isfinite(underlying_price) or underlying_price <= 0.0:
                raise AssertionError("Failed to retrieve QQQ underlying price")

            params = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
            if not params:
                raise AssertionError("No option chain metadata returned for QQQ")
            chain = max(params, key=lambda p: len(p.strikes))

            target_expiry = sorted(chain.expirations)[0]
            strikes = sorted(float(strike) for strike in chain.strikes if strike)
            if not strikes:
                raise AssertionError("No strikes returned for QQQ option chain")
            target_strike = min(strikes, key=lambda strike: abs(strike - underlying_price))

            option_contract = Option(
                symbol="QQQ",
                lastTradeDateOrContractMonth=target_expiry,
                strike=target_strike,
                right="C",
                exchange=chain.exchange or "SMART",
                currency="USD",
            )
            trading_class = getattr(chain, "tradingClass", "") or ""
            if trading_class and option_contract.exchange not in ("SMART", ""):
                option_contract.tradingClass = trading_class

            qualified_options = ib.qualifyContracts(option_contract)
            if not qualified_options:
                raise AssertionError("Unable to qualify QQQ option contract")
            option_contract = qualified_options[0]
            option_ticker = ib.reqMktData(option_contract, "", True, False)

            bid = ask = mark = float("nan")
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                bid = float(option_ticker.bid or 0.0)
                ask = float(option_ticker.ask or 0.0)
                mark = float(option_ticker.midpoint() or 0.0)
                if max(bid, ask, mark) > 0.0:
                    break
                ib.sleep(0.25)

            if max(bid, ask, mark) <= 0.0:
                raise AssertionError("Did not receive QQQ option market data")

            expiry_dt = datetime.strptime(target_expiry, "%Y%m%d")
            snapshot = OptionChainSnapshot(
                symbol="QQQ",
                underlying_price=underlying_price,
                timestamp=datetime.utcnow(),
                options=[
                    {
                        "option_type": "CALL",
                        "strike": target_strike,
                        "expiry": expiry_dt,
                        "bid": bid,
                        "ask": ask,
                        "mark": mark,
                    }
                ],
            )

            signal = TradeSignal(
                symbol="QQQ",
                expiry=expiry_dt,
                strike=target_strike,
                option_type="CALL",
                direction="BUY",
                rationale="Integration test QQQ call buy",
            )

            executor = TradeExecutor(
                ib=ib,
                config=TradeExecutionConfig(
                    enabled=True,
                    default_quantity=1,
                    allow_market_fallback=True,
                    max_spread_pct=1.5,
                ),
            )

            reports = executor.execute_finalists(
                finalists=[("integration", signal, None, None)], snapshots={"QQQ": snapshot}
            )
        finally:
            if ib.isConnected():
                ib.disconnect()

        self.assertEqual(len(reports), 1, "Expected a single execution report for QQQ option order")
        self.assertIn("QQQ", reports[0])
        log.info("QQQ option execution report: %s", reports[0])


if __name__ == "__main__":
    unittest.main()
