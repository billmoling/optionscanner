import math
import os
import time
import unittest
from datetime import datetime
from typing import Tuple

from dotenv import load_dotenv
from ib_async import IB, Option, Stock

from option_data import MARKET_DATA_TYPE_CODES

load_dotenv()


def _is_ibkr_gateway_configured() -> bool:
    required_vars = ("TWS_USERID", "TWS_PASSWORD", "TRADING_MODE")
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(
            "Skipping IBKR option-chain test: set TWS_USERID, TWS_PASSWORD, and TRADING_MODE in your .env file."
        )
        return False
    return True


def _resolve_gateway_endpoint() -> Tuple[str, int]:
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
    _is_ibkr_gateway_configured(),
    "IBKR gateway settings not provided; skipping option-chain integration test.",
)
class IBKROptionChainIntegrationTest(unittest.TestCase):
    def test_fetches_basic_nvda_option_quote(self) -> None:
        host, port = _resolve_gateway_endpoint()
        client_id = int(os.getenv("IAPI_CLIENT_ID", "1"))
        market_data_type = os.getenv("IBKR_MARKET_DATA_TYPE", "FROZEN").upper()
        if market_data_type not in MARKET_DATA_TYPE_CODES:
            raise ValueError("IBKR_MARKET_DATA_TYPE must be 'LIVE' or 'FROZEN'")

        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=5)
            ib.reqMarketDataType(MARKET_DATA_TYPE_CODES[market_data_type])

            stock = Stock("NVDA", "SMART", "USD")
            qualified_stock = ib.qualifyContracts(stock)
            if not qualified_stock:
                raise AssertionError("Unable to qualify NVDA stock contract")
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
                raise AssertionError("Failed to retrieve NVDA underlying price")

            params = ib.reqSecDefOptParams(
                stock.symbol,
                "",
                stock.secType,
                stock.conId,
            )
            if not params:
                raise AssertionError("No option chain metadata returned for NVDA")
            chain = max(params, key=lambda p: len(p.strikes))

            target_expiry = sorted(chain.expirations)[0]
            strikes = sorted(float(strike) for strike in chain.strikes if strike)
            if not strikes:
                raise AssertionError("No strikes returned for NVDA option chain")
            target_strike = min(strikes, key=lambda strike: abs(strike - underlying_price))

            option_contract = Option(
                symbol="NVDA",
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
                raise AssertionError("Unable to qualify NVDA option contract")
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
                raise AssertionError("Did not receive NVDA option market data")

            expiry_display = datetime.strptime(target_expiry, "%Y%m%d").date().isoformat()
        finally:
            if ib.isConnected():
                ib.disconnect()

        expiry = expiry_display
        self.assertGreater(ask or mark, 0.0, "Expected positive ask or mark for NVDA option.")
        print(
            f"NVDA {expiry} CALL bid/ask: {bid:.2f}/{ask:.2f} (mark {mark:.2f}) "
            f"using {market_data_type} market data."
        )


if __name__ == "__main__":
    unittest.main()
