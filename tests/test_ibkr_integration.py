import math
import os
import time
import unittest
from datetime import datetime, timezone

from dotenv import load_dotenv
from ib_async import IB, Stock

from option_data import MARKET_DATA_TYPE_CODES

load_dotenv()


def _is_ibkr_gateway_configured() -> bool:
    required_vars = ("TWS_USERID", "TWS_PASSWORD", "TRADING_MODE")
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(
            "Skipping IBKR integration test: set TWS_USERID, TWS_PASSWORD, and TRADING_MODE in your .env file."
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
    _is_ibkr_gateway_configured(), "IBKR gateway settings not provided; skipping integration test."
)
class IBKRMarketDataIntegrationTest(unittest.TestCase):
    def test_fetches_nvda_price(self) -> None:
        host, port = _resolve_gateway_endpoint()
        client_id = int(os.getenv("IAPI_CLIENT_ID", "1"))
        market_data_type = os.getenv("IBKR_MARKET_DATA_TYPE", "FROZEN").upper()
        if market_data_type not in MARKET_DATA_TYPE_CODES:
            raise ValueError("IBKR_MARKET_DATA_TYPE must be 'LIVE' or 'FROZEN'")

        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=5)
            ib.reqMarketDataType(MARKET_DATA_TYPE_CODES[market_data_type])
            contract = Stock("NVDA", "NASDAQ", "USD")
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, "", False, False)

            price = float("nan")
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                candidate = float(ticker.last or ticker.close or ticker.marketPrice() or 0.0)
                if math.isfinite(candidate) and candidate > 0.0:
                    price = candidate
                    break
                ib.sleep(0.25)

            timestamp = getattr(ticker, "time", None) or datetime.now(timezone.utc)
        finally:
            if ib.isConnected():
                ib.disconnect()

        self.assertGreater(price, 0.0, "Expected NVDA price to be positive.")
        print(
            f"NVDA price: {price:.2f} captured at {timestamp.isoformat()} "
            f"using {market_data_type} market data."
        )


if __name__ == "__main__":
    unittest.main()
