import asyncio
import math
import os
import unittest
from datetime import datetime, timezone

from dotenv import load_dotenv
from ib_insync import IB, Stock

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
class IBKRDelayedDataIntegrationTest(unittest.TestCase):
    def test_fetches_nvda_delayed_price(self) -> None:
        host, port = _resolve_gateway_endpoint()
        client_id = int(os.getenv("IAPI_CLIENT_ID", "1"))
        market_data_type = os.getenv("IBKR_MARKET_DATA_TYPE", "DELAYED").upper()

        async def _fetch_price() -> tuple[float, datetime]:
            ib = IB()
            try:
                await ib.connectAsync(host, port, clientId=client_id, timeout=5)
                market_code = MARKET_DATA_TYPE_CODES.get(
                    market_data_type, MARKET_DATA_TYPE_CODES["DELAYED"]
                )
                ib.reqMarketDataType(market_code)
                contract = Stock("NVDA", "NASDAQ", "USD")
                await ib.qualifyContractsAsync(contract)
                if hasattr(ib, "reqMktDataAsync"):
                    ticker = await ib.reqMktDataAsync(contract, "", False, False)
                else:
                    ticker = ib.reqMktData(contract, "", False, False)

                price = float("nan")
                for _ in range(20):
                    candidate = float(ticker.last or ticker.close or ticker.marketPrice() or 0.0)
                    if math.isfinite(candidate) and candidate > 0.0:
                        price = candidate
                        break
                    await asyncio.sleep(0.25)

                timestamp = getattr(ticker, "time", None) or datetime.now(timezone.utc)
                return price, timestamp
            finally:
                if ib.isConnected():
                    ib.disconnect()

        price, timestamp = asyncio.run(_fetch_price())

        self.assertGreater(price, 0.0, "Expected NVDA delayed price to be positive.")
        print(
            f"NVDA delayed price: {price:.2f} captured at {timestamp.isoformat()} "
            f"using {market_data_type} market data."
        )


if __name__ == "__main__":
    unittest.main()
