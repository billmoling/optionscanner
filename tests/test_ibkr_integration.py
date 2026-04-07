import logging
import math
import os
import sys
import time
import unittest
from datetime import datetime, timezone

from dotenv import load_dotenv
from ib_async import IB, Stock

from optionscanner.market_hours import MarketHoursChecker
from optionscanner.option_data import MARKET_DATA_TYPE_CODES

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


def _get_client_id() -> int:
    """Get client ID from environment, with fallback to default of 1."""
    client_id_str = os.getenv("IAPI_CLIENT_ID") or os.getenv("IBKR_CLIENT_ID")
    if not client_id_str:
        return 1
    try:
        return int(client_id_str)
    except ValueError:
        log.warning("Invalid IAPI_CLIENT_ID '%s', defaulting to 1", client_id_str)
        return 1


def _resolve_market_data_type(requested_type: str) -> str:
    """Resolve the market data type, handling AUTO mode.

    Args:
        requested_type: The requested market data type (LIVE, FROZEN, or AUTO).

    Returns:
        The resolved market data type (LIVE or FROZEN).
    """
    if requested_type.upper() == "AUTO":
        checker = MarketHoursChecker()
        resolved = checker.get_market_data_type()
        log.info(
            "AUTO market data type resolved to %s",
            resolved,
        )
        return resolved
    return requested_type.upper()


@unittest.skipUnless(
    _is_ibkr_gateway_configured(), "IBKR gateway settings not provided; skipping integration test."
)
class IBKRMarketDataIntegrationTest(unittest.TestCase):
    def test_fetches_nvda_price(self) -> None:
        host, port = _resolve_gateway_endpoint()
        client_id = _get_client_id()
        market_data_type_raw = os.getenv("IBKR_MARKET_DATA_TYPE", "FROZEN").upper()
        # Resolve AUTO mode to actual type
        market_data_type = _resolve_market_data_type(market_data_type_raw)

        if market_data_type not in MARKET_DATA_TYPE_CODES or MARKET_DATA_TYPE_CODES.get(market_data_type) is None:
            raise ValueError("IBKR_MARKET_DATA_TYPE must be 'LIVE', 'FROZEN', or 'AUTO'")

        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=5)
            ib.reqMarketDataType(MARKET_DATA_TYPE_CODES[market_data_type])
            contract = Stock("NVDA", "NASDAQ", "USD")
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, "", True, False)

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
        log.info(
            f"NVDA price: {price:.2f} captured at {timestamp.isoformat()} "
            f"using {market_data_type} market data (requested: {market_data_type_raw})."
        )


if __name__ == "__main__":
    unittest.main()
