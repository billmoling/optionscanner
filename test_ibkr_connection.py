#!/usr/bin/env python3
"""Simple IBKR connection test script for Docker container.

Usage:
    python test_ibkr_connection.py

This script tests the connection to IBKR Gateway from within the
algo-trade-app Docker container. It uses the same connection logic
as the main application.

Environment variables (or .env file):
    IBKR_HOST: Hostname (default: 'ib-gateway' in Docker, '127.0.0.1' otherwise)
    IBKR_PORT: Port (default: based on TRADING_MODE: 4002 for paper, 4001 for live)
    TRADING_MODE: 'paper' or 'live' (default: 'paper')
    TWS_USERID: IBKR username
    TWS_PASSWORD: IBKR password
"""
import os
import sys
import time
import math
from datetime import datetime, timezone

from dotenv import load_dotenv
from ib_async import IB, Stock

load_dotenv()


def _is_ibkr_configured() -> bool:
    """Check if IBKR credentials are configured."""
    required = ("TWS_USERID", "TWS_PASSWORD", "TRADING_MODE")
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"[WARN] Missing env vars: {missing}")
        print(f"[INFO] Set TWS_USERID, TWS_PASSWORD, TRADING_MODE in .env file")
        return False
    return True


def _resolve_gateway_endpoint() -> tuple[str, int]:
    """Resolve IBKR host and port from environment or defaults."""
    host = os.getenv("IBKR_HOST", "ib-gateway")
    explicit_port = os.getenv("IBKR_PORT")

    if explicit_port:
        return host, int(explicit_port)

    trading_mode = os.getenv("TRADING_MODE", "paper").lower()
    if trading_mode not in {"paper", "live"}:
        raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got '{trading_mode}'")

    # IBKR gateway ports in docker-compose:
    # - 4002 (paper) -> container: 4004
    # - 4001 (live) -> container: 4003
    default_port = 4003 if trading_mode == "live" else 4004
    return host, default_port


def test_connection() -> bool:
    """Test connection to IBKR Gateway and fetch NVDA price."""
    host, port = _resolve_gateway_endpoint()
    client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))
    market_data_type = os.getenv("IBKR_MARKET_DATA_TYPE", "FROZEN").upper()

    print(f"[INFO] Connecting to IBKR...")
    print(f"[INFO] Host: {host}")
    print(f"[INFO] Port: {port}")
    print(f"[INFO] Client ID: {client_id}")
    print(f"[INFO] Market Data: {market_data_type}")

    ib = IB()
    try:
        # Connect with timeout
        print(f"[INFO] Establishing connection...")
        ib.connect(host, port, clientId=client_id, timeout=10)

        if not ib.isConnected():
            print("[ERROR] Failed to connect to IBKR Gateway")
            return False

        print("[OK] Connected to IBKR Gateway")

        # Request market data type
        from option_data import MARKET_DATA_TYPE_CODES
        market_data_code = MARKET_DATA_TYPE_CODES.get(market_data_type, 2)
        ib.reqMarketDataType(market_data_code)
        print(f"[INFO] Market data type set to {market_data_type}")

        # Qualify NVDA contract
        print("[INFO] Qualifying NVDA contract...")
        contract = Stock("NVDA", "NASDAQ", "USD")
        qualified = ib.qualifyContracts(contract)

        if not qualified:
            print("[ERROR] Failed to qualify NVDA contract")
            return False

        print(f"[OK] Contract qualified: {qualified[0]}")

        # Request market data
        print("[INFO] Requesting market data...")
        ticker = ib.reqMktData(qualified[0], "", True, False)

        # Wait for price
        price = float("nan")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            candidate = float(ticker.last or ticker.close or ticker.marketPrice() or 0.0)
            if math.isfinite(candidate) and candidate > 0.0:
                price = candidate
                break
            ib.sleep(0.5)

        timestamp = getattr(ticker, "time", None) or datetime.now(timezone.utc)

        if not math.isfinite(price) or price <= 0:
            print("[ERROR] No valid price received for NVDA")
            return False

        print(f"[OK] NVDA price: ${price:.2f}")
        print(f"[INFO] Price timestamp: {timestamp.isoformat()}")
        print("[OK] IBKR connection test PASSED")
        return True

    except ConnectionRefusedError as e:
        print(f"[ERROR] Connection refused: {e}")
        print("[HINT] Ensure ib-gateway container is running: docker compose ps")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False
    finally:
        if ib.isConnected():
            print("[INFO] Disconnecting...")
            ib.disconnect()


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("IBKR Connection Test")
    print("=" * 60)

    if not _is_ibkr_configured():
        print("[ERROR] IBKR not configured. Exiting.")
        return 1

    success = test_connection()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
