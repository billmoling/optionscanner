#!/usr/bin/env uv run
"""Refresh economic calendar data using Google Gemini AI.

Run this script manually to update the economic events calendar,
or it will run automatically as part of the main scanner run (weekly).

Usage:
    uv run refresh_economic_calendar.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from optionscanner.economic_calendar_ai import EconomicCalendarAIFetcher
from loguru import logger

def main() -> int:
    """Refresh economic calendar and return exit code."""
    logger.info("Starting economic calendar refresh...")

    fetcher = EconomicCalendarAIFetcher(
        events_file=Path("data") / "economic_events.json",
        lookback_days=7,
        lookahead_days=90,
    )

    count = fetcher.refresh_calendar()

    if count > 0:
        logger.info("Successfully refreshed economic calendar | events={count}", count=count)
        return 0
    else:
        logger.warning("No new events fetched")
        return 1


if __name__ == "__main__":
    sys.exit(main())
