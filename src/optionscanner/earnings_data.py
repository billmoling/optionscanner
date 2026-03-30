"""Earnings calendar data fetcher using Financial Modeling Prep API."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from loguru import logger


@dataclass(slots=True)
class EarningsEvent:
    """Represents a single earnings announcement."""

    symbol: str
    report_date: date
    report_time: str  # "before_market", "after_market", "during_market"
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    confirmed: bool = False  # False = tentative date


@dataclass
class EarningsCache:
    """Persistent cache for earnings data."""

    events: Dict[str, List[EarningsEvent]] = field(default_factory=dict)
    last_fetch: Optional[str] = None  # ISO timestamp
    cache_version: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "cache_version": self.cache_version,
            "last_fetch": self.last_fetch,
            "events": {
                symbol: [self._event_to_dict(e) for e in events]
                for symbol, events in self.events.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EarningsCache":
        cache = cls()
        cache.cache_version = int(data.get("cache_version", 1))
        cache.last_fetch = data.get("last_fetch")
        cache.events = {
            symbol: [cls._from_event_dict(e) for e in events]
            for symbol, events in data.get("events", {}).items()
        }
        return cache

    @staticmethod
    def _event_to_dict(event: EarningsEvent) -> Dict[str, object]:
        return {
            "symbol": event.symbol,
            "report_date": event.report_date.isoformat(),
            "report_time": event.report_time,
            "eps_estimate": event.eps_estimate,
            "eps_actual": event.eps_actual,
            "revenue_estimate": event.revenue_estimate,
            "confirmed": event.confirmed,
        }

    @staticmethod
    def _from_event_dict(data: Dict[str, object]) -> EarningsEvent:
        return EarningsEvent(
            symbol=str(data.get("symbol", "")),
            report_date=date.fromisoformat(str(data.get("report_date", ""))),
            report_time=str(data.get("report_time", "before_market")),
            eps_estimate=float(data["eps_estimate"]) if data.get("eps_estimate") else None,
            eps_actual=float(data["eps_actual"]) if data.get("eps_actual") else None,
            revenue_estimate=float(data["revenue_estimate"]) if data.get("revenue_estimate") else None,
            confirmed=bool(data.get("confirmed", False)),
        )


class EarningsFetcher:
    """Fetches earnings calendar data from Financial Modeling Prep API.

    Free tier limits:
    - 250 requests/day
    - 30 calls/minute

    API docs: https://financialmodelingprep.com/developer/docs/earnings-calendar/
    """

    # FMP API endpoints
    EARNINGS_CALENDAR_URL = "https://financialmodelingprep.com/api/v3/earnings_calendar"
    EARNINGS_CONFIRMED_URL = "https://financialmodelingprep.com/api/v3/earnings_calendar/confirmed"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_path: Optional[Path] = None,
        cache_ttl_hours: int = 24,
    ) -> None:
        self._api_key = self._resolve_api_key(api_key)
        self._cache_path = cache_path or Path("data") / "earnings_cache.json"
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cache: Optional[EarningsCache] = None
        self._rate_limit_delay = 2.0  # seconds between requests

        # Ensure cache directory exists
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_api_key(self, api_key: Optional[str]) -> str:
        """Resolve API key from parameter, env var, or .env file."""
        if api_key:
            return api_key

        # Try environment variable
        env_key = os.getenv("FMP_API_KEY", "").strip()
        if env_key:
            return env_key

        # Try .env file
        env_path = Path(".env")
        if env_path.exists():
            try:
                content = env_path.read_text()
                for line in content.splitlines():
                    if line.startswith("FMP_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        if key:
                            return key
            except Exception:
                pass

        return ""

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self._api_key)

    def _load_cache(self) -> EarningsCache:
        """Load earnings cache from disk."""
        if self._cache is not None:
            return self._cache

        if not self._cache_path.exists():
            self._cache = EarningsCache()
            return self._cache

        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._cache = EarningsCache.from_dict(data)
            logger.info("Loaded earnings cache | path={path}", path=str(self._cache_path))
        except Exception as exc:
            logger.warning("Failed to load earnings cache | reason={error}", error=exc)
            self._cache = EarningsCache()

        return self._cache

    def _save_cache(self) -> None:
        """Save earnings cache to disk."""
        if self._cache is None:
            return

        try:
            data = self._cache.to_dict()
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save earnings cache | reason={error}", error=exc)

    def _is_cache_fresh(self) -> bool:
        """Check if cache is still fresh."""
        cache = self._load_cache()
        if not cache.last_fetch:
            return False

        try:
            last_fetch = datetime.fromisoformat(cache.last_fetch)
            if last_fetch.tzinfo is None:
                last_fetch = last_fetch.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - last_fetch) < self._cache_ttl
        except Exception:
            return False

    def fetch_earnings(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> List[EarningsEvent]:
        """Fetch earnings calendar for a single symbol."""
        if not self._api_key:
            logger.warning("FMP API key not configured; cannot fetch earnings")
            return []

        from_date = from_date or date.today()
        to_date = to_date or (date.today() + timedelta(days=60))

        params = {
            "symbol": symbol.upper(),
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "apikey": self._api_key,
        }

        try:
            time.sleep(self._rate_limit_delay)  # Rate limiting
            response = requests.get(self.EARNINGS_CALENDAR_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            events = self._parse_earnings_response(data, symbol)
            self._update_cache(symbol, events)

            logger.info(
                "Fetched earnings calendar | symbol={symbol} count={count}",
                symbol=symbol,
                count=len(events),
            )
            return events

        except requests.exceptions.RequestException as exc:
            logger.exception("Failed to fetch earnings | symbol={symbol}", symbol=symbol)
            return self._get_cached_events(symbol)
        except Exception as exc:
            logger.exception("Unexpected error fetching earnings | symbol={symbol}", symbol=symbol)
            return self._get_cached_events(symbol)

    def fetch_earnings_many(
        self,
        symbols: List[str],
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
    ) -> Dict[str, List[EarningsEvent]]:
        """Fetch earnings calendar for multiple symbols sequentially."""
        results: Dict[str, List[EarningsEvent]] = {}

        for symbol in symbols:
            try:
                events = self.fetch_earnings(symbol, from_date, to_date)
                results[symbol.upper()] = events
                # Rate limiting: 30 calls/minute = 2 second delay
                time.sleep(self._rate_limit_delay)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch earnings for symbol | symbol={symbol} error={error}",
                    symbol=symbol,
                    error=exc,
                )
                results[symbol.upper()] = self._get_cached_events(symbol)

        return results

    def _parse_earnings_response(
        self, data: object, symbol: str
    ) -> List[EarningsEvent]:
        """Parse FMP API response into EarningsEvent objects."""
        if not isinstance(data, list):
            return []

        events = []
        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                report_date_str = str(item.get("date", ""))
                if not report_date_str:
                    continue

                report_date = date.fromisoformat(report_date_str)

                # Determine report time
                time_str = str(item.get("time", "")).lower()
                if "after" in time_str:
                    report_time = "after_market"
                elif "before" in time_str:
                    report_time = "before_market"
                else:
                    report_time = "before_market"  # Default

                events.append(
                    EarningsEvent(
                        symbol=symbol.upper(),
                        report_date=report_date,
                        report_time=report_time,
                        eps_estimate=float(item["epsEstimated"]) if item.get("epsEstimated") else None,
                        eps_actual=float(item["epsActual"]) if item.get("epsActual") else None,
                        revenue_estimate=float(item["revenueEstimated"]) if item.get("revenueEstimated") else None,
                        confirmed=bool(item.get("confirmed", False)),
                    )
                )
            except Exception as exc:
                logger.debug("Failed to parse earnings item | item={item} error={error}", item=item, error=exc)

        return events

    def _update_cache(self, symbol: str, events: List[EarningsEvent]) -> None:
        """Update cache with new earnings data."""
        cache = self._load_cache()
        if events:
            cache.events[symbol.upper()] = events
        cache.last_fetch = datetime.now(timezone.utc).isoformat()
        self._cache = cache
        self._save_cache()

    def _get_cached_events(self, symbol: str) -> List[EarningsEvent]:
        """Get cached events for a symbol."""
        cache = self._load_cache()
        return cache.events.get(symbol.upper(), [])

    def get_earnings_date(
        self,
        symbol: str,
        forward_days: int = 30,
    ) -> Optional[date]:
        """Get next earnings date for a symbol."""
        today = date.today()
        to_date = today + timedelta(days=forward_days)

        events = self.fetch_earnings(symbol, today, to_date)
        if not events:
            return None

        # Return earliest upcoming earnings
        upcoming = [e for e in events if e.report_date >= today]
        if not upcoming:
            return None

        upcoming.sort(key=lambda e: e.report_date)
        return upcoming[0].report_date

    def is_pre_earnings(
        self,
        symbol: str,
        window_days: int = 5,
    ) -> bool:
        """Check if symbol is within pre-earnings window."""
        today = date.today()
        from_date = today
        to_date = today + timedelta(days=window_days)

        events = self.fetch_earnings(symbol, from_date, to_date)
        return len(events) > 0

    def get_earnings_phase(
        self,
        symbol: str,
        window_days: int = 5,
    ) -> str:
        """Get earnings phase for a symbol.

        Returns:
            "PRE" - Within pre-earnings window
            "POST" - Just reported (within last 2 days)
            "NONE" - No upcoming earnings
        """
        today = date.today()

        # Check for upcoming earnings
        if self.is_pre_earnings(symbol, window_days):
            return "PRE"

        # Check if just reported (look backward)
        from_date = today - timedelta(days=2)
        to_date = today

        events = self.fetch_earnings(symbol, from_date, to_date)
        if events:
            return "POST"

        return "NONE"

    def get_days_to_earnings(
        self,
        symbol: str,
    ) -> Optional[int]:
        """Get days until next earnings report."""
        today = date.today()
        earnings_date = self.get_earnings_date(symbol, forward_days=60)

        if earnings_date is None:
            return None

        return max((earnings_date - today).days, 0)

    def get_upcoming_earnings(
        self,
        symbols: List[str],
        window_days: int = 10,
    ) -> Dict[str, int]:
        """Get days to earnings for multiple symbols.

        Returns:
            Dict mapping symbol to days until earnings (only symbols with upcoming earnings)
        """
        results: Dict[str, int] = {}

        for symbol in symbols:
            days = self.get_days_to_earnings(symbol)
            if days is not None and days <= window_days:
                results[symbol.upper()] = days

        return dict(sorted(results.items(), key=lambda x: x[1]))


__all__ = [
    "EarningsFetcher",
    "EarningsEvent",
    "EarningsCache",
]
