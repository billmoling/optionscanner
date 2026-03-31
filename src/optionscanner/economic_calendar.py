"""Economic event calendar tracker for FOMC, CPI, NFP events."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


@dataclass(slots=True)
class EconomicEvent:
    """Represents a scheduled economic event."""

    event_id: str
    date: date
    event_type: str  # FOMC, CPI, NFP, GDP, etc.
    impact: str  # LOW, MEDIUM, HIGH
    description: str
    country: str = "US"  # Default to US events
    actual: Optional[str] = None  # Actual result (filled after event)
    forecast: Optional[str] = None  # Consensus forecast
    previous: Optional[str] = None  # Previous period value

    @property
    def is_high_impact(self) -> bool:
        return self.impact.upper() == "HIGH"


@dataclass
class EconomicCalendarCache:
    """Persistent cache for economic events."""

    events: List[EconomicEvent] = field(default_factory=list)
    last_updated: Optional[str] = None  # ISO timestamp

    def to_dict(self) -> Dict[str, object]:
        return {
            "last_updated": self.last_updated,
            "events": [self._event_to_dict(e) for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EconomicCalendarCache":
        cache = cls()
        cache.last_updated = data.get("last_updated")
        cache.events = [cls._from_event_dict(e) for e in data.get("events", [])]
        return cache

    @staticmethod
    def _event_to_dict(event: EconomicEvent) -> Dict[str, object]:
        return {
            "event_id": event.event_id,
            "date": event.date.isoformat(),
            "event_type": event.event_type,
            "impact": event.impact,
            "description": event.description,
            "country": event.country,
            "actual": event.actual,
            "forecast": event.forecast,
            "previous": event.previous,
        }

    @staticmethod
    def _from_event_dict(data: Dict[str, object]) -> EconomicEvent:
        return EconomicEvent(
            event_id=str(data.get("event_id", "")),
            date=date.fromisoformat(str(data.get("date", ""))),
            event_type=str(data.get("event_type", "")),
            impact=str(data.get("impact", "MEDIUM")),
            description=str(data.get("description", "")),
            country=str(data.get("country", "US")),
            actual=data.get("actual"),
            forecast=data.get("forecast"),
            previous=data.get("previous"),
        )


class EconomicEventTracker:
    """Tracks scheduled economic events (FOMC, CPI, NFP).

    Events are loaded from a JSON file that should be updated weekly.
    This avoids API dependencies for low-frequency economic data.

    Supported event types:
    - FOMC: Federal Open Market Committee rate decisions
    - CPI: Consumer Price Index (inflation)
    - NFP: Non-Farm Payrolls (employment)
    - GDP: Gross Domestic Product
    - PPI: Producer Price Index
    - PCE: Personal Consumption Expenditures
    """

    # Default events file path
    DEFAULT_EVENTS_FILE = Path("data") / "economic_events.json"

    # High-impact event types that trigger warnings
    HIGH_IMPACT_TYPES = {"FOMC", "CPI", "NFP", "PCE", "GDP"}

    # Blackout window (days before high-impact events)
    DEFAULT_BLACKOUT_DAYS = 2

    def __init__(
        self,
        events_file: Optional[Path] = None,
        blackout_days: int = DEFAULT_BLACKOUT_DAYS,
    ) -> None:
        self._events_file = events_file or self.DEFAULT_EVENTS_FILE
        self._blackout_days = blackout_days
        self._cache: Optional[EconomicCalendarCache] = None

        # Ensure directory exists
        self._events_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_events(self) -> EconomicCalendarCache:
        """Load events from JSON file."""
        if self._cache is not None:
            return self._cache

        if not self._events_file.exists():
            # Create empty calendar with template
            self._cache = EconomicCalendarCache()
            self._create_template()
            return self._cache

        try:
            data = json.loads(self._events_file.read_text(encoding="utf-8"))
            self._cache = EconomicCalendarCache.from_dict(data)
            logger.info("Loaded economic calendar | path={path}", path=str(self._events_file))
        except Exception as exc:
            logger.warning("Failed to load economic calendar | reason={error}", error=exc)
            self._cache = EconomicCalendarCache()

        return self._cache

    def _create_template(self) -> None:
        """Create template events file if it doesn't exist."""
        if self._events_file.exists():
            return

        # Create template with example events
        template = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "events": [
                {
                    "event_id": "fomc_2026_mar",
                    "date": "2026-03-19",
                    "event_type": "FOMC",
                    "impact": "HIGH",
                    "description": "FOMC Rate Decision",
                    "forecast": "4.50%",
                    "previous": "4.50%",
                },
                {
                    "event_id": "cpi_2026_mar",
                    "date": "2026-03-12",
                    "event_type": "CPI",
                    "impact": "HIGH",
                    "description": "CPI YoY",
                    "forecast": "2.8%",
                    "previous": "3.0%",
                },
                {
                    "event_id": "nfp_2026_mar",
                    "date": "2026-03-07",
                    "event_type": "NFP",
                    "impact": "HIGH",
                    "description": "Non-Farm Payrolls",
                    "forecast": "185K",
                    "previous": "220K",
                },
            ],
        }

        try:
            self._events_file.parent.mkdir(parents=True, exist_ok=True)
            self._events_file.write_text(json.dumps(template, indent=2), encoding="utf-8")
            logger.info("Created economic events template | path={path}", path=str(self._events_file))
        except Exception as exc:
            logger.warning("Failed to create events template | reason={error}", error=exc)

    def save_events(self) -> None:
        """Save current events to file."""
        cache = self._load_events()
        try:
            cache.last_updated = datetime.now(timezone.utc).isoformat()
            data = cache.to_dict()
            self._events_file.parent.mkdir(parents=True, exist_ok=True)
            self._events_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save economic calendar | reason={error}", error=exc)

    def add_event(self, event: EconomicEvent) -> None:
        """Add a new economic event."""
        cache = self._load_events()
        cache.events.append(event)
        # Sort by date
        cache.events.sort(key=lambda e: e.date)
        self._cache = cache
        logger.info(
            "Added economic event | type={type} date={date} description={desc}",
            type=event.event_type,
            date=event.date,
            desc=event.description,
        )

    def remove_event(self, event_id: str) -> bool:
        """Remove an event by ID."""
        cache = self._load_events()
        initial_count = len(cache.events)
        cache.events = [e for e in cache.events if e.event_id != event_id]

        if len(cache.events) < initial_count:
            self._cache = cache
            logger.info("Removed economic event | id={id}", id=event_id)
            return True
        return False

    def get_upcoming_events(
        self,
        window_days: int = 14,
        event_types: Optional[List[str]] = None,
        min_impact: str = "MEDIUM",
    ) -> List[EconomicEvent]:
        """Get upcoming events within the specified window.

        Args:
            window_days: Look ahead window in days
            event_types: Filter by event types (None = all)
            min_impact: Minimum impact level (LOW, MEDIUM, HIGH)

        Returns:
            List of upcoming events sorted by date
        """
        cache = self._load_events()
        today = date.today()
        end_date = today + timedelta(days=window_days)

        impact_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_impact_level = impact_order.get(min_impact.upper(), 0)

        events = []
        for event in cache.events:
            # Skip past events
            if event.date < today:
                continue

            # Skip events beyond window
            if event.date > end_date:
                continue

            # Filter by event type
            if event_types and event.event_type.upper() not in event_types:
                continue

            # Filter by impact
            if impact_order.get(event.impact.upper(), 0) < min_impact_level:
                continue

            events.append(event)

        events.sort(key=lambda e: e.date)
        return events

    def get_high_impact_events(
        self,
        window_days: int = 14,
    ) -> List[EconomicEvent]:
        """Get high-impact events within window."""
        return self.get_upcoming_events(
            window_days=window_days,
            event_types=list(self.HIGH_IMPACT_TYPES),
            min_impact="HIGH",
        )

    def is_blackout_period(
        self,
        reference_date: Optional[date] = None,
        event_types: Optional[List[str]] = None,
    ) -> bool:
        """Check if we're in a blackout period before high-impact events.

        Args:
            reference_date: Date to check (default: today)
            event_types: Specific event types to check (None = all high-impact)

        Returns:
            True if within blackout window of a high-impact event
        """
        reference = reference_date or date.today()
        events = self.get_high_impact_events(window_days=self._blackout_days * 2)

        if event_types:
            events = [e for e in events if e.event_type.upper() in event_types]

        for event in events:
            days_until = (event.date - reference).days
            if 0 <= days_until <= self._blackout_days:
                logger.info(
                    "Blackout period detected | days={days} event={type} date={date}",
                    days=days_until,
                    type=event.event_type,
                    date=event.date,
                )
                return True

        return False

    def get_next_event(
        self,
        event_type: Optional[str] = None,
        min_impact: str = "HIGH",
    ) -> Optional[EconomicEvent]:
        """Get the next upcoming event."""
        events = self.get_upcoming_events(window_days=90, min_impact=min_impact)

        if event_type:
            events = [e for e in events if e.event_type.upper() == event_type.upper()]

        return events[0] if events else None

    def get_days_to_next_event(
        self,
        event_type: Optional[str] = None,
    ) -> Optional[int]:
        """Get days until next event of specified type."""
        event = self.get_next_event(event_type)
        if event is None:
            return None

        return max((event.date - date.today()).days, 0)

    def get_warning_message(
        self,
        window_days: int = 7,
    ) -> str:
        """Generate a warning message about upcoming events."""
        events = self.get_high_impact_events(window_days=window_days)

        if not events:
            return ""

        parts = []
        for event in events[:5]:  # Limit to 5 events
            days = (event.date - date.today()).days
            if days == 0:
                when = "TODAY"
            elif days == 1:
                when = "TOMORROW"
            else:
                when = f"in {days}d"

            parts.append(f"{event.event_type} {when} ({event.description})")

        return "Upcoming events: " + ", ".join(parts)

    def get_events_summary(
        self,
        window_days: int = 14,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Get a summary of events by type."""
        events = self.get_upcoming_events(window_days=window_days, min_impact="MEDIUM")

        summary: Dict[str, List[Dict[str, str]]] = {}
        for event in events:
            if event.event_type not in summary:
                summary[event.event_type] = []

            days = (event.date - date.today()).days
            summary[event.event_type].append({
                "date": event.date.isoformat(),
                "description": event.description,
                "days_until": str(days),
                "impact": event.impact,
            })

        return summary

    def refresh_reminder(self) -> str:
        """Check if calendar needs refreshing and return reminder message."""
        cache = self._load_events()

        if not cache.last_updated:
            return "Economic calendar not initialized. Please update with upcoming events."

        try:
            last_updated = datetime.fromisoformat(cache.last_updated)
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=timezone.utc)

            days_since = (datetime.now(timezone.utc) - last_updated).days

            if days_since >= 7:
                return f"Economic calendar last updated {days_since} days ago. Consider refreshing with new events."
        except Exception:
            pass

        return ""


__all__ = [
    "EconomicEventTracker",
    "EconomicEvent",
    "EconomicCalendarCache",
]
